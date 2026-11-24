# =====================================================================================
# SpoilCheck - Fusion v2 + Ablation Study
# Paste as a NEW cell at the very END of the Colab notebook (after the fusion cell).
#
# Fixes vs. the original fusion cell:
#   1. "Sensor unplugged" now zeros the SAME tensor at test time that modality dropout
#      zeros during training (the projected features h_ir / h_gas), not the raw input.
#   2. Modality dropout is per-sample instead of per-batch.
#   3. Best epoch is chosen by validation LOSS (not accuracy, which saturates at ~98%
#      and froze the old checkpoint at epoch 2 of 60).
#   4. Fair ablation: the SAME fusion head is trained on sRGB only / +IR / +Gas / all,
#      over 3 seeds, so you can report mean +/- std.
#
# Does NOT overwrite any existing model. Saves:
#   best_multimodal_fusion_v2.pth   (same layer names as v1 -> loads into the same class)
#   fusion_ablation_results.csv
#   fusion_embeddings_cache.pt      (cached encoder outputs, so re-runs take ~1 minute)
# =====================================================================================
import os, json, copy, joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import f1_score, recall_score, classification_report
from tqdm.auto import tqdm

SAVE_DIR = "/content/drive/MyDrive/food_spoilage_model"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = device.type == "cuda"

# ---------- 1. Shared splits, classes, scalers (produced by earlier cells) ----------
mm_train = pd.read_csv(os.path.join(SAVE_DIR, "mm_train.csv"))
mm_val   = pd.read_csv(os.path.join(SAVE_DIR, "mm_val.csv"))
mm_test  = pd.read_csv(os.path.join(SAVE_DIR, "mm_test.csv"))

with open(os.path.join(SAVE_DIR, "class_indices.json")) as f:
    class_to_idx = json.load(f)
classes = [k for k, _ in sorted(class_to_idx.items(), key=lambda x: x[1])]
NUM_CLASSES = len(classes)

scalers = joblib.load(os.path.join(SAVE_DIR, "sensor_scalers.pkl"))
temp_scaler, gas_scaler = scalers["temp_scaler"], scalers["gas_scaler"]
temp_cols = ["t_min", "t_max", "t_diff"]
gas_cols  = ["adc", "rs", "rs_r0", "methane_ppm"]

# ---------- 2. Encoder definitions (identical to the notebook so weights load) ----------
class StandaloneIRModel(nn.Module):
    def __init__(self, num_classes=12):
        super().__init__()
        self.backbone = timm.create_model("resnet18", pretrained=False, num_classes=0)  # 512-d
        self.temp_net = nn.Sequential(nn.Linear(3, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU())
        self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(512 + 32, 128), nn.ReLU(),
                                  nn.Linear(128, num_classes))

    def forward(self, ir_img, temps, return_features=False):
        fused = torch.cat([self.backbone(ir_img), self.temp_net(temps)], dim=1)
        return fused if return_features else self.head(fused)

class StandaloneGasMLP(nn.Module):
    def __init__(self, in_dim=4, out_dim=12):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.2),
                                 nn.Linear(64, 64), nn.BatchNorm1d(64), nn.ReLU())
        self.head = nn.Linear(64, out_dim)

    def forward(self, x, return_features=False):
        feat = self.net(x)
        return feat if return_features else self.head(feat)

# ---------- 3. Extract (or load cached) frozen embeddings ----------
CACHE = os.path.join(SAVE_DIR, "fusion_embeddings_cache.pt")

if os.path.exists(CACHE):
    E = torch.load(CACHE)
    print("Loaded cached embeddings from", CACHE)
else:
    assert os.path.exists(mm_train.loc[0, "srgb_path"]), \
        "Dataset images not found. Run the FIRST cell (mount Drive + unzip) and then re-run this cell."

    srgb_enc = timm.create_model("convnext_tiny", pretrained=False, num_classes=NUM_CLASSES)
    srgb_enc.load_state_dict(torch.load(os.path.join(SAVE_DIR, "best_convnext_spoilage.pth"), map_location="cpu"))
    srgb_enc.reset_classifier(0)                                   # -> 768-d
    ir_enc = StandaloneIRModel(NUM_CLASSES)
    ir_enc.load_state_dict(torch.load(os.path.join(SAVE_DIR, "best_ir_model.pth"), map_location="cpu"))
    gas_enc = StandaloneGasMLP(4, NUM_CLASSES)
    gas_enc.load_state_dict(torch.load(os.path.join(SAVE_DIR, "best_gas_model.pth"), map_location="cpu"))
    for m in (srgb_enc, ir_enc, gas_enc):
        m.to(device).eval()

    eval_tfms = transforms.Compose([
        transforms.Resize((224, 224)), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    class PairedRawDataset(Dataset):
        def __init__(self, df):
            self.df = df
            self.temps = torch.tensor(temp_scaler.transform(df[temp_cols].values), dtype=torch.float32)
            self.gas   = torch.tensor(gas_scaler.transform(df[gas_cols].values), dtype=torch.float32)
        def __len__(self):
            return len(self.df)
        def __getitem__(self, i):
            srgb = eval_tfms(Image.open(self.df.loc[i, "srgb_path"]).convert("RGB"))
            ir   = eval_tfms(Image.open(self.df.loc[i, "ir_path"]).convert("RGB"))
            return srgb, ir, self.temps[i], self.gas[i], int(self.df.loc[i, "label"])

    @torch.no_grad()
    def extract(df, desc):
        loader = DataLoader(PairedRawDataset(df), batch_size=64, shuffle=False, num_workers=0, pin_memory=True)
        out = {"srgb": [], "ir": [], "gas": [], "y": []}
        for srgb, ir, temps, gas, y in tqdm(loader, desc=desc, leave=False):
            srgb, ir, temps, gas = srgb.to(device), ir.to(device), temps.to(device), gas.to(device)
            with torch.autocast(device_type=device.type, enabled=USE_AMP):
                out["srgb"].append(srgb_enc(srgb).float().cpu())
                out["ir"].append(ir_enc(ir, temps, return_features=True).float().cpu())
                out["gas"].append(gas_enc(gas, return_features=True).float().cpu())
            out["y"].append(y)
        return {k: torch.cat(v) for k, v in out.items()}

    E = {"train": extract(mm_train, "Train"), "val": extract(mm_val, "Val"), "test": extract(mm_test, "Test")}
    torch.save(E, CACHE)
    print("Saved embedding cache to", CACHE)

D = {split: {k: v.to(device) for k, v in E[split].items()} for split in E}
y_te_np = E["test"]["y"].numpy()
print({s: len(E[s]["y"]) for s in E}, "| dims:", {k: E["train"][k].shape[1] for k in ("srgb", "ir", "gas")})

# ---------- 4. Fusion network v2 (same layer names as v1) ----------
class MultimodalFusionNet(nn.Module):
    def __init__(self, num_classes=12, p_mod_drop=0.0):
        super().__init__()
        self.proj_srgb = nn.Sequential(nn.Linear(768, 256), nn.LayerNorm(256), nn.ReLU(), nn.Dropout(0.3))
        self.proj_ir   = nn.Sequential(nn.Linear(544, 128), nn.LayerNorm(128), nn.ReLU(), nn.Dropout(0.3))
        self.proj_gas  = nn.Sequential(nn.Linear(64, 64),   nn.LayerNorm(64),  nn.ReLU(), nn.Dropout(0.2))
        self.fusion_head = nn.Sequential(nn.Linear(256 + 128 + 64, 192), nn.BatchNorm1d(192), nn.ReLU(),
                                         nn.Dropout(0.3), nn.Linear(192, num_classes))
        self.p_mod_drop = p_mod_drop

    def forward(self, e_srgb, e_ir, e_gas, use_ir=True, use_gas=True):
        h_srgb, h_ir, h_gas = self.proj_srgb(e_srgb), self.proj_ir(e_ir), self.proj_gas(e_gas)
        # Missing sensor -> zero the PROJECTED features (exactly what training dropout does).
        # Use these same flags in the live backend when a sensor is unplugged.
        if not use_ir:
            h_ir = torch.zeros_like(h_ir)
        if not use_gas:
            h_gas = torch.zeros_like(h_gas)
        # Per-sample modality dropout (training only)
        if self.training and self.p_mod_drop > 0:
            B = h_srgb.size(0)
            h_ir  = h_ir  * (torch.rand(B, 1, device=h_ir.device)  >= self.p_mod_drop).float()
            h_gas = h_gas * (torch.rand(B, 1, device=h_gas.device) >= self.p_mod_drop).float()
        return self.fusion_head(torch.cat([h_srgb, h_ir, h_gas], dim=1))

# ---------- 5. Training (best epoch chosen by validation loss) ----------
counts = torch.bincount(E["train"]["y"], minlength=NUM_CLASSES).float()
class_weights = (1.0 / counts.clamp(min=1))
class_weights = (class_weights / class_weights.sum() * NUM_CLASSES).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)

def train_fusion(use_ir, use_gas, p_mod_drop, seed, epochs=60, bs=128):
    torch.manual_seed(seed); np.random.seed(seed)
    tr, va = D["train"], D["val"]
    model = MultimodalFusionNet(NUM_CLASSES, p_mod_drop).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)
    n = len(tr["y"])
    best_loss, best_state, best_ep = float("inf"), None, -1
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            if len(idx) < 2:          # BatchNorm needs >1 sample
                continue
            opt.zero_grad()
            loss = criterion(model(tr["srgb"][idx], tr["ir"][idx], tr["gas"][idx], use_ir, use_gas), tr["y"][idx])
            loss.backward()
            opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            vloss = F.cross_entropy(model(va["srgb"], va["ir"], va["gas"], use_ir, use_gas), va["y"]).item()
        if vloss < best_loss:
            best_loss, best_ep = vloss, ep
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    return model, best_ep + 1

# ---------- 6. Evaluation ----------
present = sorted(set(y_te_np.tolist()))           # classes that actually exist in the test set
is_spoiled = np.array([c.endswith("_Spoiled") for c in classes])

@torch.no_grad()
def evaluate(model, use_ir=True, use_gas=True):
    te = D["test"]
    preds = model(te["srgb"], te["ir"], te["gas"], use_ir, use_gas).argmax(1).cpu().numpy()
    y = y_te_np
    return {
        "acc": (preds == y).mean() * 100,
        "macro_f1": f1_score(y, preds, labels=present, average="macro", zero_division=0) * 100,
        "balanced_acc": recall_score(y, preds, labels=present, average="macro", zero_division=0) * 100,
        "spoiled_vs_fresh_acc": (is_spoiled[preds] == is_spoiled[y]).mean() * 100,
        "n_wrong": int((preds != y).sum()),
    }, preds

# ---------- 7. Ablation study ----------
CONFIGS = [  # (name, use_ir, use_gas, modality_dropout)
    ("A. sRGB only",                         False, False, 0.00),
    ("B. sRGB + IR",                         True,  False, 0.00),
    ("C. sRGB + Gas",                        False, True,  0.00),
    ("D. sRGB + IR + Gas",                   True,  True,  0.00),
    ("E. sRGB + IR + Gas (+ modality drop)", True,  True,  0.15),
]
UNPLUG_TESTS = [("IR unplugged", False, True), ("Gas unplugged", True, False), ("IR + Gas unplugged", False, False)]
SEEDS = [0, 1, 2]

rows, final_model, final_preds = [], None, None
for name, use_ir, use_gas, p in CONFIGS:
    for seed in SEEDS:
        model, best_ep = train_fusion(use_ir, use_gas, p, seed)
        res, preds = evaluate(model, use_ir, use_gas)
        rows.append({"config": name, "seed": seed, "best_epoch": best_ep, **res})
        if use_ir and use_gas:                      # robustness: unplug sensors at test time
            for tag, ui, ug in UNPLUG_TESTS:
                r2, _ = evaluate(model, ui, ug)
                rows.append({"config": f"{name[:2]} -> {tag}", "seed": seed, "best_epoch": best_ep, **r2})
        if p > 0 and seed == SEEDS[0]:
            final_model, final_preds = model, preds
    print(f"done: {name}")

results = pd.DataFrame(rows)
results.to_csv(os.path.join(SAVE_DIR, "fusion_ablation_results.csv"), index=False)

order = list(dict.fromkeys(results["config"]))
metrics = ["acc", "macro_f1", "balanced_acc", "spoiled_vs_fresh_acc", "n_wrong", "best_epoch"]
summary = results.groupby("config", sort=False)[metrics].agg(["mean", "std"]).loc[order]
table = pd.DataFrame({m: summary[(m, "mean")].map("{:.2f}".format) + " ± " + summary[(m, "std")].map("{:.2f}".format)
                      for m in metrics})

# ---------- 8. Reference baselines ----------
maj12 = np.bincount(y_te_np).max() / len(y_te_np) * 100
maj_bin = max(is_spoiled[y_te_np].mean(), 1 - is_spoiled[y_te_np].mean()) * 100

print("\n" + "=" * 100)
print(f"ABLATION on the same {len(y_te_np)} test samples ({len(present)}/{NUM_CLASSES} classes present), "
      f"mean ± std over {len(SEEDS)} seeds")
print("=" * 100)
print(table.to_string())
print("\nReference baselines:")
print(f"  Always predict the most common class (12-class) : {maj12:.2f}%")
print(f"  Always predict 'fresh' (spoiled vs fresh)        : {maj_bin:.2f}%")
print(f"  ConvNeXt standalone (cell 5)                     : 95.45%   <- compare with row A")
print(f"  Gas MLP standalone, spoiled vs fresh (cell 10)   : 70.55%   <- compare with the line above")

print("\nFinal model (row E, seed 0), all sensors on:")
print(classification_report(y_te_np, final_preds, labels=present,
                            target_names=[classes[i] for i in present], zero_division=0))

v2_path = os.path.join(SAVE_DIR, "best_multimodal_fusion_v2.pth")
torch.save(final_model.state_dict(), v2_path)
print("Saved:", v2_path)
print("Saved:", os.path.join(SAVE_DIR, "fusion_ablation_results.csv"))
