"""Loads the four trained models and runs inference with any combination of sensors."""
import base64
import io
import json
from pathlib import Path

import joblib
import timm
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from .background import BackgroundCleaner
from .model_defs import MultimodalFusionNet, StandaloneGasMLP, StandaloneIRModel

GAS_KEYS = ("adc", "rs", "rs_r0", "methane_ppm")
# Below these, the answer is flagged as unsure instead of shown as a confident verdict.
# Calibrated on 60 TR-6 photos (camera only, fusion head): the top fruit's summed probability was
# always >= 0.53 and >= 4x the runner-up fruit. The web tomato photo that was called mango: ~0.4 and ~2x.
FRUIT_CONF_MIN = 0.45     # summed probability of the top fruit (both of its classes)
FRUIT_RATIO_MIN = 3.0     # top fruit must be at least this many times as likely as the second fruit
STATE_MARGIN = 0.15       # spoiled probability within 0.5 ± this = unsure fresh/spoiled


def pretty(label: str) -> dict:
    state = "Spoiled" if label.endswith("_Spoiled") else "Not_spoiled"
    variety = label[: -len("_" + state)]
    return {
        "label": label,
        "variety": variety.replace("_", " "),
        "state": "Spoiled" if state == "Spoiled" else "Fresh",
    }


class SpoilCheckPredictor:
    def __init__(self, model_dir: Path, fusion_file: str = "best_multimodal_fusion.pth",
                 bg_model_file: str | None = "u2netp.onnx"):
        self.model_dir = Path(model_dir)
        bg_path = self.model_dir / bg_model_file if bg_model_file else None
        self.bg = BackgroundCleaner(bg_path) if bg_path and bg_path.is_file() else None
        self.fusion_file = fusion_file
        with open(self.model_dir / "class_indices.json") as f:
            c2i = json.load(f)
        self.classes = [k for k, _ in sorted(c2i.items(), key=lambda kv: kv[1])]
        self.spoiled_mask = torch.tensor([c.endswith("_Spoiled") for c in self.classes])
        self.class_fruit = [pretty(c)["variety"] for c in self.classes]
        self.fruits = sorted(set(self.class_fruit))
        n = len(self.classes)

        scalers = joblib.load(self.model_dir / "sensor_scalers.pkl")
        self.temp_scaler, self.gas_scaler = scalers["temp_scaler"], scalers["gas_scaler"]

        def load(model, filename):
            state = torch.load(self.model_dir / filename, map_location="cpu", weights_only=True)
            model.load_state_dict(state)
            return model.eval()

        self.srgb = load(timm.create_model("convnext_tiny", pretrained=False, num_classes=n), "best_convnext_spoilage.pth")
        self.ir = load(StandaloneIRModel(n), "best_ir_model.pth")
        self.gas = load(StandaloneGasMLP(4, n), "best_gas_model.pth")
        self.fusion = load(MultimodalFusionNet(n), fusion_file)

        # identical to eval_tfms used in training
        self.tfms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def _probs(self, logits: torch.Tensor, fruit: str | None = None) -> torch.Tensor:
        """Class probabilities (12,). If the fruit is given, only its fresh/spoiled classes compete."""
        logits = logits.float()
        if fruit:
            keep = torch.tensor([f == fruit for f in self.class_fruit])
            logits = logits.masked_fill(~keep, float("-inf"))
        return F.softmax(logits, dim=1)[0]

    def _summary(self, logits: torch.Tensor, top: int = 3, fruit: str | None = None) -> dict:
        return self._summary_probs(self._probs(logits, fruit), top, fruit)

    def _summary_probs(self, probs: torch.Tensor, top: int = 3, fruit: str | None = None) -> dict:
        if fruit:
            top = min(top, 2)
        k = int(probs.argmax())
        order = torch.argsort(probs, descending=True)[:top].tolist()
        fruit_probs = {f: 0.0 for f in self.fruits}
        for i, f in enumerate(self.class_fruit):
            fruit_probs[f] += float(probs[i])
        fruit_ranked = sorted(fruit_probs.items(), key=lambda kv: kv[1], reverse=True)
        spoiled_p = float(probs[self.spoiled_mask].sum())
        fruit_conf = fruit_ranked[0][1]
        return {
            **pretty(self.classes[k]),
            "confidence": round(float(probs[k]), 4),
            "spoiled_probability": round(spoiled_p, 4),
            "fruit_confidence": round(fruit_conf, 4),
            "fruit_candidates": [{"variety": f, "probability": round(p, 4)} for f, p in fruit_ranked[:3]],
            "unsure_fruit": fruit is None and (fruit_conf < FRUIT_CONF_MIN
                                               or fruit_conf < FRUIT_RATIO_MIN * fruit_ranked[1][1]),
            "unsure_state": abs(spoiled_p - 0.5) < STATE_MARGIN,
            "top": [{**pretty(self.classes[i]), "probability": round(float(probs[i]), 4)} for i in order],
        }

    def _clean(self, img: Image.Image, clean_bg: bool) -> tuple[Image.Image, dict]:
        """Optional: cut the fruit out and put it on white, like the TR-6 training photos."""
        background = {"requested": clean_bg, "applied": False}
        if clean_bg:
            if self.bg is None:
                background["reason"] = "background-removal model not loaded"
            else:
                img, info = self.bg.clean(img)
                background.update(info)
                if info["applied"]:
                    background["preview"] = _jpeg_data_url(img)
        return img, background

    @torch.inference_mode()
    def predict(self, srgb_imgs: Image.Image | list[Image.Image], ir_img: Image.Image | None = None,
                t_min: float | None = None, t_max: float | None = None, gas: dict | None = None,
                fruit: str | None = None, clean_bg: bool = False) -> dict:
        """One or more camera photos of the SAME fruit (e.g. different sides) + optional IR and gas.

        Each photo goes through the camera model and the fusion head (with the same IR / gas
        features); the class probabilities are then averaged over the photos for the verdict.
        """
        if fruit is not None and fruit not in self.fruits:
            raise ValueError(f"Unknown fruit '{fruit}'. Choose one of: {', '.join(self.fruits)}")
        if isinstance(srgb_imgs, Image.Image):
            srgb_imgs = [srgb_imgs]
        if not srgb_imgs:
            raise ValueError("At least one camera photo is needed")
        result = {}

        # IR thermal (optional, one reading for the whole fruit)
        use_ir = ir_img is not None
        if use_ir:
            temps = torch.tensor(self.temp_scaler.transform([[t_min, t_max, round(t_max - t_min, 2)]]), dtype=torch.float32)
            e_ir = self.ir(self.tfms(ir_img.convert("RGB")).unsqueeze(0), temps, return_features=True)
            result["ir"] = self._summary(self.ir.head(e_ir), fruit=fruit)
        else:
            e_ir = torch.zeros(1, 544)
            result["ir"] = None

        # methane gas (optional, one reading for the whole fruit)
        use_gas = gas is not None
        if use_gas:
            g = torch.tensor(self.gas_scaler.transform([[gas[k] for k in GAS_KEYS]]), dtype=torch.float32)
            e_gas = self.gas(g, return_features=True)
            result["gas"] = self._summary(self.gas.head(e_gas), fruit=fruit)
        else:
            e_gas = torch.zeros(1, 64)
            result["gas"] = None

        # camera (always on): every photo separately
        cam_probs, fus_probs, views = [], [], []
        for i, img in enumerate(srgb_imgs):
            img, background = self._clean(img, clean_bg)
            x = self.tfms(img.convert("RGB")).unsqueeze(0)
            feats = self.srgb.forward_features(x)
            e_srgb = self.srgb.forward_head(feats, pre_logits=True)      # 768-d embedding
            cp = self._probs(self.srgb.forward_head(feats), fruit)
            fp = self._probs(self.fusion(e_srgb, e_ir, e_gas, use_ir=use_ir, use_gas=use_gas), fruit)
            cam_probs.append(cp)
            fus_probs.append(fp)
            v = self._summary_probs(fp, top=1, fruit=fruit)
            views.append({"index": i, "label": v["label"], "variety": v["variety"], "state": v["state"],
                          "confidence": v["confidence"], "spoiled_probability": v["spoiled_probability"],
                          "background": background})

        result["camera"] = self._summary_probs(torch.stack(cam_probs).mean(0), fruit=fruit)
        result["fusion"] = self._summary_probs(torch.stack(fus_probs).mean(0), top=5, fruit=fruit)
        result["views"] = views
        # rot is often on one side only: report how many photos on their own look spoiled
        result["views_spoiled"] = sum(v["spoiled_probability"] > 0.5 for v in views)
        result["sensors_used"] = {"camera": True, "ir": use_ir, "gas": use_gas}
        result["fruit_given"] = fruit
        result["background"] = views[0]["background"]   # first photo; kept for single-photo clients
        return result


def _jpeg_data_url(img: Image.Image, max_side: int = 480) -> str:
    """Small JPEG preview of the image the camera model actually saw."""
    im = img.convert("RGB").copy()
    im.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=80)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
