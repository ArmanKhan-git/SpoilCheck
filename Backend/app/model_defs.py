"""Model architectures, identical to the training notebook so the saved weights load unchanged."""
import timm
import torch
import torch.nn as nn


class StandaloneIRModel(nn.Module):
    """IR thermal branch: ResNet18 on the heatmap + small MLP on (t_min, t_max, t_diff) -> 544-d."""

    def __init__(self, num_classes: int = 12):
        super().__init__()
        self.backbone = timm.create_model("resnet18", pretrained=False, num_classes=0)  # 512-d
        self.temp_net = nn.Sequential(nn.Linear(3, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU())
        self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(512 + 32, 128), nn.ReLU(), nn.Linear(128, num_classes))

    def forward(self, ir_img, temps, return_features=False):
        fused = torch.cat([self.backbone(ir_img), self.temp_net(temps)], dim=1)
        return fused if return_features else self.head(fused)


class StandaloneGasMLP(nn.Module):
    """Gas branch: MLP on (ADC, Rs, Rs/R0, methane ppm) -> 64-d."""

    def __init__(self, in_dim: int = 4, out_dim: int = 12):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 64), nn.BatchNorm1d(64), nn.ReLU(),
        )
        self.head = nn.Linear(64, out_dim)

    def forward(self, x, return_features=False):
        feat = self.net(x)
        return feat if return_features else self.head(feat)


class MultimodalFusionNet(nn.Module):
    """Late-fusion head. A missing sensor zeros its PROJECTED features (h_ir / h_gas), the same
    way the notebook's unplugged tests do. The deployed v2 head is ablation config D
    (no modality dropout); the training-only p_mod_drop argument is not needed for inference."""

    def __init__(self, num_classes: int = 12):
        super().__init__()
        self.proj_srgb = nn.Sequential(nn.Linear(768, 256), nn.LayerNorm(256), nn.ReLU(), nn.Dropout(0.3))
        self.proj_ir = nn.Sequential(nn.Linear(544, 128), nn.LayerNorm(128), nn.ReLU(), nn.Dropout(0.3))
        self.proj_gas = nn.Sequential(nn.Linear(64, 64), nn.LayerNorm(64), nn.ReLU(), nn.Dropout(0.2))
        self.fusion_head = nn.Sequential(
            nn.Linear(256 + 128 + 64, 192), nn.BatchNorm1d(192), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(192, num_classes),
        )

    def forward(self, e_srgb, e_ir, e_gas, use_ir=True, use_gas=True):
        h_srgb, h_ir, h_gas = self.proj_srgb(e_srgb), self.proj_ir(e_ir), self.proj_gas(e_gas)
        if not use_ir:
            h_ir = torch.zeros_like(h_ir)
        if not use_gas:
            h_gas = torch.zeros_like(h_gas)
        return self.fusion_head(torch.cat([h_srgb, h_ir, h_gas], dim=1))
