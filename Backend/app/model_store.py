"""Makes sure the trained model files are present in Backend/Models.

The weights are too large for GitHub (best_convnext_spoilage.pth is 107 MB), so they live on
Hugging Face: https://huggingface.co/ArmanKhan01/SpoilCheck-Models
Any file missing locally is downloaded on startup; files already on disk are used as-is, so a
machine that already has the models never touches the network.

Override the repo with the SPOILCHECK_HF_REPO environment variable if needed.
"""
import os
import shutil
from pathlib import Path

HF_REPO_ID = os.environ.get("SPOILCHECK_HF_REPO", "ArmanKhan01/SpoilCheck-Models")

# All of these must come from the same training run (see README).
MODEL_FILES = [
    "best_convnext_spoilage.pth",
    "best_ir_model.pth",
    "best_gas_model.pth",
    "best_multimodal_fusion.pth",
    "sensor_scalers.pkl",
    "class_indices.json",
]


def ensure_models(model_dir: Path, fusion_file: str = "best_multimodal_fusion.pth") -> None:
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    needed = sorted(set(MODEL_FILES) | {fusion_file})
    missing = [f for f in needed if not (model_dir / f).is_file()]
    if not missing:
        return

    from huggingface_hub import hf_hub_download

    for name in missing:
        print(f"Downloading {name} from huggingface.co/{HF_REPO_ID} ...")
        cached = hf_hub_download(repo_id=HF_REPO_ID, filename=name, token=os.environ.get("HF_TOKEN"))
        shutil.copyfile(cached, model_dir / name)
