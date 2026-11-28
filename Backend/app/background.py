"""Optional background removal for camera photos ("clean background").

The camera model was trained on TR-6 photos: one fruit on a plain white background. Phone photos
taken in a kitchen or market look different, so this step cuts the fruit out with U^2-Net-p
(a 4.6 MB salient-object segmentation model, run with onnxruntime) and pastes it onto pure white
before the photo reaches the classifier. Applied to the camera photo only, never the thermal image.

Pre/post-processing matches rembg's u2netp session (https://github.com/danielgatis/rembg),
without rembg's heavier dependencies (OpenCV, scikit-image, SciPy).
"""
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
SIZE = 320
WORK_SIZE = 1024

# If the mask keeps less than MIN or more than MAX of the frame, the cut-out probably failed
# (nothing salient found / the whole frame kept), so the original photo is used instead.
MIN_COVERAGE = 0.01
MAX_COVERAGE = 0.90


class BackgroundCleaner:
    def __init__(self, model_path: Path):
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        self.session = ort.InferenceSession(str(model_path), opts, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

    def mask(self, img: Image.Image) -> Image.Image:
        """Soft foreground mask (L mode, 0-255) at the image's own size."""
        small = np.asarray(img.convert("RGB").resize((SIZE, SIZE), Image.Resampling.LANCZOS), dtype=np.float32)
        small = small / max(float(small.max()), 1e-6)
        x = ((small - MEAN) / STD).transpose(2, 0, 1)[None].astype(np.float32)
        pred = self.session.run(None, {self.input_name: x})[0][0, 0]
        pred = (pred - pred.min()) / max(float(pred.max() - pred.min()), 1e-6)
        m = Image.fromarray((pred * 255).astype(np.uint8), mode="L")
        return m.resize(img.size, Image.Resampling.LANCZOS)

    def clean(self, img: Image.Image, crop: bool = False, pad: float = 0.15) -> tuple[Image.Image, dict]:
        """Returns (image on white, info). Falls back to the original image if the mask looks wrong.

        Works on a copy at most WORK_SIZE px on its longest side: the classifier resizes to 224 px
        anyway (training used 256-px cached copies), and a full 12 MP phone photo would make this
        step take seconds.
        """
        img = img.convert("RGB")
        if max(img.size) > WORK_SIZE:
            img = img.copy()
            img.thumbnail((WORK_SIZE, WORK_SIZE), Image.Resampling.LANCZOS)
        m = self.mask(img)
        a = np.asarray(m, dtype=np.float32) / 255.0
        coverage = float((a > 0.5).mean())
        info = {"applied": False, "mask_coverage": round(coverage, 4), "cropped": False}
        if not (MIN_COVERAGE <= coverage <= MAX_COVERAGE):
            info["reason"] = "no clear object found" if coverage < MIN_COVERAGE else "object fills the frame"
            return img, info

        rgb = np.asarray(img, dtype=np.float32)
        out = rgb * a[..., None] + 255.0 * (1.0 - a[..., None])
        result = Image.fromarray(out.clip(0, 255).astype(np.uint8))
        info["applied"] = True

        if crop:
            ys, xs = np.where(a > 0.5)
            x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
            w, h = x1 - x0, y1 - y0
            # pad, then widen to the 4:3 shape of the training photos
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            w, h = w * (1 + 2 * pad), h * (1 + 2 * pad)
            w, h = max(w, h * 4 / 3), max(h, w * 3 / 4)
            box = (int(cx - w / 2), int(cy - h / 2), int(cx + w / 2), int(cy + h / 2))
            canvas = Image.new("RGB", (box[2] - box[0], box[3] - box[1]), (255, 255, 255))
            canvas.paste(result, (-box[0], -box[1]))
            result = canvas
            info["cropped"] = True
        return result, info
