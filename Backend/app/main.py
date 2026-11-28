"""SpoilCheck model tester API.

Run from the Backend folder:
    .venv\\Scripts\\python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
then open http://127.0.0.1:8000
"""
import io
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

from .model_store import ensure_models
from .predictor import SpoilCheckPredictor
from .samples import SampleIndex

BACKEND_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BACKEND_DIR / "Models"
FRONTEND_DIR = BACKEND_DIR.parent / "Frontend"
FUSION_FILE = os.environ.get("SPOILCHECK_FUSION", "best_multimodal_fusion.pth")
MAX_UPLOAD_MB = 30

_dataset_candidates = [MODEL_DIR / "dataset" / "TR-6_fixed" / "Classified",
                       MODEL_DIR / "dataset" / "TR-6" / "TR-6" / "Classified"]
DATASET_ROOT = next((p for p in _dataset_candidates if p.is_dir()), None)

state: dict = {}


@asynccontextmanager
async def lifespan(_app: FastAPI):
    t0 = time.time()
    ensure_models(MODEL_DIR, FUSION_FILE)
    state["predictor"] = SpoilCheckPredictor(MODEL_DIR, FUSION_FILE)
    state["samples"] = SampleIndex(DATASET_ROOT)
    state["load_seconds"] = round(time.time() - t0, 1)
    print(f"Models loaded in {state['load_seconds']}s | fusion={FUSION_FILE} | samples from {DATASET_ROOT}")
    yield


app = FastAPI(title="SpoilCheck Model Tester", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


async def _read_image(upload: UploadFile, name: str) -> Image.Image:
    data = await upload.read()
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"{name} image is larger than {MAX_UPLOAD_MB} MB")
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
        return img
    except (UnidentifiedImageError, OSError):
        raise HTTPException(400, f"{name} file is not a readable image")


def _num(value: str | None, name: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        raise HTTPException(400, f"'{name}' must be a number")


@app.get("/api/health")
def health():
    p: SpoilCheckPredictor = state["predictor"]
    s: SampleIndex = state["samples"]
    return {"status": "ok", "classes": p.classes, "fruits": p.fruits, "fusion_file": p.fusion_file,
            "load_seconds": state["load_seconds"], "samples_available": s.available,
            "sample_labels": s.labels, "background_removal": p.bg is not None}


@app.get("/api/sample")
def sample(label: str | None = None):
    s: SampleIndex = state["samples"]
    if not s.available:
        raise HTTPException(404, "Dataset folder not found, so no samples are available")
    if label and label not in s.labels:
        raise HTTPException(400, f"Unknown label '{label}'")
    return s.sample(label)


@app.get("/api/sample-file/{label}/{kind}/{filename}")
def sample_file(label: str, kind: str, filename: str):
    path = state["samples"].file_path(label, kind, filename)
    if path is None or not path.is_file():
        raise HTTPException(404, "File not found")
    if kind == "ir":
        return Response(path.read_bytes(), media_type="image/jpeg")
    # sRGB photos are 4000x3000: send a 1280px copy (the model resizes to 224 anyway)
    with Image.open(path) as img:
        img = img.convert("RGB")
        img.thumbnail((1280, 1280))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=92)
    return Response(buf.getvalue(), media_type="image/jpeg")


@app.post("/api/predict")
async def predict(
    srgb: UploadFile = File(...),
    # use_ir / use_gas are optional: when omitted, a sensor counts as present only if its data was
    # sent (IR image attached; all four gas values filled). Missing sensors are switched off inside
    # the fusion net -- never send 0 as a stand-in, that is read as a real measurement.
    use_ir: bool | None = Form(None),
    ir: UploadFile | None = File(None),
    t_min: str | None = Form(None),
    t_max: str | None = Form(None),
    use_gas: bool | None = Form(None),
    adc: str | None = Form(None),
    rs: str | None = Form(None),
    rs_r0: str | None = Form(None),
    methane_ppm: str | None = Form(None),
    fruit: str | None = Form(None),   # e.g. "Tomato"; empty/omitted = detect the fruit automatically
    clean_bg: bool = Form(False),     # cut the fruit out onto a white background before the camera model
):
    fruit = fruit or None
    if fruit and fruit not in state["predictor"].fruits:
        raise HTTPException(400, f"Unknown fruit '{fruit}'")
    srgb_img = await _read_image(srgb, "Camera")

    if use_ir is None:
        use_ir = ir is not None and bool(ir.filename)
    if use_gas is None:
        use_gas = all(v not in (None, "") for v in (adc, rs, rs_r0, methane_ppm))

    ir_img, tmin, tmax = None, None, None
    if use_ir:
        if ir is None or not ir.filename:
            raise HTTPException(400, "IR sensor is on: upload a thermal image or switch the IR sensor off")
        ir_img = await _read_image(ir, "Thermal")
        tmin, tmax = _num(t_min, "t_min"), _num(t_max, "t_max")
        if tmax < tmin:
            raise HTTPException(400, "t_max must be greater than or equal to t_min")

    gas = None
    if use_gas:
        gas = {"adc": _num(adc, "ADC"), "rs": _num(rs, "Rs"), "rs_r0": _num(rs_r0, "Rs/R0"),
               "methane_ppm": _num(methane_ppm, "Methane ppm")}

    t0 = time.time()
    result = state["predictor"].predict(srgb_img, ir_img, tmin, tmax, gas, fruit, clean_bg)
    result["inference_ms"] = round((time.time() - t0) * 1000)
    return result


# Serve the frontend at / (API routes above take priority)
if FRONTEND_DIR.is_dir():
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
