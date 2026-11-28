"""Does background removal hurt the camera model on TR-6 photos (which already have a white background)?

Runs N random photos per class through the camera model three ways:
  original | white background | white background + crop around the fruit
and reports accuracy (12-class and fresh-vs-spoiled) for the camera model and the fusion head
(camera only), plus timing. Saves an example grid for the report.

Run from the Backend folder:
    .venv\\Scripts\\python -m experiments.eval_background --per-class 15

Note: photos are sampled from the whole dataset, so many are training images. This checks that the
pre-processing does no harm; it is not an accuracy figure for new fruit.
"""
import argparse
import json
import random
import time
from pathlib import Path

from PIL import Image, ImageDraw

from app.background import BackgroundCleaner
from app.model_store import BG_MODEL_FILE, ensure_models
from app.predictor import SpoilCheckPredictor

BACKEND = Path(__file__).resolve().parent.parent
MODELS = BACKEND / "Models"
DATA = MODELS / "dataset" / "TR-6_fixed" / "Classified"
REPORT_DIR = BACKEND.parent / "report assets"
VARIANTS = ["original", "white", "white_crop"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=15)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    ensure_models(MODELS)
    pred = SpoilCheckPredictor(MODELS)
    cleaner = BackgroundCleaner(MODELS / BG_MODEL_FILE)
    rng = random.Random(args.seed)

    items = []
    for label in pred.classes:
        state = "Spoiled" if label.endswith("_Spoiled") else "Not_spoiled"
        variety = label[: -len("_" + state)]
        files = sorted((DATA / variety / state / "sRGB_images").glob("*.jpg"))
        items += [(label, f) for f in rng.sample(files, min(args.per_class, len(files)))]

    stats = {v: {"cam": 0, "cam_fs": 0, "fus": 0, "fus_fs": 0} for v in VARIANTS}
    applied = 0
    t_clean, t_pred = [], []
    examples = []
    for n, (label, path) in enumerate(items, 1):
        img = Image.open(path).convert("RGB")
        truth_spoiled = label.endswith("_Spoiled")
        t0 = time.time()
        white, info = cleaner.clean(img)
        t_clean.append(time.time() - t0)
        crop, _ = cleaner.clean(img, crop=True)
        applied += info["applied"]
        imgs = {"original": img, "white": white, "white_crop": crop}
        for v in VARIANTS:
            t0 = time.time()
            r = pred.predict(imgs[v])
            t_pred.append(time.time() - t0)
            for key, m in (("cam", r["camera"]), ("fus", r["fusion"])):
                stats[v][key] += m["label"] == label
                stats[v][key + "_fs"] += (m["state"] == "Spoiled") == truth_spoiled
        if len(examples) < 6 and label.endswith("Spoiled") == (len(examples) % 2 == 1):
            examples.append((label, img, white, crop))
        if n % 20 == 0:
            print(f"{n}/{len(items)}")

    total = len(items)
    summary = {
        "photos": total,
        "per_class": args.per_class,
        "mask_applied": applied,
        "clean_seconds_avg": round(sum(t_clean) / total, 3),
        "predict_seconds_avg": round(sum(t_pred) / len(t_pred), 3),
        "accuracy": {
            v: {k: round(100 * c / total, 2) for k, c in s.items()} for v, s in stats.items()
        },
    }
    print(json.dumps(summary, indent=2))

    # example grid: rows = photos, columns = original / white / white+crop
    th = 260
    tiles = []
    for label, *ims in examples:
        row = [im.copy() for im in ims]
        for im in row:
            im.thumbnail((th * 4 // 3, th))
        tiles.append((label, row))
    W = sum(im.width for im in tiles[0][1]) + 40
    H = len(tiles) * (th + 30) + 30
    grid = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(grid)
    for j, name in enumerate(["Original", "White background", "White + crop"]):
        d.text((10 + j * (th * 4 // 3 + 10), 8), name, fill="black")
    for i, (label, row) in enumerate(tiles):
        x, y = 10, 30 + i * (th + 30)
        d.text((10, y), label, fill="black")
        for im in row:
            grid.paste(im, (x, y + 15))
            x += im.width + 10
    REPORT_DIR.mkdir(exist_ok=True)
    grid.save(REPORT_DIR / "bg_removal_examples.jpg", quality=90)
    (REPORT_DIR / "bg_removal_eval.json").write_text(json.dumps(summary, indent=2))
    print(f"Saved {REPORT_DIR / 'bg_removal_examples.jpg'}")


if __name__ == "__main__":
    main()
