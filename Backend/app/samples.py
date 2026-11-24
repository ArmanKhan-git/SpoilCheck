"""Picks ready-made test inputs (photo + nearest thermal frame + a gas reading) from the local TR-6 dataset,
pairing them the same way the training notebook did."""
import bisect
import os
import random
import re
from datetime import datetime
from pathlib import Path

IR_RE = re.compile(r"^(\d{8}_\d{6})_(\d+(?:\.\d+)?)C_(\d+(?:\.\d+)?)C\.jpg$", re.I)
SRGB_RE = re.compile(r"^(\d{8}_\d{6})\.jpg$", re.I)
GAS_RE = re.compile(r"ADC:\s*([\d.]+).*?Rs:\s*([\d.]+).*?Rs/R0:\s*([\d.]+).*?Methane:\s*([\d.]+)", re.I)
MAX_GAP_MIN = 30


def _ts(stamp: str) -> float:
    return datetime.strptime(stamp, "%Y%m%d_%H%M%S").timestamp()


def _gas_blocks(path: Path) -> list[list[tuple]]:
    blocks: list[list[tuple]] = []
    if not path.exists():
        return blocks
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if re.match(r"^Day\s+\d+", line, re.I):
                blocks.append([])
                continue
            m = GAS_RE.search(line)
            if m:
                if not blocks:
                    blocks.append([])
                blocks[-1].append(tuple(float(v) for v in m.groups()))
    return [b for b in blocks if b]


class SampleIndex:
    def __init__(self, dataset_root: Path | None):
        self.root = dataset_root
        self._cache: dict[str, dict] = {}
        self.labels: list[str] = []
        if dataset_root and dataset_root.is_dir():
            for variety in sorted(os.listdir(dataset_root)):
                for state in ("Not_spoiled", "Spoiled"):
                    if (dataset_root / variety / state / "sRGB_images").is_dir():
                        self.labels.append(f"{variety}_{state}")

    @property
    def available(self) -> bool:
        return bool(self.labels)

    def _folder(self, label: str) -> Path:
        state = "Spoiled" if label.endswith("_Spoiled") else "Not_spoiled"
        return self.root / label[: -len("_" + state)] / state

    def _class(self, label: str) -> dict:
        if label not in self._cache:
            base = self._folder(label)
            srgb = sorted(f for f in os.listdir(base / "sRGB_images") if SRGB_RE.match(f))
            ir = []
            for f in os.listdir(base / "IR_fusion_images"):
                m = IR_RE.match(f)
                if m:
                    ir.append((_ts(m.group(1)), f, float(m.group(2)), float(m.group(3))))
            ir.sort()
            blocks = _gas_blocks(base / "Methane_gas_readings.txt")
            dates = sorted({f[:8] for f in srgb})
            self._cache[label] = {
                "srgb": srgb, "srgb_set": set(srgb), "ir": ir, "ir_ts": [r[0] for r in ir],
                "ir_set": {r[1] for r in ir}, "blocks": blocks,
                "date_block": {d: min(i, len(blocks) - 1) for i, d in enumerate(dates)} if blocks else {},
            }
        return self._cache[label]

    def file_path(self, label: str, kind: str, filename: str) -> Path | None:
        """Resolve a requested file only if it is a known file of that class (no free-form paths)."""
        if label not in self.labels or kind not in ("srgb", "ir"):
            return None
        c = self._class(label)
        if filename not in (c["srgb_set"] if kind == "srgb" else c["ir_set"]):
            return None
        return self._folder(label) / ("sRGB_images" if kind == "srgb" else "IR_fusion_images") / filename

    def sample(self, label: str | None = None) -> dict:
        label = label or random.choice(self.labels)
        c = self._class(label)
        fname = random.choice(c["srgb"])
        stamp = SRGB_RE.match(fname).group(1)
        out = {"label": label, "srgb": {"name": fname, "url": f"/api/sample-file/{label}/srgb/{fname}"},
               "ir": None, "gas": None}

        if c["ir"]:
            t = _ts(stamp)
            i = bisect.bisect_left(c["ir_ts"], t)
            j = min((k for k in (i - 1, i) if 0 <= k < len(c["ir"])), key=lambda k: abs(c["ir_ts"][k] - t))
            gap = abs(c["ir_ts"][j] - t) / 60
            if gap <= MAX_GAP_MIN:
                _, ir_name, t_min, t_max = c["ir"][j]
                out["ir"] = {"name": ir_name, "url": f"/api/sample-file/{label}/ir/{ir_name}",
                             "t_min": t_min, "t_max": t_max, "gap_min": round(gap, 1)}

        if c["blocks"]:
            block = c["blocks"][c["date_block"][stamp[:8]]]
            adc, rs, rs_r0, ppm = random.choice(block)
            out["gas"] = {"adc": adc, "rs": rs, "rs_r0": rs_r0, "methane_ppm": ppm}
        return out
