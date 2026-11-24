# SpoilCheck: Multimodal Fruit Spoilage Detection

B.Tech CSE final-year project (BTKIT Dwarahat). Classifies produce as **fresh or spoiled** from three sensors: an **sRGB camera**, an **IR thermal camera** and an **MQ-4 methane (gas) sensor**, fused by a late-fusion network.

This README records the project's state, what was found during review, and what's next, so work can resume in a new session without re-deriving anything.

**Current state (22 Nov 2026):** v2 models trained on the fixed dataset and **deployed** in `Backend/Models/`. v1 models kept in `Backend/Models/v1_backup/`. Full v1 → v2 comparison: [SpoilCheck_v1_vs_v2.md](SpoilCheck_v1_vs_v2.md).

**Quick start (fresh clone):** run `Backend\run.bat` and open http://127.0.0.1:8000. The trained model weights are not in this repo (GitHub's 100 MB file limit); they are hosted at **[huggingface.co/ArmanKhan01/SpoilCheck-Models](https://huggingface.co/ArmanKhan01/SpoilCheck-Models)** and downloaded into `Backend/Models/` automatically on first start (`Backend/app/model_store.py`). The TR-6 dataset is not included either; without it the tester still works, only "Load sample" is unavailable. The report, slides and v1 notes referenced below are kept locally and are not in the repo.

---

## 1. Folder layout

```
SpoilCheck Final Year Project/
├── README.md                         ← this file
├── SpoilCheck_v1_vs_v2.md            ← v1 vs v2 changes, results and v2 capabilities (personal notes)
├── SpoilCheck_Mentor_Update.pptx     ← 9-slide mentor deck, updated with v2 results
├── SpoilCheck_Mentor_Update_v1.pptx  ← the same deck with the old v1 numbers
├── Backend/
│   ├── run.bat                       ← double-click: starts the tester at http://127.0.0.1:8000
│   ├── requirements.txt              ← torch 2.5.1 CPU, timm 1.0.11, scikit-learn 1.6.1, fastapi …
│   ├── .venv/                        ← Python 3.11 virtual env
│   ├── app/
│   │   ├── main.py                   ← FastAPI: /api/health, /api/predict, /api/sample, serves Frontend/
│   │   ├── predictor.py              ← loads the 4 models + scalers, runs inference with any sensor combo
│   │   ├── model_defs.py             ← model classes (must match the training notebook exactly)
│   │   └── samples.py                ← picks test inputs (photo + nearest IR + gas reading) from the dataset
│   └── Models/
│       ├── best_convnext_spoilage.pth     ← v2 camera model (ConvNeXt-Tiny, 12 classes)
│       ├── best_ir_model.pth              ← v2 IR model (ResNet18 + temperature MLP)
│       ├── best_gas_model.pth             ← v2 gas model (MLP, properly trained but no useful signal, see §5)
│       ├── best_multimodal_fusion.pth     ← v2 fusion head = ablation config D (renamed from best_multimodal_fusion_D.pth)
│       ├── class_indices.json             ← 12 class names → indices (same in v1 and v2)
│       ├── sensor_scalers.pkl             ← v2 StandardScalers for temps + gas (fit on v2 train split)
│       ├── v1_backup/                     ← the six v1 files, same names
│       ├── SpoilCheck_Training_v2.ipynb   ← v2 training notebook (run on Colab 22 Nov 2026)
│       ├── colab_cell_fusion_ablation.py  ← single ablation cell used on the v1 Colab run
│       └── dataset/
│           ├── TR-6/TR-6/                 ← ORIGINAL public dataset (never modified)
│           ├── TR-6_fixed/                ← CORRECTED copy (see §3), used by v2 and by the tester's samples
│           │   ├── Classified/
│           │   ├── CHANGES.txt
│           │   └── SpoilCheck_TR6_fixed.zip   ← 10.6 GB zip for Colab
│           └── fix_tr6.py                 ← script that produced TR-6_fixed (safe to re-run)
└── Frontend/
    ├── index.html, styles.css, app.js     ← model tester page (no build step, works offline)
```

Colab outputs (not copied locally): `MyDrive/spoilcheck_v2/` holds `fusion_ablation_results.csv`, `test_results_summary.csv`, `mm_train/val/test.csv`, `cm_srgb.png`, `cm_ir.png`, `cm_fusion.png`, `cm_fusion_D.png`, `gradcam_examples.png`.

**All six model files must come from the same run.** The fusion head expects embeddings from its own encoders and the scalers from its own split; mixing v1 and v2 files gives wrong answers with no error.

---

## 2. Models (v2, deployed)

| Model | Input | Architecture | Output |
|---|---|---|---|
| Camera | sRGB photo, resized 224×224, ImageNet normalisation | timm `convnext_tiny` | 768-d embedding / 12-class logits |
| IR | thermal image 224×224 + (t_min, t_max, t_diff) scaled | timm `resnet18` (512-d) + MLP 3→32→32 | 544-d / 12-class |
| Gas | (ADC, Rs, Rs/R0, methane_ppm) scaled | MLP 4→64→64 with BatchNorm | 64-d / 12-class |
| Fusion | the three embeddings | each projected (768→256, 544→128, 64→64, LayerNorm+ReLU) → concat 448 → 192 → 12 | final class |

- Architectures are identical in v1 and v2; only the weights and scalers changed.
- **Deployed fusion = config D** (all three sensors, **no** modality dropout). Config E (modality dropout 0.15), which the notebook's cell 7 saves by default, was worse in every row; see §5.
- **12 classes:** {Banana, Carrot, Guava, Indian_Gooseberry, Mango, Tomato} × {Not_spoiled, Spoiled}.
- **Missing sensor:** zero the **projected** features (`h_ir` / `h_gas`) inside the fusion net, via the `use_ir=False` / `use_gas=False` flags. Do **not** pass zeros as the raw input embedding.
- t_min / t_max come from the IR filename: `YYYYMMDD_HHMMSS_<tmin>C_<tmax>C.jpg`.
- Training preprocessing never applied EXIF rotation; inference matches that.

---

## 3. Dataset: TR-6 (public)

- **Source:** public multimodal dataset (Mendeley/IEEE; paper: "Multimodal perishable fruits and vegetables dataset"). **Not collected by us; credit it in the slides.**
- **Size:** 5,338 sRGB photos in `Classified` (4000×3000 phone JPEGs, white background), IR images 640×480, one `Methane_gas_readings.txt` per class folder.
- **Structure:**
  - **About 3 physical fruits per variety**, each photographed daily: 17 thermal shots per fruit per day (16 side angles + 1 top).
  - "Fresh" = a fruit's early days, "Spoiled" = its later days.
  - Fresh and spoiled were shot on **different dates**.
- **Gas logs:** only "Day N / case 1–3 / R / L" headers, **no timestamps**. Values barely vary (e.g. 4,860 readings with only 101 distinct values).

### Errors found and fixed in `TR-6_fixed` (original untouched)
1. **Banana/Spoiled/IR_fusion_images** was a byte-identical copy of Banana/Not_spoiled IR (153 files, 11–13 Sep). Replaced with the real 187 images (15–18 Sep) from `Normal/Banana`.
2. **17 IR images** (29 Aug 13:20–13:24) were duplicated in Indian_Gooseberry/Spoiled and Tomato/Not_spoiled. They belong to gooseberry, so they were removed from Tomato.
3. **Gas day-header typos** renumbered: Carrot/Spoiled (3,4,5,6,7,7,8 → 3…9), Mango/Not_spoiled (1,2,3,3,4 → 1…5).
4. **Not fixed:** Indian_Gooseberry/Spoiled gas has 13 Day blocks for 12 photo dates.

Result: 10,824 files, 0 cross-class duplicate IR files. v1 was trained on the unfixed data, v2 on the fixed data.

### Pairing facts
- sRGB and IR are **separate sessions**, typically 2–13 minutes apart and up to 3.7 h apart in the original.
- v2 drops pairs more than **30 min** apart: **4,980 kept**, 357 dropped (Tomato_Not_spoiled 187, Gooseberry_Spoiled 119, Gooseberry_Not_spoiled 34, Carrot_Spoiled 17).
- Gas readings are sampled at random from the matching day block, so they are **day-level context, not per-photo measurements**.

---

## 4. Review findings on the v1 training, and their status in v2

| Issue | Evidence | Status |
|---|---|---|
| Gas MLP saved after **2 optimizer steps** | Full-batch training; BatchNorm `num_batches_tracked = 2` | **Fixed in v2** (mini-batch, best epoch by val loss) |
| Fusion head saved at **epoch 2 of 60** | val acc saturated | **Fixed in v2** (val-loss selection) |
| "IR unplugged" test zeroed the raw embedding instead of `h_ir` | Code inspection | **Fixed** in v2 notebook and backend |
| Modality dropout per batch, not per sample | Code inspection | **Fixed** in v2 (and not used by the deployed config D) |
| Carrot_Not_spoiled missing from val and test; Tomato_Spoiled had 3 test samples | Split output "11/12" | **Fixed:** v2 split seed 185, all 12 classes in val + test, Tomato_Spoiled 17 test samples |
| ColorJitter used though slides said "colour-safe only" | Code | **Fixed:** removed in v2 |
| Split by Variety_State_**Date_Hour**: same day can appear in train and test | Code | Kept; `GROUP_LEVEL="date"` option exists, not run yet |
| Fusion best epoch = 1: encoders trained on the same train set, so train embeddings are "too clean" | Ablation `best_epoch` | **Still open in v2** (best_epoch = 1 for every config); optional out-of-fold embeddings |

---

## 5. Results

### v2 (fixed data, 513 test samples, all 12 classes) — current

Split: train 3,940 / val 527 / test 513.

**Standalone models (test):**

| Model | Acc | Macro-F1 | Fresh vs spoiled |
|---|---|---|---|
| Camera (ConvNeXt-Tiny) | 95.91 | 95.09 | 95.91 |
| IR (ResNet18 + temps) | 90.06 | 81.30 | 90.06 |
| Gas MLP | 9.94 | 8.68 | **40.55** (vs 70.76 for always guessing "fresh") |

**Ablation** (same fusion head, only the sensors change, mean ± std over 3 seeds):

| Sensors | Acc | Macro-F1 | Wrong / 513 |
|---|---|---|---|
| A. Camera only | 96.49 ± 0.39 | 95.63 ± 0.52 | 18.0 |
| B. Camera + IR | 97.99 ± 0.30 | 96.83 ± 0.45 | 10.3 |
| C. Camera + gas | 96.75 ± 0.11 | 95.94 ± 0.15 | 16.7 |
| **D. All three** | **98.12 ± 0.11** | **97.04 ± 0.20** | **9.7** |
| D → IR unplugged | 96.17 ± 0.11 | 95.34 ± 0.23 | 19.7 |
| D → gas unplugged | 97.79 ± 0.30 | 96.47 ± 0.37 | 11.3 |
| D → IR + gas unplugged | 96.04 ± 0.49 | 95.15 ± 0.68 | 20.3 |
| E. All three + modality dropout | 97.27 ± 0.70 | 96.15 ± 0.87 | 14.0 |
| E → IR unplugged | 96.10 ± 0.52 | 95.22 ± 0.67 | 20.0 |

**Deployed model (D, seed 0):** 98.05% acc / 97.01 macro-F1; IR unplugged 96.30 / 95.49; gas unplugged 97.47 / 96.12; both unplugged 96.49 / 95.61. The cell 7 default (E, seed 0) scored only 96.49 / 95.19, so it was replaced.

**What it means:**
- Thermal nearly **halves the errors** (18 → 10 of 513) and gives +1.4 macro-F1 over the camera alone.
- **Gas carries no signal:** alone it is worse than always guessing "fresh"; in fusion it changes results by ≤0.3 points, within seed noise. Recommendation: drop gas from the claims and keep it only as an optional hardware input.
- Modality dropout is not needed: zeroing the projected features already makes D work without IR (96.2%).
- Fruit type is never confused: every error is fresh ↔ spoiled within the same fruit.
- Banana IR is now perfect (confirms the dataset fix).
- Weakest classes (final model): **Tomato_Spoiled recall 0.65** (6 of 17 called fresh); Gooseberry_Spoiled precision 0.85. IR alone misses all spoiled tomatoes and half of spoiled carrot and mango.

### v1 (original data, 550 test samples, 11/12 classes) — superseded

Standalone: camera 95.45%, IR 90.73%, gas 17.27% (12-class) / 70.55% fresh-vs-spoiled vs 69.64% baseline.

| Sensors | Accuracy | Macro-F1 | Wrong / 550 |
|---|---|---|---|
| Camera only | 96.12 ± 1.07 | 92.54 | 21.3 |
| Camera + IR | 96.67 ± 0.21 | 95.12 | 18.3 |
| Camera + gas | 95.39 ± 0.84 | 91.47 | 25.3 |
| All three | 96.73 ± 0.18 | 95.38 | 18.0 |
| All three (modality dropout), IR unplugged | 95.15 ± 0.46 | 90.84 | 26.7 |

v1 and v2 use **different test sets**, so present v2 on its own rather than as an "improvement" over v1.

**Limits to state honestly (both versions):**
- About 3 fruits per variety, so the test uses the same fruits on other days, not new fruits.
- "Spoiled" means obvious visible rot; no early or internal-rot examples.
- White background only.
- Gas isn't time-matched to the photos.
- Small classes have 17 test samples, so one session more or less moves per-class numbers a lot.

---

## 6. Training notebook v2: `Backend/Models/SpoilCheck_Training_v2.ipynb` (run 22 Nov 2026)

Upload `SpoilCheck_TR6_fixed.zip` to My Drive root, upload the notebook to Colab, choose a T4 GPU, and **Run all** (about 45–60 min). Outputs go to `MyDrive/spoilcheck_v2/`.

| Cell | Purpose |
|---|---|
| 1 | Mount Drive, unzip dataset |
| 2 | Pair sRGB↔IR (≤30 min) ↔ gas (day blocks in order); one shared split with all 12 classes in val + test |
| 3 | Cache 256-px sRGB copies, fit scalers, shared helpers; best epoch chosen by **val loss** |
| 4–6 | Train camera (15 ep), IR (12 ep), gas (100 ep, mini-batch) + baselines |
| 7 | Fusion (saves config E by default) + unplugged tests |
| 8 | Ablation table (3 seeds) |
| 9 | Grad-CAM figure for slides |
| 10 | Summary + `SpoilCheckPredictor` class |

**Extra cell added after the run** (retrains config D and saves `best_multimodal_fusion_D.pth`; needs `EMB` in memory):

```python
fusion_D, ep = train_fusion(use_ir=True, use_gas=True, p_mod_drop=0.0, seed=0)
torch.save(fusion_D.state_dict(), os.path.join(SAVE_DIR, "best_multimodal_fusion_D.pth"))
RESULTS.append(report("Model 4: Fusion D (sRGB+IR+Gas)", fusion_preds(fusion_D), y_te, "cm_fusion_D.png"))
```

If retraining: either change cell 7 to `p_mod_drop=0.0` or run the cell above, then deploy D.

---

## 7. Model tester (backend + frontend)

- **Start:** `Backend\run.bat`, then open http://127.0.0.1:8000.
- **Manual:** `cd Backend` and run `.venv\Scripts\python -m uvicorn app.main:app --host 127.0.0.1 --port 8000`.
- **Page:** camera photo (required), **IR toggle** (thermal image + t_min/t_max from the filename), **Methane toggle** (ADC, Rs, Rs/R0, ppm), "Load sample" with ✓/✗ against the true label; shows the fusion verdict, spoilage probability, top 5, and each standalone model's answer; flipping a toggle re-predicts.
- **Fruit selector (added 22 Nov 2026):** "Which fruit is it?" under the photo. When set, every model's logits are masked to that fruit's two classes, so it only decides fresh vs spoiled and can't confuse fruits. Default "Detect automatically".
- **Unsure warning:** shown when the top fruit's summed probability is < 0.45 or < 3× the runner-up fruit (`FRUIT_CONF_MIN`, `FRUIT_RATIO_MIN` in `predictor.py`), or when the spoiled probability is within 0.5 ± 0.15. Calibrated on 60 TR-6 photos (camera only): 0 false fruit warnings; a web photo of a mouldy tomato called "Mango 34%" is flagged. Reason it exists: on out-of-dataset photos the model sometimes confuses fruit types (tomato/guava → mango), because it learned from only ~3 fruits per variety on a white background.
- **API:** `GET /api/health` (also returns `fruits`) · `GET /api/sample?label=` · `GET /api/sample-file/{label}/{srgb|ir}/{filename}` · `POST /api/predict` (multipart: `srgb`, `use_ir`, `ir`, `t_min`, `t_max`, `use_gas`, `adc`, `rs`, `rs_r0`, `methane_ppm`, optional `fruit` e.g. `Tomato`). Each model summary now includes `fruit_confidence`, `fruit_candidates`, `unsure_fruit`, `unsure_state`.
- **Use another fusion file:** set `SPOILCHECK_FUSION=<file>.pth`.
- **Roll back to v1:** copy the six files from `Models/v1_backup/` into `Models/`.
- **Verified with v2 (22 Nov 2026):** models load in 2.4 s; 36 random dataset samples (3 per class) × 4 sensor combos → 35/36 correct in every combo (the miss: one spoiled gooseberry called fresh); 0.33 s per prediction on CPU. Samples may be training images, so this is a smoke test, not an accuracy figure. No backend code changes were needed for v2 (only a docstring in `model_defs.py`).

---

## 8. Next steps / open items

1. ~~Run the v2 notebook~~ — done; ~~update slides 6–7~~ — done (also slides 2, 3, 8, 9 and notes).
2. Add the v2 Grad-CAM figure (`gradcam_examples.png`) and confusion matrix (`cm_fusion_D.png`) from Drive to the slides.
3. Look into Tomato_Spoiled (recall 0.65), the weakest class.
3b. **Fruit confusion on new photos** (tomato/guava → mango on web/phone images). Short-term fix done (fruit selector + unsure warning). Proper fix: retrain the camera model with varied fruit photos (public datasets + 30–50 own photos per fruit), stronger crop/zoom/brightness augmentation (no hue jitter), ideally a separate fruit-type classifier followed by fresh/spoiled per fruit.
4. Grouped 5-fold CV and/or a leave-one-day-out test (`GROUP_LEVEL="date"`), for a stronger "no leakage" claim.
5. Slides: credit TR-6 explicitly, add a limitations slide (slide 8 partly covers it).
6. Optional: out-of-fold embeddings for the fusion training (fixes `best_epoch = 1`).
7. **Possible extensions discussed:**
   - **Days-left estimate:** label = first spoiled date − photo date, predicted as ranges (spoiled / 1–3 / 4–7 / >7 days) from frozen camera features, tested on held-out days. Reliable only for Tomato (44 fresh dates) and Gooseberry (18).
   - **More fruit classes, camera-only:** AgriFreshNET, Fresh & Rotten Fruits (Sultana), FruitVision. Report accuracy per source dataset.
   - **Days-left data:** Hass avocado ripening (4TU), FreshLab-Tomato (IEEE DataPort).
   - **Foresight-TR6+** (IEEE DataPort): an audited extension of TR-6, worth checking.
8. **Open questions for the mentor:** drop gas from the claims now that v2 shows it adds nothing? Add a leave-one-day-out test before submission?
