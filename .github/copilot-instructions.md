# Copilot instructions — AI-Driven-Image-Forgery-Detection

Purpose: Provide a compact, actionable guide so an AI coding agent (or new developer) can start contributing immediately, with emphasis on the image forgery module (3-day sprint).

Repository facts (what's present)
- `README.md` — high-level architecture and desired components.
- `img.py` — simple LC25000 dataset splitter (shuffles, 80/20 split). Use as a template for other datasets.
- `data/` — contains `LC25000/`, `CASIA2/` (authentic/tampered + ground-truth masks), and `TCGA-COAD/` (DICOM CTs) plus `metadata.csv`.

Gaps discovered (important)
- The README references `backend/`, `frontend/`, `main.py`, `utils/`, `models/`, and `tests/`, but these directories/files are not present. Create minimal placeholders to match README when integrating.
- No `requirements.txt` content review done here — verify for TF vs PyTorch and CUDA versions before GPU runs.

Top-priority work: Image forgery detection (3 days)
- Day 1 (Data preparation & baseline):
  - Implement `utils/ela.py`: function to compute ELA given an input image and save output (consistent size/format).
  - Implement `utils/phash.py`: compute pHash and store results in `data/phash.csv` with columns [filename, phash_hex].
  - Add a deterministic splitter for CASIA (seeded) and ensure `img.py` uses a seed.
  - Create `datasets/image_dataset.py` (PyTorch Dataset recommended) that returns (original, ela, phash_feature, label).
  - Implement `training/train_baseline.py` using a small ResNet (torchvision) on original images. Verify training runs for 1 epoch on CPU.

- Day 2 (Two-stream model + evaluation):
  - Implement `models/two_stream.py` (two branches: RGB and ELA -> feature concat -> MLP). Accept optional phash numeric input.
  - Implement `training/train_two_stream.py` and `evaluation/eval.py` that compute AUC, precision/recall, confusion matrix, and optionally localization IoU using CASIA masks.
  - Produce a short report (markdown) under `docs/experiments/image_forgery_day2.md` with metrics and sample Grad-CAM visualizations.

- Day 3 (Inference API + optional GAN augmentation):
  - Add `inference/forgery_infer.py` — loads model checkpoint and returns JSON { forgery_prob, phash, explain_map_path } for a single image.
  - Add a minimal Flask route `backend/routes/forgery.py` exposing `/api/forgery` (POST multipart/form-data). Keep it lightweight and CPU-friendly initially.
  - (Optional) Add simple GAN augmentation script `augmentation/simple_gan.py` or implement cut-and-paste tampering to expand the tampered class.

Concrete conventions to follow (project-specific)
- Data: store preprocessed images under `data/processed/<dataset>/<split>/<class>/` and ELA under `data/ela/<dataset>/<split>/<class>/`.
- Models & checkpoints: `models/<model_name>/<checkpoint>.pth` and training logs under `logs/<run_name>/`.
- Scripts: put CLI-style training/inference scripts under `training/` and `inference/` respectively; use argparse for reproducibility.
- Notebooks: `notebooks/` only for experiments; production code must be importable modules.

Quick examples (PowerShell)
```powershell
# deterministic split
$env:PYTHONHASHSEED = 0
python - <<'PY'
import random
random.seed(42)
print('seeded')
PY

# split LC25000
python img.py

# run a baseline training (example placeholder)
python training/train_baseline.py --data data/LC25000 --epochs 5 --batch 32 --out models/baseline
```

Files to read first when contributing
- `img.py` — shows dataset splitting approach.
- `data/CASIA2/` — contains tampered/authentic images plus ground-truth masks useful for localization.
- `data/LC25000/` — ready-made histopathology dataset for quick prototyping.

Next immediate tasks for the team (pick one to start now)
1. Implement `utils/ela.py` and `utils/phash.py` (fast, deterministic, unit-tested functions).
2. Implement `datasets/image_dataset.py` (PyTorch) and a tiny `training/train_baseline.py` (1-epoch smoke test).
3. Create `backend/` placeholder with `routes/forgery.py` so frontend teams can integrate early.

Questions for you (need answers to finalize scaffolding)
- Do you prefer PyTorch or TensorFlow? Do you have CUDA-enabled GPU(s) available? (affects model/backbone choices)
- Which dataset(s) should be treated as authoritative for evaluation (CASIA2 or LC25000 or the TCGA subset)?

If you confirm preferences, I will create the initial utility files (ELA, pHash, deterministic splitter) and a baseline training script and run a CPU smoke test.
