# FGSVM: Fuzzy Granular Support Vector Machine

This repository is a reimplementation of the **Fuzzy Granular Support Vector Machine (FGSVM) [IEEE TITS 2025]** proposed in the following paper:

```bibtex
@article{lai2025fuzzy,
  title={A Fuzzy Granular Support Vector Machine for Network Traffic Anomaly Detection},
  author={Lai, Rong and Chen, Yumin and Li, Jinhai},
  journal={IEEE Transactions on Intelligent Transportation Systems},
  year={2025},
  publisher={IEEE}
}
```

We added **reference preprocessing code** (multi-class → binary conversion, train/test split, `.npy` saving) and **feature standardization** in the pipeline to help students and beginners run experiments more easily on public datasets (e.g. Dry Bean).

**All code comments are written in Chinese** for better accessibility to Chinese-speaking users.

## Core Files

| File              | Description                                          |
|-------------------|------------------------------------------------------|
| `granular.py`     | Core FGSVM implementation (fuzzy granular mechanism) |
| `main.py`         | Main script — training, prediction & evaluation      |
| `sample.py`       | Imbalanced data sampling / resampling utilities      |
| `smo.py`          | SMO-based SVM (GPU acceleration via cuML if available) |

## Quick Start (Dry Bean Dataset Example)

1. Prepare dataset  
   Example: `Dry_Bean_Dataset.xlsx`

2. Run preprocessing (e.g. `dbPreProcess.py` or write your own):
   - Read Excel/CSV
   - Convert multi-class to binary (positive class as defined in paper; labels preferably **+1 / -1**)
   - Follow the original class appearance order in the file
   - Split train/test (recommended 7:3 or 8:2; 5-fold cross-validation also supported)
   - Save as NumPy files: `train.npy` / `test.npy`  
     Format: last column = label

3. (Optional) Handle severe class imbalance:
   ```bash
   python sample.py
   ```
   → produces balanced `sam_train.npy`

4. Start training & evaluation:
   ```bash
   python main.py
   ```
   The script will:
   - Load `train.npy` / `test.npy` (or `sam_train.npy`)
   - Apply feature standardization
   - Train the FGSVM model
   - Evaluate on the test set

## Important Notes

- **GPU Acceleration**  
  Requires CUDA + cuML installed → `smo.py` will automatically use GPU acceleration (much faster).  
  Without GPU/cuML, falls back to pure Python SMO (slower).

- **Hyperparameters**  
  Mainly tuned in `main.py` (C, gamma, granularity thresholds, etc.).

- **Common Issues & Quick Fixes**
  - Labels not ±1 → verify binary conversion logic  
  - Shape mismatch error → check `.npy` format (should be samples × (features + 1))  
  - Unexpected positive/negative ratio → confirm positive class definition and order

Once `.npy` files are ready, simply run `python main.py` to start experiments.

For questions, collaborations, or issues about this repo (especially the added preprocessing code):  
zhwang@nuist.edu.cn

Contributions, bug reports, and dataset support PRs are welcome!
