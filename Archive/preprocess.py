"""
preprocess.py
-------------
Loads radar_dataset.mat (MATLAB v7.3 / HDF5 format), crops the feature
vectors to the relevant range window, splits into train/val/test, and
saves a compact radar_preprocessed.npz for use by classify.py.

Run from the Radar_Simulation/ directory:
    python phase2/preprocess.py

Requirements:
    pip install h5py numpy scikit-learn matplotlib
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os

# ── Radar parameters (must match fmcw_simulation.m) ───────────────────────────
C          = 3e8
BW         = 1e9
SWEEP_TIME = 100e-6
FS         = 2 * BW
MAX_RANGE  = 6.0     # metres — crop features beyond this range

CLASS_NAMES = ['ball', 'cup', 'chair']

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR     = os.path.dirname(SCRIPT_DIR)
MAT_PATH     = os.path.join(ROOT_DIR, 'radar_dataset.mat')
OUTPUT_PATH  = os.path.join(SCRIPT_DIR, 'radar_preprocessed.npz')

# ── Step 1: Load .mat (HDF5) ───────────────────────────────────────────────────
print(f"Loading {MAT_PATH}  (this may take a moment for a large file)...")

with h5py.File(MAT_PATH, 'r') as f:
    # MATLAB stores matrices column-major, so HDF5 dimensions are transposed.
    # In MATLAB: features is (n_samples x n_features)
    # In HDF5:   features is (n_features x n_samples)  -> transpose after load
    features   = f['features'][:].T          # (n_samples, n_features)
    labels     = f['labels'][:].flatten().astype(np.int32)      # (n_samples,)
    ranges     = f['ranges'][:].flatten()
    azimuths   = f['azimuths'][:].flatten()
    elevations = f['elevations'][:].flatten()
    rng_ax     = f['rng_ax'][:].flatten()    # range axis in metres

n_samples, n_features = features.shape
print(f"  Loaded : {n_samples} samples  x  {n_features} features")
print(f"  Labels : {np.sum(labels==1)} ball | {np.sum(labels==2)} cup | {np.sum(labels==3)} chair")

# ── Step 2: Crop to relevant range window ─────────────────────────────────────
# Targets are at 1-5 m; beyond 6 m is all noise.
crop_idx  = np.searchsorted(rng_ax, MAX_RANGE) + 1
features  = features[:, :crop_idx]
rng_ax    = rng_ax[:crop_idx]

print(f"\nCropped features : {n_features} -> {crop_idx} bins  (0 to {MAX_RANGE} m)")
print(f"Memory reduction : {n_samples*n_features*8/1e6:.0f} MB -> {n_samples*crop_idx*8/1e3:.1f} KB")

# ── Step 3: Sanity-check — plot one profile per class ─────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(9, 7))
for cls_idx, (cls_name, ax) in enumerate(zip(CLASS_NAMES, axes), start=1):
    mask = (labels == cls_idx) & (ranges == 1.0) & (azimuths == 0) & (elevations == 0)
    sample = features[mask][0]
    ax.plot(rng_ax, sample)
    ax.set_xlim([0, MAX_RANGE])
    ax.set_title(f'{cls_name}  (R=1 m, az=0°, el=0°)')
    ax.set_xlabel('Range (m)')
    ax.set_ylabel('Norm. Amplitude')
    ax.grid(True)
fig.suptitle('Cropped Range Profiles — One Sample Per Class')
fig.tight_layout()
plot_path = os.path.join(SCRIPT_DIR, 'sanity_check_profiles.png')
plt.savefig(plot_path, dpi=120)
plt.show()
print(f"\nSanity-check plot saved: {plot_path}")

# ── Step 4: Train / Val / Test split (70 / 15 / 15, stratified) ───────────────
X_tr, X_tmp, y_tr, y_tmp, r_tr, r_tmp = train_test_split(
    features, labels, ranges,
    test_size=0.30, stratify=labels, random_state=42
)
X_val, X_te, y_val, y_te, r_val, r_te = train_test_split(
    X_tmp, y_tmp, r_tmp,
    test_size=0.50, stratify=y_tmp, random_state=42
)

print(f"\nSplit  ->  train: {len(y_tr)}  |  val: {len(y_val)}  |  test: {len(y_te)}")

# ── Step 5: Save preprocessed dataset ─────────────────────────────────────────
np.savez(OUTPUT_PATH,
         X_train=X_tr,  y_train=y_tr,
         X_val=X_val,   y_val=y_val,
         X_test=X_te,   y_test=y_te,
         rng_ax=rng_ax,
         class_names=np.array(CLASS_NAMES))

size_kb = os.path.getsize(OUTPUT_PATH) / 1e3
print(f"\nSaved : {OUTPUT_PATH}  ({size_kb:.1f} KB)")
print("Done. Run classify.py next.")
