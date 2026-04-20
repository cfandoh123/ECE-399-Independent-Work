"""
=============================================================
 radar_shape_simple_cnn.py
 Simple CNN for Radar Shape Classification
 
 Task   : Classify shape only
 Classes: Circle, Square, Rectangle, Triangle, Oval (5)
 Input  : Range-Angle heatmap [512 × 256]
 Output : Shape class (0-4)

 Usage:
   python radar_shape_simple_cnn.py

 Requirements:
   pip install torch scipy numpy matplotlib seaborn scikit-learn

 Project: Radar Object Detection + ML (Junior IW)
=============================================================
"""

import scipy.io
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import time
import os
import csv
import random
import argparse

# ── Reproducibility / multi-seed configuration ───────────────
# Each run is trained once per seed so that test accuracy is
# reported as mean ± std across independent runs. With only
# ~2,160 samples, a single run has several percentage points of
# noise. Override from the CLI with --seeds 0,1,2.
SEEDS = [0, 1, 2, 3, 4]


def set_seed(seed):
    """Seed torch, numpy and random for one training run."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ── Device ──────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}\n")

# ── Class names — must match MATLAB label order ──────────────
CLASS_NAMES = ['Circle', 'Square', 'Rectangle', 'Triangle', 'Oval']
N_CLASSES   = len(CLASS_NAMES)
CLASS_COLORS = ['#3498DB', '#E74C3C', '#2ECC71', '#9B59B6', '#F39C12']

# ============================================================
#  SECTION 1: LOAD DATASET
# ============================================================

def load_dataset(filepath='radar_shapes_simple.mat'):
    """Load MATLAB dataset and return PyTorch DataLoaders."""

    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Dataset not found: {filepath}\n"
            f"Run radar_shape_simple.m in MATLAB first."
        )

    print(f"Loading: {filepath}")
    data = scipy.io.loadmat(filepath)

    def prep(X, Y):
        # [N × range × angle] → [N × 1 × range × angle]
        X = np.expand_dims(X.astype(np.float32), axis=1)
        Y = Y.flatten().astype(np.int64) - 1   # 0-indexed
        return torch.tensor(X), torch.tensor(Y)

    X_tr, Y_tr = prep(data['X_train'], data['Y_train'])
    X_vl, Y_vl = prep(data['X_val'],   data['Y_val'])
    X_te, Y_te = prep(data['X_test'],  data['Y_test'])

    print(f"Train : {X_tr.shape}  ({Y_tr.shape})")
    print(f"Val   : {X_vl.shape}  ({Y_vl.shape})")
    print(f"Test  : {X_te.shape}  ({Y_te.shape})")

    # Class distribution
    print("\nClass distribution (train):")
    for i, name in enumerate(CLASS_NAMES):
        n = (Y_tr == i).sum().item()
        print(f"  {i+1}. {name:12s}: {n} samples")

    # Larger batch reduces gradient/val-loss variance.
    # Val/test use a bigger batch since forward-only has no memory risk.
    BATCH      = 256
    EVAL_BATCH = 512
    train_dl = DataLoader(TensorDataset(X_tr, Y_tr),
                          batch_size=BATCH, shuffle=True, drop_last=True)
    val_dl   = DataLoader(TensorDataset(X_vl, Y_vl),
                          batch_size=EVAL_BATCH, shuffle=False)
    test_dl  = DataLoader(TensorDataset(X_te, Y_te),
                          batch_size=EVAL_BATCH, shuffle=False)

    return train_dl, val_dl, test_dl


# ============================================================
#  SECTION 2: CNN ARCHITECTURE
#
#  Simple and clean — 4 conv blocks + global average pooling
#  + 2 FC layers → 5 class output.
#
#  Input:  [B × 1 × 512 × 256]
#  Output: [B × 5]
# ============================================================

class ShapeCNN(nn.Module):
    """
    Simple 2D CNN for radar shape classification.
    
    Designed to be:
      - Easy to understand and explain
      - Easy to debug and modify
      - Fast enough to train on CPU in reasonable time
    """

    def __init__(self, n_classes=5):
        super().__init__()

        # ── Feature extractor ────────────────────────────────
        self.features = nn.Sequential(

            # Block 1 — detect broad shape structure
            # 7×7 kernel sees a large patch at once
            # good for picking up the overall signature shape
            nn.Conv2d(1,  16, kernel_size=7, padding=3),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),         # 512×256 → 256×128

            # Block 2 — detect mid-level features
            # spacing between scatterers, band patterns
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),         # 256×128 → 128×64

            # Block 3 — detect fine structure
            # individual peak shapes, edge sharpness
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),         # 128×64 → 64×32

            # Block 4 — high-level combinations
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),         # 64×32 → 32×16
        )

        # Global Average Pooling
        # Collapses spatial dims: [B×128×32×16] → [B×128]
        # More robust than Flatten for varying input sizes
        self.gap = nn.AdaptiveAvgPool2d(1)

        # ── Classifier ───────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)
        x = self.classifier(x)
        return x


# ============================================================
#  SECTION 3: TRAINING
# ============================================================

def train(model, train_dl, val_dl, n_epochs=40, lr=1e-3, patience=8):
    """Train with Adam, CrossEntropy, ReduceLROnPlateau, early stopping."""

    model     = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    # Cosine annealing — deterministic LR decay over n_epochs.
    # Replaces ReduceLROnPlateau, which reacted to noisy val loss
    # and introduced its own instability.
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs, eta_min=1e-6
    )

    history = {
        'train_loss'  : [], 'val_loss'    : [],
        'train_acc'   : [], 'val_acc'     : [],
        'val_loss_ema': [], 'val_acc_ema' : [],
    }
    ema_val_loss = None
    ema_val_acc  = None
    EMA_ALPHA    = 0.3          # 0 = fully smoothed, 1 = no smoothing

    best_val   = float('inf')
    best_wts   = None
    no_improve = 0

    print(f"\n{'='*60}")
    print(f"  Training ShapeCNN | {n_epochs} epochs | lr={lr}")
    print(f"{'='*60}")
    print(f"{'Ep':>4} | {'TrLoss':>8} | {'VaLoss':>8} | "
          f"{'TrAcc':>7} | {'VaAcc':>7} | {'LR':>9}")
    print(f"{'-'*60}")

    for ep in range(1, n_epochs+1):
        t0 = time.time()

        # ── Train ────────────────────────────────────────────
        model.train()
        tr_loss = tr_corr = tr_n = 0

        for Xb, Yb in train_dl:
            Xb, Yb = Xb.to(device), Yb.to(device)
            optimizer.zero_grad()
            out  = model(Xb)
            loss = criterion(out, Yb)
            loss.backward()
            optimizer.step()

            tr_loss += loss.item() * Xb.size(0)
            tr_corr += (out.argmax(1) == Yb).sum().item()
            tr_n    += Xb.size(0)

        tr_loss /= tr_n
        tr_acc   = tr_corr / tr_n

        # ── Validate ─────────────────────────────────────────
        model.eval()
        va_loss = va_corr = va_n = 0

        with torch.no_grad():
            for Xb, Yb in val_dl:
                Xb, Yb = Xb.to(device), Yb.to(device)
                out      = model(Xb)
                loss     = criterion(out, Yb)
                va_loss += loss.item() * Xb.size(0)
                va_corr += (out.argmax(1) == Yb).sum().item()
                va_n    += Xb.size(0)

        va_loss /= va_n
        va_acc   = va_corr / va_n

        # -- EMA smoothing of val metrics --
        # Damps per-epoch noise so training curves and early stopping
        # react to trends, not to single-batch spikes.
        if ema_val_loss is None:
            ema_val_loss = va_loss
            ema_val_acc  = va_acc
        else:
            ema_val_loss = EMA_ALPHA * va_loss + (1 - EMA_ALPHA) * ema_val_loss
            ema_val_acc  = EMA_ALPHA * va_acc  + (1 - EMA_ALPHA) * ema_val_acc

        scheduler.step()
        cur_lr = optimizer.param_groups[0]['lr']

        history['train_loss'].append(tr_loss)
        history['val_loss'].append(va_loss)
        history['train_acc'].append(tr_acc)
        history['val_acc'].append(va_acc)
        history['val_loss_ema'].append(ema_val_loss)
        history['val_acc_ema'].append(ema_val_acc)

        print(f"{ep:>4} | {tr_loss:>8.4f} | {va_loss:>8.4f} | "
              f"{tr_acc*100:>6.1f}% | {va_acc*100:>6.1f}% | "
              f"{cur_lr:>9.6f}  ({time.time()-t0:.1f}s)  "
              f"EMA_VaLoss={ema_val_loss:.4f}")

        # ── Early stopping (on smoothed val loss, not raw) ───
        if ema_val_loss < best_val:
            best_val = ema_val_loss
            best_wts = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"\n  Early stopping at epoch {ep}.")
                break

    model.load_state_dict(best_wts)
    print(f"\n  Best val loss: {best_val:.4f}")
    return model, history


# ============================================================
#  SECTION 4: EVALUATION
# ============================================================

def evaluate(model, test_dl):
    """Full test set evaluation with confusion matrix."""

    model.eval()
    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for Xb, Yb in test_dl:
            out = model(Xb.to(device))
            all_preds.extend(out.argmax(1).cpu().numpy())
            all_labels.extend(Yb.numpy())

    preds  = np.array(all_preds)
    labels = np.array(all_labels)
    acc    = np.mean(preds == labels)

    print(f"\n{'='*45}")
    print(f"  Test Accuracy: {acc*100:.2f}%")
    print(f"{'='*45}\n")
    print(classification_report(labels, preds, target_names=CLASS_NAMES))

    # Confusion matrix
    cm      = confusion_matrix(labels, preds)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle('Confusion Matrix — Radar Shape Classification',
                 fontsize=13, fontweight='bold')

    for ax, data, fmt, title in zip(
        axes,
        [cm, cm_norm],
        ['d', '.1%'],
        ['Counts', 'Normalised']
    ):
        sns.heatmap(data, annot=True, fmt=fmt, cmap='Blues',
                    xticklabels=CLASS_NAMES,
                    yticklabels=CLASS_NAMES,
                    ax=ax, linewidths=0.5, square=True)
        ax.set_xlabel('Predicted', fontsize=11)
        ax.set_ylabel('True',      fontsize=11)
        ax.set_title(title,        fontsize=11)

    plt.tight_layout()
    plt.savefig('confusion_matrix_simple.png', dpi=150,
                bbox_inches='tight')
    plt.show()

    return acc, preds, labels


# ============================================================
#  SECTION 5: TRAINING CURVES
# ============================================================

def plot_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle('Training History — Shape CNN', fontsize=13,
                 fontweight='bold')

    eps = range(1, len(history['train_loss'])+1)

    # Raw val curves in faded background; EMA curves in foreground.
    has_ema = 'val_loss_ema' in history and len(history['val_loss_ema']) > 0

    ax1.plot(eps, history['train_loss'], 'b-o', ms=3, label='Train', alpha=0.85)
    if has_ema:
        ax1.plot(eps, history['val_loss'],     'r-',  lw=1, alpha=0.3,
                 label='Val (raw)')
        ax1.plot(eps, history['val_loss_ema'], 'r-o', ms=3,
                 label='Val (EMA)')
    else:
        ax1.plot(eps, history['val_loss'], 'r-o', ms=3, label='Val')
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss')
    ax1.set_title('Loss'); ax1.legend(); ax1.grid(True)

    ax2.plot(eps, [a*100 for a in history['train_acc']],
             'b-o', ms=3, label='Train', alpha=0.85)
    if has_ema:
        ax2.plot(eps, [a*100 for a in history['val_acc']],
                 'r-', lw=1, alpha=0.3, label='Val (raw)')
        ax2.plot(eps, [a*100 for a in history['val_acc_ema']],
                 'r-o', ms=3, label='Val (EMA)')
    else:
        ax2.plot(eps, [a*100 for a in history['val_acc']],
                 'r-o', ms=3, label='Val')
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Accuracy'); ax2.legend()
    ax2.grid(True); ax2.set_ylim([0, 105])

    plt.tight_layout()
    plt.savefig('training_history_simple.png', dpi=150,
                bbox_inches='tight')
    plt.show()


# ============================================================
#  SECTION 6: SINGLE SAMPLE INFERENCE
#
#  This is the function you call on any new heatmap —
#  from simulation, Simulink Radar Toolbox, or real radar.
# ============================================================

def predict_shape(heatmap, model, true_label=None):
    """
    Predict the shape of a single radar heatmap.

    Args:
        heatmap   : numpy array [range_bins × angle_bins]
                    Any Range-Angle map normalised to [0,1]
        model     : trained ShapeCNN
        true_label: int (0-4) if known, for display only

    Returns:
        pred_name  : string — predicted shape name
        confidence : float  — confidence in prediction (0-1)
        all_probs  : dict   — probability per class
    """
    model.eval()

    # Preprocess
    hm = heatmap.astype(np.float32)
    if hm.max() > 0:
        hm = hm / hm.max()

    # Resize to training dimensions if needed
    if hm.shape != (512, 256):
        from scipy.ndimage import zoom
        hm = zoom(hm, (512/hm.shape[0], 256/hm.shape[1]), order=1)
        hm = hm / (hm.max() + 1e-9)

    tensor = torch.tensor(hm).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1).cpu().numpy()[0]

    pred_idx   = int(np.argmax(probs))
    pred_name  = CLASS_NAMES[pred_idx]
    confidence = float(probs[pred_idx])
    all_probs  = {CLASS_NAMES[i]: float(probs[i])
                  for i in range(N_CLASSES)}

    # ── Console output ───────────────────────────────────────
    print(f"\n{'='*45}")
    print(f"  RADAR SHAPE PREDICTION")
    print(f"{'='*45}")
    if true_label is not None:
        true_name = CLASS_NAMES[true_label]
        correct   = '✓' if pred_idx == true_label else '✗'
        print(f"  True shape  : {true_name}")
        print(f"  Prediction  : {pred_name} {correct}")
    else:
        print(f"  Prediction  : {pred_name}")
    print(f"  Confidence  : {confidence*100:.1f}%")
    print(f"\n  All probabilities:")
    for name, prob in sorted(all_probs.items(),
                              key=lambda x: -x[1]):
        bar = '█' * int(prob * 30)
        print(f"    {name:12s}: {prob*100:5.1f}%  {bar}")
    print(f"{'='*45}\n")

    # ── Visualisation ────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    title_str = f"Prediction: {pred_name} ({confidence*100:.1f}%)"
    if true_label is not None:
        correct = pred_idx == true_label
        title_str += f" — {'CORRECT ✓' if correct else 'WRONG ✗'}"
    fig.suptitle(title_str, fontsize=13, fontweight='bold')

    # Heatmap
    im = ax1.imshow(20*np.log10(hm + 1e-6),
                    cmap='jet', aspect='auto', vmin=-40, vmax=0)
    plt.colorbar(im, ax=ax1, label='dB')
    ax1.set_xlabel('Angle bins'); ax1.set_ylabel('Range bins')
    ax1.set_title('Input Heatmap')

    # Probability bars
    names = list(all_probs.keys())
    probs_list = list(all_probs.values())
    colors = [CLASS_COLORS[CLASS_NAMES.index(n)] for n in names]
    bars   = ax2.barh(names, [p*100 for p in probs_list],
                      color=colors, edgecolor='white', height=0.6)
    bars[pred_idx].set_edgecolor('black')
    bars[pred_idx].set_linewidth(2.5)
    ax2.set_xlabel('Confidence (%)'); ax2.set_xlim([0, 110])
    ax2.set_title('Class Probabilities'); ax2.grid(axis='x', alpha=0.3)
    ax2.axvline(50, color='gray', linestyle='--', alpha=0.4)
    for i, (b, p) in enumerate(zip(bars, probs_list)):
        ax2.text(b.get_width()+1, b.get_y()+b.get_height()/2,
                 f'{p*100:.1f}%', va='center', fontsize=9)

    plt.tight_layout()
    plt.savefig('shape_prediction.png', dpi=150, bbox_inches='tight')
    plt.show()

    return pred_name, confidence, all_probs


# ============================================================
#  SECTION 6b: MULTI-SEED HELPERS
# ============================================================

def per_class_accuracy(preds, labels):
    """Return accuracy dict keyed by class name."""
    out = {}
    for i, name in enumerate(CLASS_NAMES):
        mask = labels == i
        out[name] = (float(np.mean(preds[mask] == labels[mask]))
                     if mask.sum() > 0 else 0.0)
    return out


def plot_history_multi(histories, seeds):
    """Overlay training curves across seeds; mean curve solid on top."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f'Training History (seeds={seeds})',
                 fontsize=13, fontweight='bold')

    min_len = min(len(h['train_loss']) for h in histories)
    eps     = range(1, min_len + 1)

    has_ema = all('val_loss_ema' in h and len(h['val_loss_ema']) > 0
                  for h in histories)

    def stack(key):
        return np.stack([np.array(h[key][:min_len]) for h in histories])

    tr_loss = stack('train_loss')
    va_loss = stack('val_loss_ema' if has_ema else 'val_loss')
    tr_acc  = stack('train_acc')  * 100
    va_acc  = stack('val_acc_ema' if has_ema else 'val_acc') * 100

    for h in histories:
        ax1.plot(eps, h['train_loss'][:min_len], 'b-', lw=1, alpha=0.25)
        ax1.plot(eps, (h.get('val_loss_ema') or h['val_loss'])[:min_len],
                 'r-', lw=1, alpha=0.25)
        ax2.plot(eps, [a*100 for a in h['train_acc'][:min_len]],
                 'b-', lw=1, alpha=0.25)
        ax2.plot(eps, [a*100 for a in
                       (h.get('val_acc_ema') or h['val_acc'])[:min_len]],
                 'r-', lw=1, alpha=0.25)

    ax1.plot(eps, tr_loss.mean(0), 'b-', lw=2, label='Train (mean)')
    ax1.plot(eps, va_loss.mean(0), 'r-', lw=2,
             label='Val (EMA mean)' if has_ema else 'Val (mean)')
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss')
    ax1.set_title('Loss'); ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2.plot(eps, tr_acc.mean(0), 'b-', lw=2, label='Train (mean)')
    ax2.plot(eps, va_acc.mean(0), 'r-', lw=2,
             label='Val (EMA mean)' if has_ema else 'Val (mean)')
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Accuracy'); ax2.legend(); ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 105])

    plt.tight_layout()
    plt.savefig('training_history_simple.png', dpi=150,
                bbox_inches='tight')
    plt.show()


def save_results_csv(accuracies, per_class_seed, seeds,
                     filepath='results_per_seed_single.csv'):
    """Dump per-seed test accuracies to CSV."""
    rows = []
    for i, seed in enumerate(seeds):
        row = {'seed': seed, 'test_accuracy': accuracies[i]}
        for name in CLASS_NAMES:
            row[f'acc_{name}'] = per_class_seed[name][i]
        rows.append(row)

    if not rows:
        return

    with open(filepath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved per-seed results: {filepath}")


# ============================================================
#  SECTION 7: MAIN
# ============================================================

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--file', type=str, default='radar_shapes_simple.mat',
        help='Path to the .mat dataset file',
    )
    parser.add_argument(
        '--seeds', type=str, default=None, metavar='LIST',
        help='Comma-separated list of RNG seeds, e.g. "0,1,2". '
             'Defaults to the module-level SEEDS list.',
    )
    args = parser.parse_args()

    if args.seeds is not None:
        try:
            seeds_run = [int(s.strip()) for s in args.seeds.split(',') if s.strip()]
        except ValueError:
            raise SystemExit(f"Invalid --seeds value: {args.seeds!r}")
        if not seeds_run:
            raise SystemExit("--seeds must list at least one integer")
    else:
        seeds_run = SEEDS

    print("=" * 62)
    print(f"  RADAR SHAPE CLASSIFICATION — SINGLE ANTENNA, MULTI-SEED")
    print(f"  file  : {args.file}")
    print(f"  seeds : {seeds_run}")
    print("=" * 62)

    # Per-seed accumulators
    seed_accs      = []
    seed_histories = []
    seed_per_class = {name: [] for name in CLASS_NAMES}
    best_acc       = -1.0
    best_seed      = None
    best_model     = None

    for seed in seeds_run:
        print(f"\n---- seed {seed} ----")
        set_seed(seed)

        # Reload per seed so the DataLoader shuffle also varies
        train_dl, val_dl, test_dl = load_dataset(args.file)

        model  = ShapeCNN(n_classes=N_CLASSES)
        if seed == seeds_run[0]:
            params = sum(p.numel() for p in model.parameters())
            print(f"Model: ShapeCNN  |  Parameters: {params:,}")

        model, history = train(
            model, train_dl, val_dl,
            n_epochs = 40,
            lr       = 1e-3,
            patience = 8,
        )

        acc, preds, labels = evaluate(model, test_dl)
        pc = per_class_accuracy(preds, labels)

        seed_accs.append(acc)
        seed_histories.append(history)
        for name in CLASS_NAMES:
            seed_per_class[name].append(pc.get(name, 0.0))

        if acc > best_acc:
            best_acc   = acc
            best_seed  = seed
            best_model = model

        print(f"seed={seed}  |  acc={acc*100:.2f}%")

    # Aggregate
    mean_acc = float(np.mean(seed_accs))
    std_acc  = (float(np.std(seed_accs, ddof=1))
                if len(seed_accs) > 1 else 0.0)

    print(f"\n{'='*62}")
    print(f"  SUMMARY  ({len(seeds_run)} seeds)")
    print(f"{'='*62}")
    print(f"  Mean test accuracy : {mean_acc*100:.2f}% ± {std_acc*100:.2f}%")
    print(f"  Best seed          : {best_seed}  ({best_acc*100:.2f}%)")
    print(f"  Per-seed accuracies:")
    for s, a in zip(seeds_run, seed_accs):
        print(f"    seed {s}: {a*100:.2f}%")
    print(f"{'='*62}\n")

    # Plot overlaid training curves across seeds
    plot_history_multi(seed_histories, seeds_run)

    # Save best model
    torch.save({
        'model_state_dict'  : best_model.state_dict(),
        'class_names'       : CLASS_NAMES,
        'n_classes'         : N_CLASSES,
        'best_test_accuracy': best_acc,
        'best_seed'         : best_seed,
        'mean_test_accuracy': mean_acc,
        'std_test_accuracy' : std_acc,
        'seeds'             : seeds_run,
    }, 'radar_shape_simple_model.pt')
    print("Model saved: radar_shape_simple_model.pt")

    # Dump per-seed CSV for the writeup
    save_results_csv(seed_accs, seed_per_class, seeds_run)

    # Demo inference using the best model
    print("\n--- Demo: Predict one test sample (best seed) ---")
    import scipy.io
    data      = scipy.io.loadmat(args.file)
    X_test_np = data['X_test'].astype(np.float32)
    Y_test_np = data['Y_test'].flatten().astype(int) - 1

    print("\nRunning inference on one sample per class:")
    for cls in range(N_CLASSES):
        idx = np.where(Y_test_np == cls)[0][0]
        hm  = X_test_np[idx]   # [512 × 256]
        predict_shape(hm, best_model, true_label=cls)

    # Step 8 — How to use on Simulink / real radar data
    print("\n" + "="*55)
    print("  HOW TO TEST ON SIMULINK OR REAL RADAR DATA")
    print("="*55)
    print("""
  1. Generate heatmap in MATLAB using either:
       - radar_shape_simple.m  (pure simulation)
       - radar_toolbox_datagen.m  (Simulink Radar Toolbox)
       - process_real_data.m  (real TI radar hardware)

  2. Save as .mat:
       save('my_heatmap.mat', 'RA_normalised')

  3. Load and predict in Python:
       import scipy.io
       data = scipy.io.loadmat('my_heatmap.mat')
       hm   = data['RA_normalised']
       
       model = ShapeCNN(n_classes=5)
       ckpt  = torch.load('radar_shape_simple_model.pt')
       model.load_state_dict(ckpt['model_state_dict'])
       
       pred, conf, probs = predict_shape(hm, model)
       print(f'Shape: {pred} ({conf*100:.1f}% confidence)')
    """)
