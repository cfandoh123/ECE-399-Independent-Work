"""
=============================================================
 inspect_heatmaps.py
 Interactive heatmap viewer for the trained shape CNN

 Purpose
 -------
 Answer the question "why is this class getting misclassified?"
 by looking directly at the radar Range-Angle heatmaps the
 network sees, overlaid with its predicted class probabilities.

 Typical workflows
 -----------------
 1) Browse every Rectangle heatmap that N_ant=16 got wrong and
    compare it against one that it got right:
        python inspect_heatmaps.py \
            --mat results/radar_shapes_N16.mat \
            --model results/radar_shape_model_N16.pt \
            --shape Rectangle --only-errors

 2) Grid-compare the same sample across all antenna counts (needs
    the matching .mat + .pt files for each N):
        python inspect_heatmaps.py --mat-dir . --model-dir results --shape Triangle --sample 0 --compare-n

 3) Save a per-class "correct vs wrong" montage for a writeup:
        python inspect_heatmaps.py --mat radar_shapes_N4.mat --model radar_shape_model_N4.pt --shape Rectangle --montage --n-examples 6 --save rect_N4_montage.png

 The script never modifies the model — it is read-only analysis.

 Requirements:
   pip install torch scipy h5py numpy matplotlib
=============================================================
"""

import argparse
import glob
import os
import re
import sys

import h5py
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────────────
#  Class labels — must match the MATLAB generator
# ─────────────────────────────────────────────────────────────
CLASS_NAMES  = ['Circle', 'Square', 'Rectangle', 'Triangle', 'Oval']
N_CLASSES    = len(CLASS_NAMES)
CLASS_COLORS = ['#3498DB', '#E74C3C', '#2ECC71', '#9B59B6', '#F39C12']

# Expected heatmap dimensions set by the MATLAB data generator.
# The trained ShapeCNN was fed (N, 1, NFFT_R, NFFT_A) tensors, so any
# .mat file we load must be forced into that orientation before
# inference or the CNN's learned features point the wrong way and
# accuracy collapses to chance.
NFFT_R = 512   # range bins (rows)
NFFT_A = 256   # angle bins (cols)


# ─────────────────────────────────────────────────────────────
#  Model definition — copied verbatim from the trainer so this
#  script can be run without importing the training module.
#  Keep in sync with radar_obj_classifier_cnn_multiple_antenna.py.
# ─────────────────────────────────────────────────────────────
class ShapeCNN(nn.Module):
    def __init__(self, n_classes=5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=7, padding=3),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
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
        return self.classifier(x)


# ─────────────────────────────────────────────────────────────
#  Dataset loading — supports both scipy v7 and HDF5 v7.3 .mat
# ─────────────────────────────────────────────────────────────
def _canonicalise_X(X):
    """
    Force X into the orientation the CNN was trained on: (N, NFFT_R, NFFT_A).

    Depending on the MATLAB save version and the path the loader takes
    (scipy vs h5py) the same file can come back shaped as
    (N, 512, 256) or (N, 256, 512). Feeding the wrong orientation
    silently drops accuracy to chance (≈20 % for 5 classes), because
    the CNN still runs but the features it learned no longer line up
    with the input axes. This helper picks the right transposition so
    the rest of the script and the model see (N, range, angle).
    """
    if X.ndim != 3:
        return X, False

    n, h, w = X.shape
    if h == NFFT_R and w == NFFT_A:
        return X, False                                  # already correct
    if h == NFFT_A and w == NFFT_R:
        return np.ascontiguousarray(X.transpose(0, 2, 1)), True
    # Last-ditch fallback: the larger spatial dim is almost always
    # the range axis (512 vs 256). Transpose so the bigger dim comes
    # first. This handles future generators that might change NFFT.
    if h < w:
        return np.ascontiguousarray(X.transpose(0, 2, 1)), True
    return X, False


def _load_mat(path):
    """
    Returns (X_test, Y_test, range_axis, ang_axis, n_ant).
    X_test is canonicalised to shape (N, 512, 256); labels are 0-indexed.
    Range/angle axes are reordered to match X's orientation.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    try:
        d = scipy.io.loadmat(path)
        X = d['X_test'].astype(np.float32)
        Y = d['Y_test'].flatten().astype(np.int64) - 1
        range_axis = d['range_axis'].flatten() if 'range_axis' in d else None
        ang_axis   = d['ang_axis'].flatten()   if 'ang_axis'   in d else None
        n_ant      = int(np.array(d['N_ant']).flatten()[0]) if 'N_ant' in d else None
    except NotImplementedError:
        with h5py.File(path, 'r') as f:
            X = np.array(f['X_test'], dtype=np.float32)
            Y = np.array(f['Y_test'], dtype=np.int64)
            if X.ndim == 3:
                X = X.transpose(2, 0, 1)
            Y = Y.flatten() - 1
            range_axis = np.array(f['range_axis']).flatten() if 'range_axis' in f else None
            ang_axis   = np.array(f['ang_axis']).flatten()   if 'ang_axis'   in f else None
            n_ant = int(np.array(f['N_ant']).flatten()[0]) if 'N_ant' in f else None

    if n_ant is None:
        m = re.search(r'N(\d+)', os.path.basename(path))
        n_ant = int(m.group(1)) if m else -1

    shape_before = X.shape
    X, transposed = _canonicalise_X(X)
    if transposed:
        print(f'  NOTE: X_test was shape {shape_before}; transposed to '
              f'{X.shape} to match the model (range × angle).')
        # Keep axis ↔ dim correspondence after transpose: range on dim 1
        # (size 512), angle on dim 2 (size 256). Swap the saved axes if
        # they came back tagged to the pre-transpose orientation.
        if (range_axis is not None and ang_axis is not None
                and len(range_axis) != NFFT_R and len(ang_axis) != NFFT_A):
            range_axis, ang_axis = ang_axis, range_axis

    return X, Y, range_axis, ang_axis, n_ant


def _load_model(path, device):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    ckpt  = torch.load(path, map_location=device, weights_only=False)
    model = ShapeCNN(n_classes=ckpt.get('n_classes', N_CLASSES))
    model.load_state_dict(ckpt['model_state_dict'])
    return model.to(device).eval(), ckpt


# ─────────────────────────────────────────────────────────────
#  Inference helpers
# ─────────────────────────────────────────────────────────────
def _predict_batch(model, X, device, batch=64):
    """Run X (N,512,256) through the model. Returns (preds, probs)."""
    preds_all = []
    probs_all = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            x = torch.from_numpy(X[i:i+batch]).unsqueeze(1).to(device)
            logits = model(x)
            p = torch.softmax(logits, dim=1).cpu().numpy()
            preds_all.append(p.argmax(1))
            probs_all.append(p)
    return np.concatenate(preds_all), np.concatenate(probs_all, axis=0)


def _resolve_shape(name):
    """Accept 'triangle', 'Triangle', 'TRI', '3', etc."""
    if isinstance(name, int) or name.isdigit():
        idx = int(name)
        if 0 <= idx < N_CLASSES:
            return idx
        raise ValueError(f"Class index {idx} out of range 0..{N_CLASSES-1}")
    want = name.strip().lower()
    for i, n in enumerate(CLASS_NAMES):
        if n.lower().startswith(want):
            return i
    raise ValueError(f"Unknown class: {name!r}. Try one of {CLASS_NAMES}.")


# ─────────────────────────────────────────────────────────────
#  Rendering
# ─────────────────────────────────────────────────────────────
def _render_panel(ax, hm, range_axis=None, ang_axis=None,
                  vmin=-40, vmax=0, title=None, mark_peak=False):
    """Render one RA heatmap panel in dB."""
    db = 20 * np.log10(hm + 1e-6)
    H, W = db.shape

    # Only use the saved axis vectors when their lengths line up with
    # the heatmap — otherwise fall back to bin indices so a stale or
    # mismatched axis never crashes rendering.
    axes_ok = (
        range_axis is not None
        and ang_axis is not None
        and len(range_axis) == H
        and len(ang_axis) == W
        and np.all(np.isfinite(ang_axis))
        and np.all(np.isfinite(range_axis))
    )

    if axes_ok:
        extent = [float(np.nanmin(ang_axis)), float(np.nanmax(ang_axis)),
                  float(np.nanmin(range_axis)), float(np.nanmax(range_axis))]
        im = ax.imshow(db, cmap='jet', aspect='auto',
                       origin='lower', extent=extent,
                       vmin=vmin, vmax=vmax)
        ax.set_xlabel('Angle (°)')
        ax.set_ylabel('Range (m)')
    else:
        im = ax.imshow(db, cmap='jet', aspect='auto',
                       vmin=vmin, vmax=vmax)
        ax.set_xlabel('Angle bin')
        ax.set_ylabel('Range bin')

    if mark_peak:
        r, c = np.unravel_index(np.argmax(db), db.shape)
        if axes_ok:
            ax.plot(ang_axis[c], range_axis[r], 'wx', ms=8, mew=2)
        else:
            ax.plot(c, r, 'wx', ms=8, mew=2)

    if title:
        ax.set_title(title, fontsize=10)
    return im


def _prob_bar(ax, probs, true_idx, pred_idx):
    """Horizontal probability bar chart next to a heatmap panel."""
    y = np.arange(N_CLASSES)
    bars = ax.barh(y, probs * 100,
                   color=CLASS_COLORS, edgecolor='white', height=0.6)
    bars[pred_idx].set_edgecolor('black')
    bars[pred_idx].set_linewidth(2.5)
    ax.set_yticks(y)
    ax.set_yticklabels(CLASS_NAMES, fontsize=9)
    ax.set_xlim([0, 115])
    ax.set_xlabel('Prob (%)')
    ax.grid(axis='x', alpha=0.3)
    ax.axvline(50, color='gray', linestyle='--', alpha=0.4)
    for b, p in zip(bars, probs):
        ax.text(b.get_width() + 1,
                b.get_y() + b.get_height() / 2,
                f'{p*100:.1f}%', va='center', fontsize=8)
    # Highlight true class on the y-axis label
    for i, lbl in enumerate(ax.get_yticklabels()):
        if i == true_idx:
            lbl.set_fontweight('bold')
            lbl.set_color('black')


# ─────────────────────────────────────────────────────────────
#  Main rendering modes
# ─────────────────────────────────────────────────────────────
def view_examples(X, Y, preds, probs, class_idx,
                  range_axis, ang_axis, n_ant,
                  only_errors=False, max_examples=6, save=None):
    """
    Render a grid of examples for one class, one per row:
        [heatmap]   [class-probability bar chart]

    Rows are sorted so that the most confidently wrong examples
    appear first when only_errors=True (i.e. easiest to inspect).
    """
    mask = Y == class_idx
    if only_errors:
        mask = mask & (preds != Y)

    idxs = np.where(mask)[0]
    if len(idxs) == 0:
        kind = 'misclassified ' if only_errors else ''
        print(f"No {kind}{CLASS_NAMES[class_idx]} samples found in this test set.")
        return

    # If we're looking at errors, put the highest-confidence wrong
    # predictions first because those are the most informative.
    if only_errors:
        pred_confidence = probs[np.arange(len(Y)), preds]
        idxs = idxs[np.argsort(-pred_confidence[idxs])]

    idxs = idxs[:max_examples]
    n    = len(idxs)

    fig, axes = plt.subplots(n, 2, figsize=(10, 2.6 * n),
                             gridspec_kw={'width_ratios': [3, 1]},
                             squeeze=False)
    tag = 'ERRORS' if only_errors else 'SAMPLES'
    fig.suptitle(
        f'{CLASS_NAMES[class_idx]} — {tag} — N_ant={n_ant}  '
        f'(showing {n})',
        fontsize=13, fontweight='bold'
    )

    for row, idx in enumerate(idxs):
        hm = X[idx]
        if hm.ndim == 3 and hm.shape[0] == 1:
            hm = hm[0]
        pred = int(preds[idx])
        true = int(Y[idx])
        verdict = 'CORRECT' if pred == true else 'WRONG'
        conf    = float(probs[idx, pred])

        title = (f'idx={idx}  true={CLASS_NAMES[true]}  '
                 f'pred={CLASS_NAMES[pred]} '
                 f'({conf*100:.1f}%)  [{verdict}]')

        _render_panel(axes[row, 0], hm, range_axis, ang_axis,
                      title=title, mark_peak=True)
        _prob_bar(axes[row, 1], probs[idx], true, pred)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    if save:
        plt.savefig(save, dpi=150, bbox_inches='tight')
        print(f"Saved: {save}")
    plt.show()


def view_montage(X, Y, preds, probs, class_idx,
                 range_axis, ang_axis, n_ant,
                 n_examples=6, save=None):
    """
    Side-by-side montage: N correct vs N wrong predictions for one class.
    Useful for writeups because it isolates what changed between the
    heatmaps the model gets right and the ones it gets wrong.
    """
    mask_cls   = Y == class_idx
    mask_right = mask_cls & (preds == Y)
    mask_wrong = mask_cls & (preds != Y)

    def _pick(mask, k, rank_by_conf=True):
        idxs = np.where(mask)[0]
        if len(idxs) == 0:
            return idxs
        if rank_by_conf:
            conf = probs[np.arange(len(Y)), preds]
            idxs = idxs[np.argsort(-conf[idxs])]
        return idxs[:k]

    right_idx = _pick(mask_right, n_examples)
    wrong_idx = _pick(mask_wrong, n_examples)

    cols = max(len(right_idx), len(wrong_idx), 1)
    fig, axes = plt.subplots(2, cols, figsize=(2.4 * cols, 5.4),
                             squeeze=False)
    fig.suptitle(
        f'{CLASS_NAMES[class_idx]} — Correct (top) vs Wrong (bottom)  |  '
        f'N_ant={n_ant}',
        fontsize=12, fontweight='bold'
    )

    # Shared dB range makes intensity drift obvious
    pool = []
    for idxs in (right_idx, wrong_idx):
        for i in idxs:
            img = X[i]
            if img.ndim == 3 and img.shape[0] == 1:
                img = img[0]
            pool.append(20 * np.log10(img + 1e-6))
    vmax = np.percentile(np.concatenate([p.ravel() for p in pool]), 99) if pool else 0
    vmin = vmax - 40

    def _draw_row(row, idxs, label):
        for c in range(cols):
            ax = axes[row, c]
            if c < len(idxs):
                i = idxs[c]
                hm = X[i]
                if hm.ndim == 3 and hm.shape[0] == 1:
                    hm = hm[0]
                pred = int(preds[i]); true = int(Y[i])
                conf = float(probs[i, pred])
                _render_panel(ax, hm, range_axis, ang_axis,
                              vmin=vmin, vmax=vmax,
                              title=f'pred={CLASS_NAMES[pred]} '
                                    f'({conf*100:.0f}%)',
                              mark_peak=True)
            else:
                ax.axis('off')
            if c == 0:
                ax.set_ylabel(label, fontsize=11, fontweight='bold')

    _draw_row(0, right_idx, 'CORRECT')
    _draw_row(1, wrong_idx, 'WRONG')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    if save:
        plt.savefig(save, dpi=150, bbox_inches='tight')
        print(f"Saved: {save}")
    plt.show()


def view_single(X, Y, preds, probs, idx,
                range_axis, ang_axis, n_ant, save=None):
    """Render one sample by index (for drilling into a specific case)."""
    if idx < 0 or idx >= len(X):
        raise IndexError(f'Sample index {idx} out of range 0..{len(X)-1}')

    hm = X[idx]
    if hm.ndim == 3 and hm.shape[0] == 1:
        hm = hm[0]
    pred = int(preds[idx]); true = int(Y[idx])
    conf = float(probs[idx, pred])
    verdict = 'CORRECT' if pred == true else 'WRONG'

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.5),
                                  gridspec_kw={'width_ratios': [3, 1]})
    fig.suptitle(
        f'N_ant={n_ant}  idx={idx}  true={CLASS_NAMES[true]}  '
        f'pred={CLASS_NAMES[pred]} ({conf*100:.1f}%)  [{verdict}]',
        fontsize=12, fontweight='bold'
    )
    _render_panel(a1, hm, range_axis, ang_axis, mark_peak=True)
    _prob_bar(a2, probs[idx], true, pred)
    plt.tight_layout()
    if save:
        plt.savefig(save, dpi=150, bbox_inches='tight')
        print(f"Saved: {save}")
    plt.show()


def compare_n(mat_dir, model_dir, class_idx, sample_rank,
              only_errors, save=None):
    """
    Show the SAME test index for every available N_ant so you can
    watch how the input heatmap changes as angular resolution
    increases.

    Because the train/val/test split in the MATLAB generator is
    seeded the same way for every N_ant, the test set is aligned
    index-by-index, so a single `sample_rank` picks comparable rows.
    """
    pairs = []
    for mat_path in sorted(glob.glob(os.path.join(mat_dir, 'radar_shapes_N*.mat')),
                           key=lambda p: int(re.search(r'N(\d+)', p).group(1))):
        m = re.search(r'N(\d+)', os.path.basename(mat_path))
        if not m:
            continue
        n = int(m.group(1))
        mdl_path = os.path.join(model_dir, f'radar_shape_model_N{n}.pt')
        if not os.path.exists(mdl_path):
            print(f'  (skipping N={n}, no model file at {mdl_path})')
            continue
        pairs.append((n, mat_path, mdl_path))

    if not pairs:
        raise SystemExit('No matching .mat/.pt pairs found for compare_n mode.')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    cols = len(pairs)
    fig, axes = plt.subplots(2, cols, figsize=(3.2 * cols, 6.5),
                             gridspec_kw={'height_ratios': [3, 1]},
                             squeeze=False)
    fig.suptitle(
        f'{CLASS_NAMES[class_idx]} — same test position across N_ant  |  '
        f'rank={sample_rank}  |  only_errors={only_errors}',
        fontsize=12, fontweight='bold'
    )

    for c, (n_ant, mat_path, mdl_path) in enumerate(pairs):
        X, Y, range_axis, ang_axis, _ = _load_mat(mat_path)
        model, _ = _load_model(mdl_path, device)
        preds, probs = _predict_batch(model, X, device)

        mask = Y == class_idx
        if only_errors:
            mask = mask & (preds != Y)
        idxs = np.where(mask)[0]
        if len(idxs) == 0:
            axes[0, c].axis('off'); axes[1, c].axis('off')
            axes[0, c].set_title(f'N={n_ant}\n(none)', fontsize=10)
            continue
        if only_errors:
            conf = probs[np.arange(len(Y)), preds]
            idxs = idxs[np.argsort(-conf[idxs])]

        pick = idxs[min(sample_rank, len(idxs) - 1)]
        hm = X[pick]
        if hm.ndim == 3 and hm.shape[0] == 1:
            hm = hm[0]
        pred = int(preds[pick]); true = int(Y[pick])
        pconf = float(probs[pick, pred])

        _render_panel(axes[0, c], hm, range_axis, ang_axis,
                      title=f'N={n_ant}   idx={pick}\n'
                            f'pred={CLASS_NAMES[pred]} ({pconf*100:.0f}%)',
                      mark_peak=True)
        _prob_bar(axes[1, c], probs[pick], true, pred)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    if save:
        plt.savefig(save, dpi=150, bbox_inches='tight')
        print(f"Saved: {save}")
    plt.show()


# ─────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────
def build_parser():
    p = argparse.ArgumentParser(
        description='Inspect heatmaps and model predictions for '
                    'the radar shape CNN.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument('--mat', type=str,
                   help='Path to a single radar_shapes_N*.mat file. '
                        'Required unless --compare-n is used.')
    p.add_argument('--model', type=str,
                   help='Path to the matching radar_shape_model_N*.pt. '
                        'Required unless --compare-n is used.')

    p.add_argument('--shape', type=str, default='Rectangle',
                   help='Class to inspect: Circle/Square/Rectangle/'
                        'Triangle/Oval or an index 0-4.')
    p.add_argument('--sample', type=int, default=None,
                   help='Show a single test index instead of the grid.')
    p.add_argument('--only-errors', action='store_true',
                   help='Filter to test samples of the chosen shape that '
                        'the model got wrong.')
    p.add_argument('--n-examples', type=int, default=6,
                   help='Grid size for the default and --montage views.')
    p.add_argument('--montage', action='store_true',
                   help='Show a "correct on top / wrong on bottom" grid '
                        'side-by-side for the chosen class.')
    p.add_argument('--save', type=str, default=None,
                   help='Save the rendered figure to this path '
                        '(also still shows it).')

    p.add_argument('--compare-n', action='store_true',
                   help='Compare the same shape across every N_ant by '
                        'loading every radar_shapes_N*.mat + .pt pair.')
    p.add_argument('--mat-dir', type=str, default='.',
                   help='Directory to search for .mat files in --compare-n mode.')
    p.add_argument('--model-dir', type=str, default='.',
                   help='Directory to search for .pt files in --compare-n mode.')
    p.add_argument('--rank', type=int, default=0,
                   help='In --compare-n mode, which ranked sample to show '
                        '(0 = highest-confidence wrong when --only-errors).')

    return p


def main(argv=None):
    args = build_parser().parse_args(argv)

    if args.compare_n:
        class_idx = _resolve_shape(args.shape)
        compare_n(args.mat_dir, args.model_dir,
                  class_idx=class_idx, sample_rank=args.rank,
                  only_errors=args.only_errors, save=args.save)
        return

    if not args.mat or not args.model:
        print('ERROR: --mat and --model are required '
              '(unless using --compare-n).', file=sys.stderr)
        sys.exit(2)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    print(f'Loading data : {args.mat}')
    X, Y, range_axis, ang_axis, n_ant = _load_mat(args.mat)
    print(f'  Test set : {X.shape}')
    print(f'  N_ant    : {n_ant}')

    print(f'Loading model: {args.model}')
    model, ckpt = _load_model(args.model, device)
    print(f'  Ckpt N_ant      : {ckpt.get("n_ant", "unknown")}')
    print(f'  Ckpt test acc   : {ckpt.get("test_accuracy", 0)*100:.1f}%')

    if ckpt.get('n_ant', n_ant) != n_ant:
        print(f'  WARNING: model was trained on N={ckpt.get("n_ant")} '
              f'but .mat is N={n_ant}. Predictions may be meaningless.')

    print('Running inference on test set...')
    preds, probs = _predict_batch(model, X, device)
    acc = (preds == Y).mean()
    print(f'  Test accuracy (recomputed): {acc*100:.2f}%')

    # If the recomputed accuracy is near chance, something upstream is
    # wrong (shape mismatch, wrong .mat paired with wrong .pt, labels
    # re-indexed differently). Flag it loudly so the user doesn't stare
    # at figures that are effectively random.
    ckpt_acc = float(ckpt.get('test_accuracy', 0))
    chance   = 1.0 / N_CLASSES
    if acc < 1.5 * chance:
        print('  WARNING: recomputed accuracy is near chance. Likely causes:')
        print('           - this .mat file is not the one the model was trained on')
        print('           - the shape is transposed and not being auto-corrected')
        print('           - class-label encoding differs from training (check Y unique values)')
    elif ckpt_acc > 0 and acc < 0.7 * ckpt_acc:
        print(f'  WARNING: recomputed accuracy ({acc*100:.1f}%) is well below the '
              f'checkpoint accuracy ({ckpt_acc*100:.1f}%). Worth double-checking '
              'that the .mat and .pt were produced from the same run.')

    class_idx = _resolve_shape(args.shape)
    print(f'Focus class  : {CLASS_NAMES[class_idx]}')

    if args.sample is not None:
        view_single(X, Y, preds, probs, args.sample,
                    range_axis, ang_axis, n_ant, save=args.save)
    elif args.montage:
        view_montage(X, Y, preds, probs, class_idx,
                     range_axis, ang_axis, n_ant,
                     n_examples=args.n_examples, save=args.save)
    else:
        view_examples(X, Y, preds, probs, class_idx,
                      range_axis, ang_axis, n_ant,
                      only_errors=args.only_errors,
                      max_examples=args.n_examples,
                      save=args.save)


if __name__ == '__main__':
    main()
