"""
check_heatmap_normalization.py

Diagnostic: compare the radar heatmaps across N_ant values to rule out
per-file normalization drift as a cause of the inverted accuracy trend.

What this script does
---------------------
For every radar_shapes_N{N}.mat file it finds in the current directory:

  1. Loads X_test, Y_test, (and X_train for stats).
  2. Records dynamic-range statistics (min, max, mean, std, 50th/99th
     percentile, fraction of zeros) across the whole test set.
  3. Pulls one representative sample per class from X_test.
  4. Plots a grid: rows = N_ant, cols = 5 shape classes.
     Each cell shows the heatmap in 20*log10 space with a SHARED color
     range across the whole grid so intensity mismatches are obvious.
  5. Saves heatmap_normalisation_check.png and prints the stats table.

If you see any of these in the output, the inverted-trend reading of
the sweep is contaminated and you need to fix the data generator first:

  * Peak dB column drifts more than ~3 dB between N values.
  * "Frac zeros" column differs by more than a few percent.
  * The figure's colorbars show that some rows are mostly dark while
    others are saturated — that means per-file normalisation.

Usage:
    cd to the directory containing radar_shapes_N*.mat
    python check_heatmap_normalization.py
"""

import os
import glob
import numpy as np
import scipy.io
import h5py
import matplotlib.pyplot as plt


CLASS_NAMES = ['Circle', 'Square', 'Rectangle', 'Triangle', 'Oval']
N_CLASSES   = len(CLASS_NAMES)


def load_file(path):
    """Return (X_test, Y_test, X_train_stats, n_ant) for one .mat file."""
    try:
        d = scipy.io.loadmat(path)
        X_te = d['X_test'].astype(np.float32)
        Y_te = d['Y_test'].flatten().astype(np.int64) - 1
        X_tr = d['X_train'].astype(np.float32)
        n_ant = int(np.array(d['N_ant']).flatten()[0]) if 'N_ant' in d else None
    except NotImplementedError:
        with h5py.File(path, 'r') as f:
            X_te = np.array(f['X_test'], dtype=np.float32)
            Y_te = np.array(f['Y_test'], dtype=np.int64)
            X_tr = np.array(f['X_train'], dtype=np.float32)
            if X_te.ndim == 3:
                X_te = X_te.transpose(2, 0, 1)
            if X_tr.ndim == 3:
                X_tr = X_tr.transpose(2, 0, 1)
            Y_te = Y_te.flatten() - 1
            n_ant_raw = f.get('N_ant', None)
            n_ant = int(np.array(n_ant_raw).flatten()[0]) if n_ant_raw is not None else None

    if n_ant is None:
        base = os.path.basename(path)
        digits = ''.join(c for c in base.split('N')[-1].split('.')[0] if c.isdigit())
        n_ant = int(digits) if digits else -1

    return X_te, Y_te, X_tr, n_ant


def summarise(X, label):
    """Return a dict of dynamic-range stats for one tensor."""
    flat = X.reshape(-1)
    peak = flat.max()
    return {
        'label'    : label,
        'min'      : float(flat.min()),
        'max'      : float(peak),
        'mean'     : float(flat.mean()),
        'std'      : float(flat.std()),
        'p50'      : float(np.percentile(flat, 50)),
        'p99'      : float(np.percentile(flat, 99)),
        'frac_zero': float((flat == 0).mean()),
        'peak_dB'  : float(20 * np.log10(peak + 1e-12)),
    }


def main():
    paths = sorted(glob.glob('radar_shapes_N*.mat'),
                   key=lambda p: int(''.join(c for c in os.path.basename(p).split('N')[-1].split('.')[0] if c.isdigit())))
    if not paths:
        raise SystemExit("No radar_shapes_N*.mat files found in cwd.")

    files = []
    for p in paths:
        print(f"Loading {p}...")
        X_te, Y_te, X_tr, n_ant = load_file(p)
        files.append({
            'path': p, 'n_ant': n_ant,
            'X_te': X_te, 'Y_te': Y_te,
            'stats_test' : summarise(X_te, f'N={n_ant} test'),
            'stats_train': summarise(X_tr, f'N={n_ant} train'),
        })

    # ---- Stats table ----
    print()
    print("=" * 92)
    print(f"  {'set':>12} | {'min':>8} | {'max':>8} | {'mean':>8} | "
          f"{'std':>8} | {'p99':>8} | {'peak dB':>8} | {'%zero':>6}")
    print("-" * 92)
    for f in files:
        for s in (f['stats_train'], f['stats_test']):
            print(f"  {s['label']:>12} | "
                  f"{s['min']:>8.4f} | {s['max']:>8.4f} | "
                  f"{s['mean']:>8.4f} | {s['std']:>8.4f} | "
                  f"{s['p99']:>8.4f} | {s['peak_dB']:>8.2f} | "
                  f"{s['frac_zero']*100:>5.1f}%")
    print("=" * 92)

    peak_dBs = [f['stats_test']['peak_dB'] for f in files]
    print(f"\nPeak dB spread across N: {max(peak_dBs) - min(peak_dBs):.2f} dB")
    print("  (If > ~3 dB, the MATLAB generator is normalising per-file "
          "and the CNN is seeing different intensity scales at each N.)")

    # ---- Grid figure: rows = N_ant, cols = classes, same dB scale ----
    # Pick one sample per class per file — first hit in X_test
    picks = []
    for f in files:
        per_class = []
        for cls in range(N_CLASSES):
            idx = np.where(f['Y_te'] == cls)[0]
            if len(idx) == 0:
                per_class.append(None)
            else:
                img = f['X_te'][idx[0]]
                if img.ndim == 3 and img.shape[0] == 1:
                    img = img[0]
                per_class.append(img)
        picks.append(per_class)

    # Shared dB color range: take the 1st/99th percentile in dB across
    # everything we're about to draw so intensity drift is visible.
    vals = []
    for row in picks:
        for img in row:
            if img is None:
                continue
            vals.append(20 * np.log10(img + 1e-6))
    vals = np.concatenate([v.ravel() for v in vals])
    vmax = np.percentile(vals, 99)
    vmin = vmax - 40  # 40 dB dynamic range

    n_rows = len(files)
    fig, axes = plt.subplots(n_rows, N_CLASSES,
                             figsize=(2.4 * N_CLASSES, 2.2 * n_rows),
                             squeeze=False)
    fig.suptitle(
        f'Per-N heatmap check  |  same sample per class  |  '
        f'shared dB range [{vmin:.0f}, {vmax:.0f}] dB',
        fontsize=11, fontweight='bold'
    )

    for r, (f, row) in enumerate(zip(files, picks)):
        for c, img in enumerate(row):
            ax = axes[r, c]
            if img is None:
                ax.axis('off')
                continue
            im = ax.imshow(20 * np.log10(img + 1e-6),
                           cmap='jet', aspect='auto',
                           vmin=vmin, vmax=vmax)
            if r == 0:
                ax.set_title(CLASS_NAMES[c], fontsize=10)
            if c == 0:
                ax.set_ylabel(f"N={f['n_ant']}", fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])

    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    fig.colorbar(im, cax=cbar_ax, label='dB')
    plt.subplots_adjust(left=0.05, right=0.90, top=0.92, bottom=0.05,
                        wspace=0.05, hspace=0.1)
    out = 'heatmap_normalisation_check.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {out}")
    plt.show()


if __name__ == '__main__':
    main()
