"""
=============================================================
 predict_real.py
 Run Trained Multi-Task CNN on Real Radar Heatmap

 Usage:
   python predict_real.py
   python predict_real.py --file my_heatmap.mat
   python predict_real.py --file my_heatmap.mat --model radar_multitask_model.pt

 Input:  real_heatmap.mat  (from process_real_data.m)
 Output: Shape + property prediction + likely real objects

 Project: Radar Object Detection + ML (Junior IW)
=============================================================
"""

import scipy.io
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import argparse
import os
import sys

# ── Device ──────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============================================================
#  SECTION 1: PROPERTY DEFINITIONS
#  Must match radar_multitask_cnn.py exactly
# ============================================================

SHAPE_CLASSES    = ['Circle',       'Square',       'Rectangle',   'Triangle']
SIZE_CLASSES     = ['Small',        'Medium',        'Large']
ASPECT_CLASSES   = ['Square_ratio', 'Wide',          'Tall']
SYMMETRY_CLASSES = ['Symmetric',    'Asymmetric']
MATERIAL_CLASSES = ['Soft',         'Hard_plastic',  'Metal']

N_SHAPES     = len(SHAPE_CLASSES)
N_SIZES      = len(SIZE_CLASSES)
N_ASPECTS    = len(ASPECT_CLASSES)
N_SYMMETRIES = len(SYMMETRY_CLASSES)
N_MATERIALS  = len(MATERIAL_CLASSES)

OBJECT_LOOKUP = {
    (0, 0, 0, 0, 0): ['Soccer Ball', 'Tennis Ball', 'Apple'],
    (0, 0, 0, 0, 1): ['Plastic Ball', 'Helmet'],
    (0, 1, 0, 0, 0): ['Basketball', 'Volleyball', 'Head'],
    (0, 1, 0, 0, 2): ['Metal Drum (small)', 'Wheel Hub'],
    (0, 2, 0, 0, 2): ['Oil Drum', 'Large Wheel', 'Tire'],
    (0, 2, 0, 0, 0): ['Bean Bag', 'Large Cushion'],
    (1, 0, 0, 0, 1): ['Small Box', 'Book', 'Tablet'],
    (1, 0, 0, 0, 2): ['Metal Box', 'Electronic Device'],
    (1, 1, 0, 0, 1): ['Plastic Crate', 'Storage Box'],
    (1, 1, 0, 0, 2): ['Metal Crate', 'Safe', 'Server'],
    (1, 2, 0, 0, 2): ['Large Metal Cabinet', 'Refrigerator'],
    (1, 2, 0, 0, 1): ['Large Plastic Container', 'Bin'],
    (2, 0, 1, 1, 1): ['Laptop', 'Small Tray', 'Clipboard'],
    (2, 0, 1, 1, 2): ['Metal Tray', 'Small Sheet'],
    (2, 1, 1, 1, 2): ['Chair (metal)', 'Shelf', 'Table (small)'],
    (2, 1, 1, 1, 1): ['Plastic Chair', 'Plastic Table'],
    (2, 2, 1, 1, 2): ['Table', 'Workbench', 'Car (side)'],
    (2, 2, 1, 1, 1): ['Plastic Table', 'Sofa'],
    (2, 1, 2, 1, 0): ['Person (standing)', 'Mannequin'],
    (2, 1, 2, 1, 1): ['Door', 'Whiteboard', 'Panel'],
    (2, 2, 2, 1, 2): ['Metal Door', 'Large Cabinet', 'Locker'],
    (2, 2, 2, 1, 1): ['Bookshelf', 'Wardrobe'],
    (3, 0, 0, 1, 1): ['Traffic Cone (small)', 'Wedge'],
    (3, 1, 0, 1, 1): ['Traffic Cone', 'Tent Peg'],
    (3, 1, 0, 1, 2): ['Metal Cone', 'Funnel'],
    (3, 2, 0, 1, 0): ['Tent', 'Roof Structure'],
    (3, 2, 0, 1, 2): ['Metal Pyramid', 'Industrial Hopper'],
}

SHAPE_FALLBACK = {
    'Circle'    : ['Ball-like object',       'Cylindrical object'],
    'Square'    : ['Box-like object',         'Cubic object'],
    'Rectangle' : ['Rectangular object',      'Flat panel'],
    'Triangle'  : ['Conical object',          'Wedge-shaped object'],
}

# ============================================================
#  SECTION 2: MODEL ARCHITECTURE
#  Must exactly match the architecture in radar_multitask_cnn.py
# ============================================================

class MultiTaskRadarCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 32,  7, padding=3), nn.BatchNorm2d(32),  nn.ReLU(True), nn.MaxPool2d(2,2),
            nn.Conv2d(32, 64, 5, padding=2), nn.BatchNorm2d(64),  nn.ReLU(True), nn.MaxPool2d(2,2),
            nn.Conv2d(64,128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(True), nn.MaxPool2d(2,2),
            nn.Conv2d(128,256,3, padding=1), nn.BatchNorm2d(256), nn.ReLU(True), nn.MaxPool2d(2,2),
        )
        self.gap       = nn.AdaptiveAvgPool2d(1)
        self.shared_fc = nn.Sequential(nn.Flatten(), nn.Linear(256,128), nn.ReLU(True), nn.Dropout(0.4))
        self.head_shape    = nn.Sequential(nn.Linear(128,64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, N_SHAPES))
        self.head_size     = nn.Sequential(nn.Linear(128,32), nn.ReLU(), nn.Linear(32, N_SIZES))
        self.head_aspect   = nn.Sequential(nn.Linear(128,32), nn.ReLU(), nn.Linear(32, N_ASPECTS))
        self.head_symmetry = nn.Sequential(nn.Linear(128,16), nn.ReLU(), nn.Linear(16, N_SYMMETRIES))
        self.head_material = nn.Sequential(nn.Linear(128,32), nn.ReLU(), nn.Linear(32, N_MATERIALS))

    def forward(self, x):
        emb = self.shared_fc(self.gap(self.backbone(x)))
        return {
            'shape'    : self.head_shape(emb),
            'size'     : self.head_size(emb),
            'aspect'   : self.head_aspect(emb),
            'symmetry' : self.head_symmetry(emb),
            'material' : self.head_material(emb),
        }


# ============================================================
#  SECTION 3: LOAD MODEL
# ============================================================

def load_model(model_path='radar_multitask_model.pt'):
    if not os.path.exists(model_path):
        print(f"\nERROR: Model file not found: {model_path}")
        print("Train the model first by running: python radar_multitask_cnn.py")
        sys.exit(1)

    model = MultiTaskRadarCNN()
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    print(f"Model loaded from: {model_path}")
    return model


# ============================================================
#  SECTION 4: LOAD HEATMAP
# ============================================================

def load_heatmap(mat_path='real_heatmap.mat'):
    if not os.path.exists(mat_path):
        print(f"\nERROR: Heatmap file not found: {mat_path}")
        print("Generate it first by running process_real_data.m in MATLAB")
        sys.exit(1)

    data = scipy.io.loadmat(mat_path)

    # Try both normalised and raw
    if 'RA_normalised' in data:
        heatmap = data['RA_normalised'].astype(np.float32)
    elif 'RA_map' in data:
        heatmap = data['RA_map'].astype(np.float32)
        heatmap = heatmap / (heatmap.max() + 1e-9)
    else:
        raise ValueError("Cannot find heatmap in .mat file")

    # Load axes if available
    range_axis = data.get('range_axis', None)
    ang_axis   = data.get('ang_axis',   None)

    if range_axis is not None:
        range_axis = range_axis.flatten()
    if ang_axis is not None:
        ang_axis = ang_axis.flatten()

    print(f"Heatmap loaded: {heatmap.shape}")
    print(f"Value range   : [{heatmap.min():.4f}, {heatmap.max():.4f}]")

    return heatmap, range_axis, ang_axis


# ============================================================
#  SECTION 5: PREPROCESSING
#  Match the exact preprocessing used in training
# ============================================================

def preprocess(heatmap, target_shape=(512, 256)):
    """
    Resize heatmap to match training dimensions if needed,
    then convert to PyTorch tensor.
    """
    h, w = heatmap.shape

    # Resize if dimensions don't match training data
    if (h, w) != target_shape:
        print(f"Resizing heatmap from {heatmap.shape} to {target_shape}")
        from scipy.ndimage import zoom
        zoom_h = target_shape[0] / h
        zoom_w = target_shape[1] / w
        heatmap = zoom(heatmap, (zoom_h, zoom_w), order=1)

    # Renormalise after resize
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()

    # [H × W] → [1 × 1 × H × W]  (batch=1, channel=1)
    tensor = torch.tensor(heatmap, dtype=torch.float32)
    tensor = tensor.unsqueeze(0).unsqueeze(0).to(device)

    return tensor


# ============================================================
#  SECTION 6: OBJECT LOOKUP
# ============================================================

def lookup_object(p_shape, p_size, p_aspect, p_symmetry, p_material):
    key = (p_shape, p_size, p_aspect, p_symmetry, p_material)

    if key in OBJECT_LOOKUP:
        return OBJECT_LOOKUP[key], 'Exact match'

    for mat in range(N_MATERIALS):
        k = (p_shape, p_size, p_aspect, p_symmetry, mat)
        if k in OBJECT_LOOKUP:
            return OBJECT_LOOKUP[k], 'Relaxed: material'

    for asp in range(N_ASPECTS):
        k = (p_shape, p_size, asp, p_symmetry, p_material)
        if k in OBJECT_LOOKUP:
            return OBJECT_LOOKUP[k], 'Relaxed: aspect'

    for sz in range(N_SIZES):
        k = (p_shape, sz, p_aspect, p_symmetry, p_material)
        if k in OBJECT_LOOKUP:
            return OBJECT_LOOKUP[k], 'Relaxed: size'

    shape_name = SHAPE_CLASSES[p_shape]
    return SHAPE_FALLBACK[shape_name], 'Fallback: shape only'


# ============================================================
#  SECTION 7: INFERENCE
# ============================================================

def predict(heatmap_tensor, model):
    """Run forward pass and extract predictions."""
    model.eval()
    with torch.no_grad():
        outputs = model(heatmap_tensor)

    results = {}
    for prop, logits in outputs.items():
        probs    = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred_idx = int(np.argmax(probs))
        results[prop] = {
            'pred_idx'  : pred_idx,
            'probs'     : probs,
            'confidence': float(probs[pred_idx]),
        }
    return results


# ============================================================
#  SECTION 8: VISUALISATION
# ============================================================

def visualise_result(heatmap, results, objects, match_type,
                     range_axis=None, ang_axis=None):

    prop_config = [
        ('shape',    SHAPE_CLASSES,    'Shape',    '#3498DB'),
        ('size',     SIZE_CLASSES,     'Size',     '#E74C3C'),
        ('aspect',   ASPECT_CLASSES,   'Aspect',   '#2ECC71'),
        ('symmetry', SYMMETRY_CLASSES, 'Symmetry', '#F39C12'),
        ('material', MATERIAL_CLASSES, 'Material', '#9B59B6'),
    ]

    fig = plt.figure(figsize=(16, 7), facecolor='white')
    gs  = gridspec.GridSpec(2, 6, figure=fig, hspace=0.45, wspace=0.5)

    # ── Heatmap (full) ───────────────────────────────────────
    ax_full = fig.add_subplot(gs[:, 0])
    x_ax = ang_axis   if ang_axis   is not None else np.arange(heatmap.shape[1])
    y_ax = range_axis if range_axis is not None else np.arange(heatmap.shape[0])
    im = ax_full.imshow(20*np.log10(heatmap + 1e-6),
                        cmap='jet', aspect='auto',
                        extent=[x_ax[0], x_ax[-1], y_ax[0], y_ax[-1]],
                        vmin=-40, vmax=0, origin='lower')
    plt.colorbar(im, ax=ax_full, label='dB', fraction=0.046)
    ax_full.set_xlabel('Angle (°)' if ang_axis is not None else 'Angle bins')
    ax_full.set_ylabel('Range (m)' if range_axis is not None else 'Range bins')
    ax_full.set_title('Input Heatmap', fontsize=10, fontweight='bold')

    # ── Heatmap (zoomed) ─────────────────────────────────────
    ax_zoom = fig.add_subplot(gs[:, 1])
    if range_axis is not None:
        peak_r = range_axis[np.argmax(heatmap.mean(axis=1))]
        ax_zoom.set_ylim([max(y_ax[0], peak_r-1.5),
                          min(y_ax[-1], peak_r+1.5)])
    ax_zoom.imshow(20*np.log10(heatmap + 1e-6),
                   cmap='jet', aspect='auto',
                   extent=[x_ax[0], x_ax[-1], y_ax[0], y_ax[-1]],
                   vmin=-40, vmax=0, origin='lower')
    ax_zoom.set_xlim([-60, 60])
    ax_zoom.set_xlabel('Angle (°)' if ang_axis is not None else 'bins')
    ax_zoom.set_title('Zoomed', fontsize=10, fontweight='bold')

    # ── Property bars ─────────────────────────────────────────
    for col, (key, class_names, title, color) in enumerate(prop_config):
        row_top = fig.add_subplot(gs[0, col+1])
        row_bot = fig.add_subplot(gs[1, col+1])

        for ax, (prop_key, prop_classes, prop_title, prop_color) in zip(
            [row_top, row_bot],
            [prop_config[col], prop_config[min(col+1, len(prop_config)-1)]]
        ):
            break

        ax = fig.add_subplot(gs[:, col+1]) if col == 0 else None

    # Simpler layout — one row of property bars
    fig2, axes = plt.subplots(1, 5, figsize=(16, 3), facecolor='white')
    fig2.suptitle(
        f"Prediction: {', '.join(objects[:2])}  |  {match_type}",
        fontsize=13, fontweight='bold'
    )

    for ax, (key, class_names, title, color) in zip(axes, prop_config):
        probs = results[key]['probs']
        pred  = results[key]['pred_idx']
        conf  = results[key]['confidence']

        bars = ax.barh(class_names, probs*100,
                       color=color, alpha=0.7,
                       edgecolor='white', height=0.5)
        bars[pred].set_alpha(1.0)
        bars[pred].set_edgecolor('black')
        bars[pred].set_linewidth(2.5)

        for i, b in enumerate(bars):
            ax.text(b.get_width()+1, b.get_y()+b.get_height()/2,
                    f"{probs[i]*100:.0f}%", va='center', fontsize=8)

        ax.set_xlim([0, 120])
        ax.set_title(f"{title}\n→ {class_names[pred]} ({conf*100:.0f}%)",
                     fontsize=9, fontweight='bold')
        ax.set_xlabel('Confidence %', fontsize=8)
        ax.axvline(50, color='gray', linestyle='--', alpha=0.4)
        ax.grid(axis='x', alpha=0.2)

    plt.tight_layout()
    plt.savefig('real_prediction.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: real_prediction.png")


# ============================================================
#  SECTION 9: MAIN
# ============================================================

def main(mat_file='real_heatmap.mat', model_file='radar_multitask_model.pt'):

    print("\n" + "="*55)
    print("  RADAR OBJECT CLASSIFICATION — REAL DATA INFERENCE")
    print("="*55)

    # Step 1: Load
    model   = load_model(model_file)
    heatmap, range_axis, ang_axis = load_heatmap(mat_file)

    # Step 2: Preprocess
    tensor = preprocess(heatmap, target_shape=(512, 256))

    # Step 3: Predict
    results = predict(tensor, model)

    # Step 4: Extract predictions
    p_shape    = results['shape']['pred_idx']
    p_size     = results['size']['pred_idx']
    p_aspect   = results['aspect']['pred_idx']
    p_symmetry = results['symmetry']['pred_idx']
    p_material = results['material']['pred_idx']

    objects, match_type = lookup_object(
        p_shape, p_size, p_aspect, p_symmetry, p_material
    )

    # Step 5: Print results
    print(f"\n{'─'*55}")
    print(f"  PROPERTY PROFILE:")
    print(f"{'─'*55}")

    prop_map = [
        ('Shape',    SHAPE_CLASSES,    'shape'),
        ('Size',     SIZE_CLASSES,     'size'),
        ('Aspect',   ASPECT_CLASSES,   'aspect'),
        ('Symmetry', SYMMETRY_CLASSES, 'symmetry'),
        ('Material', MATERIAL_CLASSES, 'material'),
    ]

    for label, classes, key in prop_map:
        idx  = results[key]['pred_idx']
        conf = results[key]['confidence']
        bar  = '█' * int(conf*25)
        print(f"  {label:12s}: {classes[idx]:16s} {conf*100:5.1f}%  {bar}")

    print(f"\n{'─'*55}")
    print(f"  PREDICTED OBJECTS : {', '.join(objects)}")
    print(f"  MATCH TYPE        : {match_type}")
    print(f"{'─'*55}\n")

    # Step 6: Visualise
    visualise_result(heatmap, results, objects, match_type,
                     range_axis, ang_axis)

    # Step 7: Sim-to-real gap check
    print("SIM-TO-REAL GAP CHECK:")
    print("Compare these confidence scores to training performance.")
    low_conf = [(lbl, classes[results[key]['pred_idx']],
                 results[key]['confidence'])
                for lbl, classes, key in prop_map
                if results[key]['confidence'] < 0.70]

    if low_conf:
        print("  Low confidence properties (< 70%) — possible domain gap:")
        for lbl, pred, conf in low_conf:
            print(f"    {lbl:12s}: {pred} ({conf*100:.1f}%)")
        print("  Consider fine-tuning with a few real examples.")
    else:
        print("  All properties predicted with >70% confidence.")
        print("  Sim-to-real transfer appears successful for this sample.")

    return results, objects


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file',  default='real_heatmap.mat',
                        help='Path to heatmap .mat file from MATLAB')
    parser.add_argument('--model', default='radar_multitask_model.pt',
                        help='Path to trained model .pt file')
    args = parser.parse_args()

    main(mat_file=args.file, model_file=args.model)
