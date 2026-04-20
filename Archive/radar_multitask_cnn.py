"""
=============================================================
 Radar Multi-Task CNN — Hierarchical Property Learning
 
 Architecture:
   Shared CNN Backbone
       ↓
   ┌───────────────────────────────────┐
   │  Head 1: Shape      (4 classes)  │
   │  Head 2: Size       (3 classes)  │
   │  Head 3: Aspect     (3 classes)  │
   │  Head 4: Symmetry   (2 classes)  │
   │  Head 5: Material   (3 classes)  │
   └───────────────────────────────────┘
       ↓
   Property Vector → Object Lookup Table
       ↓
   Final Object Prediction

 Key idea:
   Each head learns ONE specific physical property.
   The combination of all properties uniquely identifies
   the real object — far more precisely than shape alone.

 Project: Radar Object Detection + ML (Junior IW)
=============================================================
"""

import scipy.io
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import itertools
import time

torch.manual_seed(42)
np.random.seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# ============================================================
#  SECTION 1: PROPERTY DEFINITIONS
#
#  Every sample in the dataset gets labels for ALL properties.
#  These are derived programmatically from the shape type
#  and simulation parameters — no manual labelling needed.
# ============================================================

# ── Property 1: Shape ───────────────────────────────────────
SHAPE_CLASSES = ['Circle', 'Square', 'Rectangle', 'Triangle']
N_SHAPES = len(SHAPE_CLASSES)

# ── Property 2: Size (longest dimension in metres) ──────────
SIZE_CLASSES   = ['Small', 'Medium', 'Large']
SIZE_THRESHOLDS = [0.35, 0.60]   # boundaries in metres
# Small  = longest dim < 0.35m
# Medium = 0.35m to 0.60m
# Large  = > 0.60m
N_SIZES = len(SIZE_CLASSES)

# ── Property 3: Aspect Ratio (width / height) ────────────────
ASPECT_CLASSES = ['Square_ratio', 'Wide', 'Tall']
# Square_ratio = aspect ratio 0.7 to 1.3
# Wide         = aspect ratio > 1.3  (width >> height)
# Tall         = aspect ratio < 0.7  (height >> width)
N_ASPECTS = len(ASPECT_CLASSES)

# ── Property 4: Symmetry ─────────────────────────────────────
SYMMETRY_CLASSES = ['Symmetric', 'Asymmetric']
# Symmetric   = circle, square (look same from multiple angles)
# Asymmetric  = rectangle, triangle (look different each side)
N_SYMMETRIES = len(SYMMETRY_CLASSES)

# ── Property 5: Corner Strength (dominant material/geometry) ─
MATERIAL_CLASSES = ['Soft', 'Hard_plastic', 'Metal']
# Soft          = no sharp corners (circle → ball, pillow)
# Hard_plastic  = moderate corners (plastic chair, box)
# Metal         = strong corner reflectors (metal furniture)
N_MATERIALS = len(MATERIAL_CLASSES)

# ── Master object lookup table ───────────────────────────────
# Key: (shape_idx, size_idx, aspect_idx, symmetry_idx, material_idx)
# Value: list of real-world objects this combination represents
#
# This is the explicit mapping from property vector to object.
# The more properties you add, the more specific the prediction.

OBJECT_LOOKUP = {
    # Circle combinations
    (0, 0, 0, 0, 0): ['Soccer Ball', 'Tennis Ball', 'Apple'],
    (0, 0, 0, 0, 1): ['Plastic Ball', 'Helmet'],
    (0, 1, 0, 0, 0): ['Basketball', 'Volleyball', 'Head'],
    (0, 1, 0, 0, 2): ['Metal Drum (small)', 'Wheel Hub'],
    (0, 2, 0, 0, 2): ['Oil Drum', 'Large Wheel', 'Tire'],
    (0, 2, 0, 0, 0): ['Bean Bag', 'Large Cushion'],

    # Square combinations
    (1, 0, 0, 0, 1): ['Small Box', 'Book', 'Tablet'],
    (1, 0, 0, 0, 2): ['Metal Box', 'Electronic Device'],
    (1, 1, 0, 0, 1): ['Plastic Crate', 'Storage Box'],
    (1, 1, 0, 0, 2): ['Metal Crate', 'Safe', 'Server'],
    (1, 2, 0, 0, 2): ['Large Metal Cabinet', 'Refrigerator'],
    (1, 2, 0, 0, 1): ['Large Plastic Container', 'Bin'],

    # Rectangle Wide combinations (width >> height — flat objects)
    (2, 0, 1, 1, 1): ['Laptop', 'Small Tray', 'Clipboard'],
    (2, 0, 1, 1, 2): ['Metal Tray', 'Small Sheet'],
    (2, 1, 1, 1, 2): ['Chair (metal)', 'Shelf', 'Table (small)'],
    (2, 1, 1, 1, 1): ['Plastic Chair', 'Plastic Table'],
    (2, 2, 1, 1, 2): ['Table', 'Workbench', 'Car (side)'],
    (2, 2, 1, 1, 1): ['Plastic Table', 'Sofa'],

    # Rectangle Tall combinations (height >> width — upright objects)
    (2, 1, 2, 1, 0): ['Person (standing)', 'Mannequin'],
    (2, 1, 2, 1, 1): ['Door', 'Whiteboard', 'Panel'],
    (2, 2, 2, 1, 2): ['Metal Door', 'Large Cabinet', 'Locker'],
    (2, 2, 2, 1, 1): ['Bookshelf', 'Wardrobe'],

    # Triangle combinations
    (3, 0, 0, 1, 1): ['Traffic Cone (small)', 'Wedge'],
    (3, 1, 0, 1, 1): ['Traffic Cone', 'Tent Peg'],
    (3, 1, 0, 1, 2): ['Metal Cone', 'Funnel'],
    (3, 2, 0, 1, 0): ['Tent', 'Roof Structure'],
    (3, 2, 0, 1, 2): ['Metal Pyramid', 'Industrial Hopper'],
}

# Fallback for combinations not in the lookup
DEFAULT_LOOKUP = "Unknown object — combination not in lookup table"


# ============================================================
#  SECTION 2: DATASET WITH MULTI-PROPERTY LABELS
#
#  The MATLAB simulation knows the exact shape, size, and
#  geometry of each generated sample. We derive all
#  property labels from the simulation parameters directly.
# ============================================================

def derive_property_labels(shape_labels, shape_params):
    """
    Derive all 5 property labels from simulation parameters.

    shape_labels  : [N] array of shape indices (0-3)
    shape_params  : dict with keys 'sizes', 'aspects', 'materials'
                    each [N] array derived from simulation metadata

    Returns:
        labels dict with keys: 'shape','size','aspect','symmetry','material'
        each a [N] int array
    """
    N = len(shape_labels)

    # Shape — directly from simulation
    label_shape = shape_labels.copy()

    # Size — from longest dimension of each shape
    size_m = shape_params['longest_dim']   # metres
    label_size = np.zeros(N, dtype=np.int64)
    label_size[size_m >= SIZE_THRESHOLDS[0]] = 1
    label_size[size_m >= SIZE_THRESHOLDS[1]] = 2

    # Aspect ratio
    aspect = shape_params['aspect_ratio']  # width / height
    label_aspect = np.ones(N, dtype=np.int64)   # default: square
    label_aspect[aspect > 1.3] = 1   # Wide
    label_aspect[aspect < 0.7] = 2   # Tall
    # square-like: stays 0

    # Symmetry — circle and square are symmetric, rect and triangle not
    label_sym = np.ones(N, dtype=np.int64)   # default: asymmetric
    label_sym[(label_shape == 0) | (label_shape == 1)] = 0   # symmetric

    # Material — from amplitude distribution in simulation
    amp = shape_params['max_amplitude']
    label_mat = np.zeros(N, dtype=np.int64)   # default: soft
    label_mat[amp >= 0.5] = 1   # hard plastic
    label_mat[amp >= 0.85] = 2  # metal

    return {
        'shape'    : label_shape,
        'size'     : label_size,
        'aspect'   : label_aspect,
        'symmetry' : label_sym,
        'material' : label_mat,
    }


class RadarMultiTaskDataset(Dataset):
    """
    Dataset that returns a heatmap and ALL property labels.
    Each __getitem__ returns (heatmap, label_dict).
    """
    def __init__(self, X, property_labels, augment=False):
        self.X              = torch.tensor(X, dtype=torch.float32)
        self.property_labels = {
            k: torch.tensor(v, dtype=torch.long)
            for k, v in property_labels.items()
        }
        self.augment = augment

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        heatmap = self.X[idx]

        # Data augmentation — horizontal flip (mirrors angle axis)
        # A flipped heatmap still shows the same shape
        if self.augment and torch.rand(1).item() > 0.5:
            heatmap = torch.flip(heatmap, dims=[-1])

        labels = {k: v[idx] for k, v in self.property_labels.items()}
        return heatmap, labels


# ============================================================
#  SECTION 3: MULTI-TASK CNN ARCHITECTURE
#
#  Shared backbone: 4 convolutional blocks
#  Five independent heads: one per property
#
#  Each head has its own FC layers and output size.
#  The backbone gradient receives signals from ALL heads
#  simultaneously — it learns features useful for ALL
#  properties at once, producing richer representations
#  than a single-task model.
# ============================================================

class MultiTaskRadarCNN(nn.Module):
    def __init__(self):
        super(MultiTaskRadarCNN, self).__init__()

        # ── Shared Backbone ──────────────────────────────────
        # Processes raw heatmap into a rich feature representation
        # that is useful for ALL property heads simultaneously
        self.backbone = nn.Sequential(

            # Block 1 — broad spatial structure
            # Large 7×7 kernel captures the overall shape footprint
            nn.Conv2d(1, 32, kernel_size=7, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),   # → 256×128

            # Block 2 — mid-level features
            # Learns peak arrangements, spacing between reflectors
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),   # → 128×64

            # Block 3 — fine structure
            # Learns corner vs edge vs curved surface patterns
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),   # → 64×32

            # Block 4 — high-level combinations
            # Combines lower-level patterns into abstract features
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),   # → 32×16
        )

        # Global Average Pooling: [B×256×32×16] → [B×256]
        self.gap = nn.AdaptiveAvgPool2d(1)

        # Shared embedding: compress 256 → 128
        # All heads branch off from this 128-dim representation
        self.shared_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
        )

        # ── Property Heads ────────────────────────────────────
        # Each head has its own FC → output
        # Head depth can differ — harder properties get more layers

        # Head 1: Shape (hardest — 4 classes, orientation-variant)
        self.head_shape = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, N_SHAPES)
        )

        # Head 2: Size (medium — 3 classes)
        self.head_size = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, N_SIZES)
        )

        # Head 3: Aspect Ratio (medium — 3 classes)
        self.head_aspect = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, N_ASPECTS)
        )

        # Head 4: Symmetry (easiest — 2 classes)
        self.head_symmetry = nn.Sequential(
            nn.Linear(128, 16),
            nn.ReLU(),
            nn.Linear(16, N_SYMMETRIES)
        )

        # Head 5: Material / Corner Strength (3 classes)
        self.head_material = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, N_MATERIALS)
        )

    def forward(self, x):
        # Shared backbone
        features = self.backbone(x)
        features = self.gap(features)
        embedding = self.shared_fc(features)

        # Each head independently processes the shared embedding
        out_shape    = self.head_shape(embedding)
        out_size     = self.head_size(embedding)
        out_aspect   = self.head_aspect(embedding)
        out_symmetry = self.head_symmetry(embedding)
        out_material = self.head_material(embedding)

        return {
            'shape'    : out_shape,
            'size'     : out_size,
            'aspect'   : out_aspect,
            'symmetry' : out_symmetry,
            'material' : out_material,
        }

    def get_embedding(self, x):
        """Return the 128-dim shared embedding for analysis."""
        with torch.no_grad():
            features  = self.backbone(x)
            features  = self.gap(features)
            embedding = self.shared_fc(features)
        return embedding


# ============================================================
#  SECTION 4: MULTI-TASK LOSS
#
#  Total loss = weighted sum of individual property losses.
#  Weights reflect task difficulty and importance:
#    Shape    → highest weight (primary classification goal)
#    Size     → medium weight  (strongly discriminative)
#    Aspect   → medium weight  (distinguishes rect from square)
#    Symmetry → lower weight   (easier, fewer classes)
#    Material → medium weight  (helps with object lookup)
# ============================================================

class MultiTaskLoss(nn.Module):
    def __init__(self, weights=None):
        super(MultiTaskLoss, self).__init__()

        # Loss weight per property head
        # Increase a weight if that property matters more for your objects
        if weights is None:
            weights = {
                'shape'    : 1.5,   # primary goal
                'size'     : 1.0,
                'aspect'   : 1.0,
                'symmetry' : 0.5,   # easiest task, lower weight
                'material' : 1.0,
            }
        self.weights  = weights
        self.ce_loss  = nn.CrossEntropyLoss()

    def forward(self, outputs, labels):
        """
        outputs : dict of logits per property
        labels  : dict of ground truth indices per property
        Returns : total loss + individual losses for logging
        """
        individual_losses = {}
        total_loss = 0.0

        for prop in outputs:
            loss = self.ce_loss(outputs[prop], labels[prop])
            individual_losses[prop] = loss.item()
            total_loss += self.weights[prop] * loss

        return total_loss, individual_losses


# ============================================================
#  SECTION 5: TRAINING LOOP
# ============================================================

def train_multitask(model, train_loader, val_loader,
                    n_epochs=40, lr=1e-3, patience=8):

    model      = model.to(device)
    criterion  = MultiTaskLoss()
    optimizer  = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler  = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', factor=0.5, patience=3, verbose=True
    )

    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc_shape': [], 'val_acc_shape': [],
        'prop_losses': {p: [] for p in
                        ['shape','size','aspect','symmetry','material']}
    }

    best_val_loss  = float('inf')
    best_weights   = None
    patience_count = 0

    print(f"\n{'='*70}")
    print(f"  Multi-Task Training | {n_epochs} epochs | Device: {device}")
    print(f"{'='*70}")
    print(f"{'Epoch':>5} | {'Loss':>7} | {'Val':>7} | "
          f"{'Shape%':>7} | {'Size%':>6} | {'Asp%':>5} | "
          f"{'Sym%':>5} | {'Mat%':>5}")
    print(f"{'-'*70}")

    for epoch in range(1, n_epochs+1):
        t0 = time.time()

        # ── Training ─────────────────────────────────────────
        model.train()
        train_loss = 0.0
        correct    = {p: 0 for p in ['shape','size','aspect','symmetry','material']}
        total      = 0

        for X_batch, Y_batch in train_loader:
            X_batch = X_batch.to(device)
            Y_batch = {k: v.to(device) for k, v in Y_batch.items()}

            optimizer.zero_grad()
            outputs          = model(X_batch)
            loss, ind_losses = criterion(outputs, Y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * X_batch.size(0)
            total      += X_batch.size(0)

            for prop in correct:
                preds          = outputs[prop].argmax(dim=1)
                correct[prop] += (preds == Y_batch[prop]).sum().item()

        train_loss /= total
        train_accs  = {p: correct[p]/total for p in correct}

        # ── Validation ───────────────────────────────────────
        model.eval()
        val_loss = 0.0
        val_correct = {p: 0 for p in correct}
        val_total   = 0

        with torch.no_grad():
            for X_batch, Y_batch in val_loader:
                X_batch = X_batch.to(device)
                Y_batch = {k: v.to(device) for k, v in Y_batch.items()}

                outputs          = model(X_batch)
                loss, ind_losses = criterion(outputs, Y_batch)
                val_loss        += loss.item() * X_batch.size(0)
                val_total       += X_batch.size(0)

                for prop in val_correct:
                    preds              = outputs[prop].argmax(dim=1)
                    val_correct[prop] += (preds == Y_batch[prop]).sum().item()

        val_loss /= val_total
        val_accs  = {p: val_correct[p]/val_total for p in val_correct}

        scheduler.step(val_loss)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc_shape'].append(train_accs['shape'])
        history['val_acc_shape'].append(val_accs['shape'])

        print(f"{epoch:>5} | {train_loss:>7.3f} | {val_loss:>7.3f} | "
              f"{train_accs['shape']*100:>6.1f}% | "
              f"{train_accs['size']*100:>5.1f}% | "
              f"{train_accs['aspect']*100:>4.1f}% | "
              f"{train_accs['symmetry']*100:>4.1f}% | "
              f"{train_accs['material']*100:>4.1f}%  "
              f"({time.time()-t0:.1f}s)")

        if val_loss < best_val_loss:
            best_val_loss  = val_loss
            best_weights   = {k: v.clone()
                              for k, v in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= patience:
                print(f"\n  Early stopping at epoch {epoch}")
                break

    model.load_state_dict(best_weights)
    return model, history


# ============================================================
#  SECTION 6: OBJECT LOOKUP ENGINE
#
#  Takes the predicted property vector and returns the
#  most specific object match from the lookup table.
#  Falls back gracefully if the exact combination is missing.
# ============================================================

def lookup_object(pred_shape, pred_size, pred_aspect,
                  pred_symmetry, pred_material):
    """
    Query the object lookup table with predicted properties.

    Tries exact match first. If not found, relaxes one
    property at a time until a match is found.
    Returns matched objects and confidence level.
    """
    key = (pred_shape, pred_size, pred_aspect,
           pred_symmetry, pred_material)

    # Exact match
    if key in OBJECT_LOOKUP:
        return OBJECT_LOOKUP[key], 'Exact match'

    # Relax material (most variable property in practice)
    for mat in range(N_MATERIALS):
        relaxed = (pred_shape, pred_size, pred_aspect, pred_symmetry, mat)
        if relaxed in OBJECT_LOOKUP:
            return OBJECT_LOOKUP[relaxed], 'Relaxed material'

    # Relax aspect ratio
    for asp in range(N_ASPECTS):
        relaxed = (pred_shape, pred_size, asp, pred_symmetry, pred_material)
        if relaxed in OBJECT_LOOKUP:
            return OBJECT_LOOKUP[relaxed], 'Relaxed aspect'

    # Relax size
    for sz in range(N_SIZES):
        relaxed = (pred_shape, sz, pred_aspect, pred_symmetry, pred_material)
        if relaxed in OBJECT_LOOKUP:
            return OBJECT_LOOKUP[relaxed], 'Relaxed size'

    # Shape-only fallback
    shape_name = SHAPE_CLASSES[pred_shape]
    fallback = {
        'Circle'   : ['Ball-like object', 'Cylindrical object'],
        'Square'   : ['Box-like object', 'Cubic object'],
        'Rectangle': ['Rectangular object', 'Flat panel'],
        'Triangle' : ['Conical object', 'Wedge-shaped object'],
    }
    return fallback[shape_name], 'Shape only (fallback)'


# ============================================================
#  SECTION 7: FULL INFERENCE PIPELINE
#
#  Input:  raw Range-Angle heatmap [range_bins × angle_bins]
#  Output: predicted shape + size + aspect + material
#          → specific real object prediction
# ============================================================

def predict_object_multitask(heatmap, model, visualise=True):
    """
    Full multi-task inference pipeline.
    Returns a detailed property profile and object prediction.
    """
    model.eval()

    # Preprocess
    hm = heatmap.astype(np.float32)
    if hm.max() > 0:
        hm = hm / hm.max()

    hm_t = torch.tensor(hm).unsqueeze(0).unsqueeze(0).to(device)

    # Forward pass — get all property predictions
    with torch.no_grad():
        outputs = model(hm_t)

    # Probabilities and predictions per property
    props = {}
    for name, head_output in outputs.items():
        probs   = torch.softmax(head_output, dim=1).cpu().numpy()[0]
        pred_idx = int(np.argmax(probs))
        props[name] = {
            'pred_idx'  : pred_idx,
            'probs'     : probs,
            'confidence': float(probs[pred_idx]),
        }

    # Extract predicted indices
    p_shape    = props['shape']['pred_idx']
    p_size     = props['size']['pred_idx']
    p_aspect   = props['aspect']['pred_idx']
    p_symmetry = props['symmetry']['pred_idx']
    p_material = props['material']['pred_idx']

    # Object lookup
    objects, match_type = lookup_object(
        p_shape, p_size, p_aspect, p_symmetry, p_material
    )

    # Build property profile string
    profile = {
        'Shape'    : SHAPE_CLASSES[p_shape],
        'Size'     : SIZE_CLASSES[p_size],
        'Aspect'   : ASPECT_CLASSES[p_aspect],
        'Symmetry' : SYMMETRY_CLASSES[p_symmetry],
        'Material' : MATERIAL_CLASSES[p_material],
    }

    # Console output
    print(f"\n{'='*55}")
    print(f"  MULTI-TASK RADAR OBJECT PREDICTION")
    print(f"{'='*55}")
    print(f"  Property Profile:")
    for prop_name, val in profile.items():
        conf = props[prop_name.lower()]['confidence']
        bar  = '█' * int(conf * 20)
        print(f"    {prop_name:12s}: {val:16s}  {conf*100:4.0f}%  {bar}")
    print(f"\n  Predicted Objects  : {', '.join(objects)}")
    print(f"  Lookup match type  : {match_type}")
    print(f"{'='*55}")

    if visualise:
        _plot_multitask_prediction(hm, props, profile, objects)

    return profile, objects, props


def _plot_multitask_prediction(hm, props, profile, objects):
    """Visualise heatmap + all 5 property confidence bars."""
    fig = plt.figure(figsize=(15, 6))
    fig.patch.set_facecolor('white')

    # Object title
    object_str = ' / '.join(objects[:3])
    fig.suptitle(f"Multi-Task Prediction → {object_str}",
                 fontsize=14, fontweight='bold')

    # ── Heatmap ──────────────────────────────────────────────
    ax_hm = fig.add_subplot(1, 6, 1)
    im = ax_hm.imshow(20*np.log10(hm + 1e-6),
                      cmap='jet', aspect='auto',
                      vmin=-40, vmax=0)
    plt.colorbar(im, ax=ax_hm, label='dB', fraction=0.046)
    ax_hm.set_title('Input Heatmap', fontsize=10)
    ax_hm.set_xlabel('Angle bins')
    ax_hm.set_ylabel('Range bins')

    # ── Property bars ─────────────────────────────────────────
    prop_config = [
        ('shape',    SHAPE_CLASSES,    'Shape',     '#3498DB'),
        ('size',     SIZE_CLASSES,     'Size',      '#E74C3C'),
        ('aspect',   ASPECT_CLASSES,   'Aspect',    '#2ECC71'),
        ('symmetry', SYMMETRY_CLASSES, 'Symmetry',  '#F39C12'),
        ('material', MATERIAL_CLASSES, 'Material',  '#9B59B6'),
    ]

    for idx, (key, class_names, title, color) in enumerate(prop_config):
        ax = fig.add_subplot(1, 6, idx + 2)
        probs = props[key]['probs']
        pred  = props[key]['pred_idx']

        bars = ax.barh(class_names, probs * 100,
                       color=color, alpha=0.7,
                       edgecolor='white', height=0.6)

        # Highlight predicted class
        bars[pred].set_alpha(1.0)
        bars[pred].set_edgecolor('black')
        bars[pred].set_linewidth(2.0)

        # Value labels
        for i, bar in enumerate(bars):
            ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                    f"{probs[i]*100:.0f}%",
                    va='center', fontsize=8)

        ax.set_xlim([0, 115])
        ax.set_title(f"{title}\n→ {class_names[pred]}",
                     fontsize=9, fontweight='bold')
        ax.set_xlabel('Confidence %', fontsize=8)
        ax.axvline(50, color='gray', linestyle='--', alpha=0.4, linewidth=1)
        ax.grid(axis='x', alpha=0.2)

    plt.tight_layout()
    plt.savefig('multitask_prediction.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: multitask_prediction.png")


# ============================================================
#  SECTION 8: PROPERTY ACCURACY PLOT
#  After training, shows accuracy per property head
#  to understand which properties the model learned best
# ============================================================

def plot_property_accuracies(model, test_loader):
    """Evaluate and plot per-property accuracy."""
    model.eval()
    correct = {p: 0 for p in
               ['shape','size','aspect','symmetry','material']}
    total = 0

    with torch.no_grad():
        for X_batch, Y_batch in test_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            total  += X_batch.size(0)

            for prop in correct:
                preds          = outputs[prop].cpu().argmax(dim=1)
                correct[prop] += (preds == Y_batch[prop]).sum().item()

    accuracies = {p: correct[p]/total*100 for p in correct}

    fig, ax = plt.subplots(figsize=(8, 4))
    props   = list(accuracies.keys())
    accs    = list(accuracies.values())
    colors  = ['#3498DB','#E74C3C','#2ECC71','#F39C12','#9B59B6']

    bars = ax.bar(props, accs, color=colors, edgecolor='white',
                  linewidth=1.5)
    ax.set_ylim([0, 110])
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Per-Property Accuracy — Multi-Task CNN',
                 fontsize=13, fontweight='bold')
    ax.axhline(100, color='gray', linestyle='--', alpha=0.4)
    ax.grid(axis='y', alpha=0.3)

    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 1,
                f"{acc:.1f}%",
                ha='center', fontsize=11, fontweight='bold')

    # Expected difficulty order annotation
    ax.text(0.98, 0.05,
            "Expected: Symmetry > Size > Material > Aspect > Shape",
            transform=ax.transAxes, fontsize=8,
            ha='right', color='gray', style='italic')

    plt.tight_layout()
    plt.savefig('property_accuracies.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: property_accuracies.png")

    print("\nPer-property test accuracy:")
    for prop, acc in accuracies.items():
        print(f"  {prop:12s}: {acc:.1f}%")

    return accuracies


# ============================================================
#  SECTION 9: MAIN
# ============================================================

if __name__ == '__main__':

    # ── Load MATLAB dataset ──────────────────────────────────
    print("Loading MATLAB dataset...")
    data = scipy.io.loadmat('radar_shapes_dataset.mat')

    X_train = np.expand_dims(
        data['X_train'].astype(np.float32), axis=1)
    X_val   = np.expand_dims(
        data['X_val'].astype(np.float32),   axis=1)
    X_test  = np.expand_dims(
        data['X_test'].astype(np.float32),  axis=1)

    Y_shape_train = data['Y_train'].flatten().astype(np.int64) - 1
    Y_shape_val   = data['Y_val'].flatten().astype(np.int64)   - 1
    Y_shape_test  = data['Y_test'].flatten().astype(np.int64)  - 1

    # ── Derive property labels ───────────────────────────────
    # These shape→property mappings encode what each shape's
    # simulation parameters physically imply
    shape_property_map = {
        # shape_idx: (size_m, aspect_ratio, max_amplitude)
        0: (0.25*2, 1.0,  0.35),   # Circle:    diam=0.5m, symmetric, soft
        1: (0.50,   1.0,  0.90),   # Square:    0.5m, symmetric, metal
        2: (0.70,   0.70/0.30, 0.90),  # Rectangle: 0.7×0.3m, wide, metal
        3: (0.50,   1.0,  0.90),   # Triangle:  0.5m, asymmetric, metal
    }

    def build_labels(Y_shape):
        N = len(Y_shape)
        sizes   = np.array([shape_property_map[s][0] for s in Y_shape])
        aspects = np.array([shape_property_map[s][1] for s in Y_shape])
        amps    = np.array([shape_property_map[s][2] for s in Y_shape])

        params = {'longest_dim':   sizes,
                  'aspect_ratio':  aspects,
                  'max_amplitude': amps}
        return derive_property_labels(Y_shape, params)

    labels_train = build_labels(Y_shape_train)
    labels_val   = build_labels(Y_shape_val)
    labels_test  = build_labels(Y_shape_test)

    # ── Build datasets ───────────────────────────────────────
    train_ds = RadarMultiTaskDataset(X_train, labels_train, augment=True)
    val_ds   = RadarMultiTaskDataset(X_val,   labels_val,   augment=False)
    test_ds  = RadarMultiTaskDataset(X_test,  labels_test,  augment=False)

    train_loader = DataLoader(train_ds, batch_size=64,
                              shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=64,
                              shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=64,
                              shuffle=False, num_workers=0)

    # ── Build and train model ────────────────────────────────
    model  = MultiTaskRadarCNN()
    params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {params:,}")

    model, history = train_multitask(
        model, train_loader, val_loader,
        n_epochs=40, lr=1e-3, patience=8
    )

    # ── Evaluate ─────────────────────────────────────────────
    accuracies = plot_property_accuracies(model, test_loader)

    # ── Save ─────────────────────────────────────────────────
    torch.save({
        'model_state_dict' : model.state_dict(),
        'property_names'   : ['shape','size','aspect','symmetry','material'],
        'class_names'      : {
            'shape'    : SHAPE_CLASSES,
            'size'     : SIZE_CLASSES,
            'aspect'   : ASPECT_CLASSES,
            'symmetry' : SYMMETRY_CLASSES,
            'material' : MATERIAL_CLASSES,
        },
        'object_lookup'    : OBJECT_LOOKUP,
    }, 'radar_multitask_model.pt')
    print("\nModel saved: radar_multitask_model.pt")

    # ── Demo inference ───────────────────────────────────────
    sample_hm = test_ds[0][0].numpy().squeeze()
    profile, objects, props = predict_object_multitask(
        sample_hm, model, visualise=True
    )
