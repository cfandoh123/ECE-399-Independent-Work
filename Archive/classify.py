"""
classify.py
-----------
Trains and evaluates two classifiers on the preprocessed radar dataset:
  1. SVM  — fast baseline, interpretable
  2. 1D CNN — learns spatial features along the range axis

Run from the Radar_Simulation/ directory:
    python phase2/classify.py

Requirements:
    pip install numpy scikit-learn matplotlib torch
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix,
                              ConfusionMatrixDisplay, accuracy_score)

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(SCRIPT_DIR, 'radar_preprocessed.npz')

# ── Load preprocessed data ─────────────────────────────────────────────────────
print("Loading radar_preprocessed.npz...")
data        = np.load(DATA_PATH, allow_pickle=True)
X_train     = data['X_train'].astype(np.float32)
y_train     = data['y_train'].astype(np.int64)
X_val       = data['X_val'].astype(np.float32)
y_val       = data['y_val'].astype(np.int64)
X_test      = data['X_test'].astype(np.float32)
y_test      = data['y_test'].astype(np.int64)
rng_ax      = data['rng_ax']
class_names = list(data['class_names'])

# Labels in the .mat are 1-indexed (1=ball, 2=cup, 3=chair).
# Convert to 0-indexed for PyTorch (0=ball, 1=cup, 2=chair).
y_train_0 = y_train - 1
y_val_0   = y_val   - 1
y_test_0  = y_test  - 1

n_features = X_train.shape[1]
n_classes  = len(class_names)
print(f"  Features : {n_features}  |  Classes : {class_names}")
print(f"  Train {len(y_train)}  |  Val {len(y_val)}  |  Test {len(y_test)}\n")


# ══════════════════════════════════════════════════════════════════════════════
#  MODEL 1 — SVM
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("MODEL 1: SVM (RBF kernel)")
print("=" * 60)

scaler       = StandardScaler()
X_tr_sc      = scaler.fit_transform(X_train)
X_val_sc     = scaler.transform(X_val)
X_te_sc      = scaler.transform(X_test)

svm = SVC(kernel='rbf', C=10, gamma='scale', random_state=42)
svm.fit(X_tr_sc, y_train)

svm_val_acc  = accuracy_score(y_val,  svm.predict(X_val_sc))
svm_test_acc = accuracy_score(y_test, svm.predict(X_te_sc))

print(f"  Validation accuracy : {svm_val_acc*100:.1f}%")
print(f"  Test accuracy       : {svm_test_acc*100:.1f}%\n")
print(classification_report(y_test, svm.predict(X_te_sc),
                             target_names=class_names))


# ══════════════════════════════════════════════════════════════════════════════
#  MODEL 2 — 1D CNN
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("MODEL 2: 1D CNN")
print("=" * 60)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"  Device: {device}\n")

# ── Architecture ───────────────────────────────────────────────────────────────
class RadarCNN(nn.Module):
    """
    Two conv layers extract local range-profile features.
    MaxPool halves the sequence length each time.
    Two FC layers map to class logits.

    Input shape: (batch, 1, n_features)
    """
    def __init__(self, n_features, n_classes):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, padding=2),  # (B, 16, L)
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),                             # (B, 16, L/2)

            nn.Conv1d(16, 32, kernel_size=5, padding=2), # (B, 32, L/2)
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),                             # (B, 32, L/4)
        )
        conv_out = 32 * (n_features // 4)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_out, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, n_classes)
        )

    def forward(self, x):
        return self.fc(self.conv(x))


# ── DataLoaders ────────────────────────────────────────────────────────────────
def make_loader(X, y, batch_size=64, shuffle=False):
    X_t = torch.tensor(X).unsqueeze(1)   # (N, 1, n_features) — channel dim for Conv1d
    y_t = torch.tensor(y)
    return DataLoader(TensorDataset(X_t, y_t), batch_size=batch_size, shuffle=shuffle)

train_loader = make_loader(X_train, y_train_0, batch_size=64, shuffle=True)
val_loader   = make_loader(X_val,   y_val_0,   batch_size=128)
test_loader  = make_loader(X_test,  y_test_0,  batch_size=128)

# ── Training ───────────────────────────────────────────────────────────────────
model     = RadarCNN(n_features, n_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

N_EPOCHS   = 30
train_loss = []
val_acc_history = []

for epoch in range(1, N_EPOCHS + 1):
    # -- train --
    model.train()
    epoch_loss = 0.0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        loss = criterion(model(X_batch), y_batch)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * len(y_batch)
    train_loss.append(epoch_loss / len(y_train))

    # -- validate --
    model.eval()
    correct = 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            preds = model(X_batch.to(device)).argmax(dim=1).cpu()
            correct += (preds == y_batch).sum().item()
    val_acc_history.append(correct / len(y_val))
    scheduler.step()

    if epoch % 5 == 0 or epoch == 1:
        print(f"  Epoch {epoch:3d}/{N_EPOCHS}  "
              f"loss={train_loss[-1]:.4f}  "
              f"val_acc={val_acc_history[-1]*100:.1f}%")

# ── Test evaluation ────────────────────────────────────────────────────────────
model.eval()
all_preds, all_true = [], []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        preds = model(X_batch.to(device)).argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_true.extend(y_batch.numpy())

all_preds = np.array(all_preds)
all_true  = np.array(all_true)
cnn_test_acc = accuracy_score(all_true, all_preds)

print(f"\n  Test accuracy: {cnn_test_acc*100:.1f}%\n")
print(classification_report(all_true, all_preds, target_names=class_names))


# ══════════════════════════════════════════════════════════════════════════════
#  PLOTS
# ══════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(14, 10))
gs  = gridspec.GridSpec(2, 3, figure=fig)

# -- CNN training curves --
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(train_loss, label='Train loss')
ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss')
ax1.set_title('CNN Training Loss'); ax1.grid(True); ax1.legend()

ax2 = fig.add_subplot(gs[0, 1])
ax2.plot([v*100 for v in val_acc_history])
ax2.axhline(100/n_classes, color='gray', linestyle='--', label='Chance')
ax2.set_xlabel('Epoch'); ax2.set_ylabel('Accuracy (%)')
ax2.set_title('CNN Validation Accuracy'); ax2.grid(True); ax2.legend()

# -- Accuracy comparison bar chart --
ax3 = fig.add_subplot(gs[0, 2])
bars = ax3.bar(['SVM', 'CNN'], [svm_test_acc*100, cnn_test_acc*100],
               color=['steelblue', 'darkorange'], width=0.4)
ax3.bar_label(bars, fmt='%.1f%%', padding=3)
ax3.set_ylim([0, 110]); ax3.set_ylabel('Test Accuracy (%)')
ax3.set_title('Model Comparison'); ax3.grid(axis='y')

# -- Confusion matrices --
ax4 = fig.add_subplot(gs[1, 0])
cm_svm = confusion_matrix(y_test, svm.predict(X_te_sc))
ConfusionMatrixDisplay(cm_svm, display_labels=class_names).plot(ax=ax4, colorbar=False)
ax4.set_title('SVM Confusion Matrix')

ax5 = fig.add_subplot(gs[1, 1])
cm_cnn = confusion_matrix(all_true, all_preds)
ConfusionMatrixDisplay(cm_cnn, display_labels=class_names).plot(ax=ax5, colorbar=False)
ax5.set_title('CNN Confusion Matrix')

# -- Per-class accuracy comparison --
ax6 = fig.add_subplot(gs[1, 2])
x = np.arange(n_classes)
w = 0.35
svm_per_class = cm_svm.diagonal() / cm_svm.sum(axis=1) * 100
cnn_per_class = cm_cnn.diagonal() / cm_cnn.sum(axis=1) * 100
ax6.bar(x - w/2, svm_per_class, w, label='SVM',  color='steelblue')
ax6.bar(x + w/2, cnn_per_class, w, label='CNN',  color='darkorange')
ax6.set_xticks(x); ax6.set_xticklabels(class_names)
ax6.set_ylabel('Per-class Accuracy (%)')
ax6.set_title('Per-class Accuracy'); ax6.legend(); ax6.grid(axis='y')

fig.suptitle('Phase 2: Radar Object Classification Results', fontsize=14)
fig.tight_layout()
plot_path = os.path.join(SCRIPT_DIR, 'classification_results.png')
plt.savefig(plot_path, dpi=120)
plt.show()
print(f"Results plot saved: {plot_path}")

# ── Save model weights ─────────────────────────────────────────────────────────
model_path = os.path.join(SCRIPT_DIR, 'radar_cnn.pth')
torch.save({'model_state': model.state_dict(),
            'n_features':  n_features,
            'n_classes':   n_classes,
            'class_names': class_names,
            'scaler_mean': scaler.mean_,
            'scaler_std':  scaler.scale_}, model_path)
print(f"CNN weights saved : {model_path}")
