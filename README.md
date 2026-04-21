# Independent Work — Radar Object Detection + ML Classification

A pipeline for classifying the **shape** of an object from its FMCW radar
Range-Angle heatmap. Training data is synthetic and generated in MATLAB;
models are trained and run in Python (PyTorch). For inference on
real-world objects, an STL file is converted into a radar heatmap via
the same simulation model and then fed to the trained CNN.

Project: Junior IW — ECE 399 (Princeton).

---

## Pipeline

### Training
```
MATLAB synthetic scatterer model
      → Range-Angle heatmaps  (.mat dataset)
      → PyTorch CNN training
      → trained model + history plots
```

### Inference
```
Object (Tinkercad / CAD)
      → STL export
      → STL → scatterers → Range-Angle heatmap (MATLAB)
      → predict shape with trained CNN (Python)
```

### Shape classes
Circle · Square · Rectangle · Triangle · Oval

### Dataset size per shape
36 orientations × 3 sizes × 4 ranges = 432 samples
→ 5 shapes × 432 = **2,160 samples** per antenna configuration.

The multi-antenna data generator sweeps `N_ant ∈ {4, 8, 10, 12, 16, 32}`
to produce one `.mat` file per antenna count so the CNN can be trained
independently for each.

---

## Repository layout

```
IW/
├── README.md                       This file
│
├── src/
│   ├── matlab/
│   │   ├── data_generation/        Synthetic dataset generators
│   │   │   ├── radar_data_generation_single_antenna.m
│   │   │   └── radar_data_generation_multiple_antenna.m
│   │   ├── inference/              STL → heatmap for real objects
│   │   │   ├── generate_RA.m            helper used by radar_stl_to_heatmap
│   │   │   ├── radar_stl_to_heatmap.m
│   │   │   └── verify_dim_stl.m         sanity-check STL dimensions
│   │   └── exploration/            Early learning / DSP fundamentals
│   │       └── radar_simulation_basics.m
│   └── python/
│       ├── training/               PyTorch CNN trainers
│       │   ├── radar_obj_classifier_cnn_single_antenna.py
│       │   └── radar_obj_classifier_cnn_multiple_antenna.py
│       └── ui/
│           └── streamlit_radar.py       interactive DSP demo
│
├── data/
│   ├── stl/                        STL test objects (400 mm family)
│   └── heatmaps/                   .mat heatmaps generated from those STLs
│
├── results/
│   └── N4/                         first training run (N_ant = 4)
│       ├── first_training_stats.txt
│       └── history.png
│
└── Archive/                        Prior iterations — kept for reference
    ├── matlab/                     older .m and .mlx files
    ├── python/                     older .py files
    └── docs/                       old READMEs, LICENSE
```

See `Archive/` for earlier approaches (1-D CNN / SVM baseline,
multi-task CNN with shape+size+aspect+symmetry+material heads, CFAR-style
FMCW target detection, Radar Toolbox–based generator).

---

## Usage

### 1. Generate synthetic training data (MATLAB)

From `src/matlab/data_generation/`:

```matlab
% single antenna configuration
radar_data_generation_single_antenna

% or, sweep over antenna counts {4, 8, 10, 12, 16, 32}
radar_data_generation_multiple_antenna
```

This writes one or more `radar_shapes_*.mat` files containing the
Range-Angle heatmaps and labels.

### 2. Train the CNN (Python)

From `src/python/training/`:

```bash
# train on the single-antenna dataset
python radar_obj_classifier_cnn_single_antenna.py

# train one model per antenna count and compare accuracy vs N_ant
python radar_obj_classifier_cnn_multiple_antenna.py

# or train on just one antenna file
python radar_obj_classifier_cnn_multiple_antenna.py --single radar_shapes_N16.mat
```

Dependencies: `torch scipy h5py numpy matplotlib seaborn scikit-learn`.

### 3. Inference on a real / CAD object

1. Design the shape in Tinkercad (or any CAD tool) and export as `.stl`.
   Place the file in `data/stl/`.
2. In `src/matlab/inference/radar_stl_to_heatmap.m`, set `STL_FILE`,
   `TRUE_SHAPE`, `OBJ_RANGE`, `OBJ_ORIENTATION`.
3. Run the script to produce a `.mat` heatmap (save into `data/heatmaps/`).
4. Run the trained CNN on the heatmap.

### 4. Interactive DSP demo (optional)

```bash
streamlit run src/python/ui/streamlit_radar.py
```

---

