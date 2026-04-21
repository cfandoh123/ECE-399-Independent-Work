# FMCW Radar Shape Classification

**Progress Report — Junior Independent Work**

*Radar Object Detection + Machine Learning*

Advisor: Prof. Yasaman Ghasempour
Princeton University  |  ECE 399  |  Spring 2026

---

## Executive Summary

This report documents the second phase of an independent research project investigating the application of machine learning to FMCW (Frequency-Modulated Continuous Wave) radar data. The long-term objective of the project is a radar-based recognition system suitable for privacy-preserving monitoring — a system that can identify objects and eventually human activity without the use of optical cameras.

The first phase of the project explored radar-based classification of concrete objects (chairs, tables, people) directly. That approach proved unworkably broad for a single-semester scope: the space of real-world objects is effectively unbounded, each object requires a separate scatterer model, and the resulting dataset would be heavily biased toward whichever objects happen to be included. The current phase narrows the problem deliberately.

The reformulated task is **shape classification**: given a Range-Angle heatmap produced from an FMCW radar scene, predict which of five geometric primitives (Circle, Square, Rectangle, Triangle, Oval) generated the reflection. Shapes are a tractable proxy because every real object decomposes, at some level of abstraction, into a small set of recognisable silhouettes. A classifier that reliably maps a radar heatmap to a geometric primitive is the first step toward the broader recognition pipeline; one that cannot is unlikely to succeed at the harder problem.

The work described here covers: (i) a synthetic data-generation pipeline implemented in MATLAB that converts scatterer models into Range-Angle heatmaps at configurable antenna counts; (ii) a convolutional neural network trained in PyTorch to classify those heatmaps; (iii) an STL-to-heatmap inference pipeline that applies the trained model to CAD objects; and (iv) an experimental protocol — multi-seed training across a sweep of antenna counts — that is designed to produce statistically defensible conclusions about the trade-off between hardware cost (number of receive antennas) and classification performance. Results from the full sweep are forthcoming and are reserved for Section 9.

---

## Table of Contents

1. Project Scope and Motivation
2. Theoretical Foundation
3. Dataset Design
4. Model Selection — From SVM to CNN
5. CNN Architecture
6. Training Methodology
7. Experimental Protocol
8. Inference on CAD Objects
9. Results (placeholder — awaiting full sweep)
10. Discussion
11. Summary and Next Steps

---

## 1. Project Scope and Motivation

### 1.1 Long-term Vision

The long-term goal of the project is to use FMCW radar as an input modality for recognition tasks that are today performed with optical cameras. FMCW radar has three properties that make it attractive for privacy-sensitive applications: it does not produce identifiable images of people, it operates in complete darkness, and it can penetrate light obstructions such as bedding or upholstery. Concrete applications include elderly-care monitoring, occupancy sensing in buildings, and activity recognition without on-camera surveillance.

### 1.2 Why Shape Classification First

The initial formulation of this project was radar-based classification of specific real-world objects — distinguishing, for example, a chair from a table from a standing person. In practice that framing has two problems:

1. **Unbounded class set.** There is no principled way to decide which objects belong in the training set. Adding or removing a class (a sofa, a dog, a bicycle) changes every design decision downstream, from the scatterer model to the dataset size to the network architecture.
2. **High per-class modelling cost.** Each real object requires its own multi-point scatterer model with geometry-specific and material-specific parameters. A convincing "chair" model, for example, requires four leg scatterers, a seat scatterer, and a backrest scatterer, each with amplitudes tuned to plausible material values. Producing many such models by hand is slow and error-prone, and any real application would need dozens.

Shape classification sidesteps both problems. The five shapes chosen — Circle, Square, Rectangle, Triangle, Oval — cover the major categories of two-dimensional silhouette (rotationally symmetric, right-angled, elongated-smooth, elongated-angular) without the arbitrariness of an object list. Every real object, viewed top-down, has a silhouette that is approximately one of these shapes; a classifier that can identify the silhouette is a natural precursor to one that can identify the object.

Framed this way, shape classification is not a toy problem but a tractable building block. It exposes every piece of the eventual pipeline — synthetic data generation, sim-to-real transfer, heatmap representation, network architecture, evaluation protocol — in a setting where ground truth is unambiguous and the class boundary is well-defined.

### 1.3 Scope of This Report

This report covers the shape-classification phase. Section 2 reviews the radar theory required to understand the input representation. Section 3 describes the synthetic dataset and its design. Sections 4 through 6 cover model selection, architecture, and training methodology. Sections 7 and 8 describe the experimental protocol and the inference pipeline on CAD objects. Section 9 is a placeholder for the results of the full multi-seed antenna-count sweep. Section 10 discusses two questions that motivate the next phase of the work: why some shapes are easier to classify than others, and why per-shape performance depends on the number of receive antennas in a non-monotonic way.

---

## 2. Theoretical Foundation

### 2.1 FMCW Radar and the Range Measurement

An FMCW radar transmits a linear chirp — a sinusoid whose frequency rises linearly over a fixed duration `T_c`. Mixing the received reflection with the transmitted chirp produces an Intermediate Frequency (IF) signal whose dominant frequency component is proportional to the round-trip delay to the target. Concretely, for a target at range `d`:

```
f_beat = S × (2d / c)
```

where `S = B / T_c` is the chirp slope. Taking an FFT along fast-time samples within one chirp converts the beat frequency into a range axis. The fundamental range resolution is `d_res = c / (2B)` — it depends only on bandwidth, not on chirp duration or sampling rate.

For the present project the radar parameters are chosen to match the automotive radar band in common use, so that the simulation remains directly portable to hardware like the Texas Instruments IWR1642:

| Parameter            | Value   | Rationale                                 |
|----------------------|---------|-------------------------------------------|
| Carrier frequency    | 77 GHz  | Standard automotive/ISM band              |
| Bandwidth `B`        | 4 GHz   | Range resolution of 3.75 cm               |
| Chirp duration `T_c` | 40 μs   | Consistent with lecture baseline          |
| ADC sample rate      | 4 MHz   | Sets unambiguous range of ~15 m           |
| Antenna spacing `D`  | λ/2     | Full ±90° angular field of view           |

### 2.2 Angle of Arrival and the Resolution Limit

A target at off-boresight angle `θ` imposes a linear phase progression across a receive array of spacing `D`. Taking an FFT across the antenna dimension ("spatial FFT") recovers the angle:

```
θ = sin⁻¹(λω / 2πD)
```

The resolution of this measurement is set almost entirely by the number of physical receive antennas `N`:

```
θ_res ≈ 2 / N   (radians, at boresight)
```

For `N = 4` the resolution is about 29°; for `N = 32` it is about 3.6°. This relationship is the single most important design variable in the present project, because it directly controls how finely the classifier can resolve the spatial structure of a shape. **Zero-padding the angle FFT does not improve resolution** — only adding physical antennas does. This is the reason antenna count appears explicitly in the experimental sweep.

### 2.3 Why the Range-Angle Heatmap

The natural four-dimensional radar data cube has axes `(sample, chirp, antenna, frame)`. Applying FFTs along the first three dimensions yields `(range, velocity, angle)` at each frame. From this cube one can extract either a **Range-Doppler map** (averaged across antennas) or a **Range-Angle map** (averaged across chirps).

For the shape-classification task, the Range-Angle map is the correct representation, for the following reasons:

1. **Shapes are static.** They do not move during a single frame, so the Doppler axis carries no discriminative information — every shape produces a spike at zero velocity. The Range-Doppler map would effectively reduce to a 1-D range profile.
2. **Shape information lives in the spatial dimensions.** A shape is defined by how its scatterers are distributed in 2-D space. The Range-Angle map is the radar-side projection of that 2-D layout.
3. **Integration across chirps is denoising.** A single chirp gives one noisy snapshot of the range-angle footprint; averaging over all chirps in a frame yields a cleaner representation without losing information that varies on a chirp timescale (because, for static shapes, nothing of interest does).

The output shape of each heatmap is **512 × 256** (range bins × angle bins) after padding; this is the input size the CNN expects.

---

## 3. Dataset Design

### 3.1 Synthetic Generation via Scatterer Models

Each shape is represented in the simulator as a set of point scatterers placed around its geometric outline. A scatterer at local position `(x, y)` with amplitude `a` contributes a complex exponential to the IF signal whose phase terms encode its true range and angle relative to the radar. The MATLAB generator rotates the scatterer set into world coordinates according to the shape's target orientation, places it at the target range, and synthesises the full chirp-by-chirp, antenna-by-antenna IF signal before FFT-ing into the Range-Angle heatmap. Gaussian noise is added at fixed SNR to approximate ADC noise.

This **physics-based synthesis** is preferred over any form of heatmap-level data augmentation for three reasons: ground truth (range, angle, orientation, size) is known exactly; the simulation is deterministic and reproducible; and the forward model is the same one used for inference on CAD objects (Section 8), which makes the sim-to-real story a single consistent pipeline rather than two separate codepaths.

### 3.2 The Five-Shape Set

The five classes were chosen to span the major categories of 2-D silhouette with minimal overlap:

| Class      | Geometric Character              | Scatterer Signature                          |
|------------|----------------------------------|----------------------------------------------|
| Circle     | Rotationally symmetric, smooth   | Uniform ring (24 curve scatterers, amp 0.50) |
| Square     | 4-fold symmetric, angular        | 4 dominant corners (amp 1.00) + edges        |
| Rectangle  | 2-fold symmetric, angular        | 4 corners + elongated edges                  |
| Triangle   | 3-fold symmetric, angular        | 3 dominant corners + edges                   |
| Oval       | 2-fold symmetric, smooth         | Elongated curve ring (smooth, no corners)    |

The amplitude hierarchy — corner scatterers at 1.00, edges at 0.70, curves at 0.50 — reflects the physical intuition that metallic corners act as retroreflectors while smooth surfaces scatter energy isotropically. This asymmetry is what makes corner-bearing shapes (Square, Rectangle, Triangle) distinguishable from smooth shapes (Circle, Oval) even in principle.

The set covers three important axes of variation:

- **Symmetry order** (Circle/Oval/Square/Rectangle/Triangle = ∞, 2, 4, 2, 3): how many orientations look identical.
- **Angularity** (corner-bearing vs smooth): presence or absence of strong point reflectors.
- **Aspect ratio** (isotropic Circle/Square/Triangle vs elongated Rectangle/Oval): spatial extent asymmetry between two axes.

Any two shapes in the set differ on at least one axis, which is what makes the task learnable. Section 10 returns to this taxonomy to explain why the network finds some classes harder than others.

### 3.3 Diversity Axes

For each shape, the dataset is generated by sweeping three independent axes:

| Axis        | Values                       | Count | Rationale                                    |
|-------------|------------------------------|-------|----------------------------------------------|
| Orientation | 0° to 350° in 10° steps      | 36    | Full rotational coverage of the class         |
| Size        | Small, Medium, Large         | 3     | Scale invariance within physical bounds       |
| Range       | 4 distances (short → long)   | 4     | Range-dependent angular resolution in metres  |

This gives `36 × 3 × 4 = 432` samples per shape and **5 × 432 = 2,160 samples** per antenna-count dataset. The dataset is partitioned 80/10/10 for train/val/test.

The orientation sweep is the most important axis. A rectangle at 0° and at 90° produce completely different Range-Angle footprints, so the model must learn the class as an invariance over the orientation orbit, not as a single canonical appearance. The 10° step is chosen fine enough that the training set covers each shape's orbit densely, but coarse enough that the total dataset remains tractable for a single-GPU experiment.

### 3.4 Antenna-Count Sweep

The MATLAB generator produces one `.mat` file per value of `N_ant ∈ {4, 8, 10, 12, 16, 32}`. The scatterer physics is identical across files; only the angular dimension of the Range-Angle heatmap differs, because the angle FFT is taken across the simulated receive array. This isolates antenna count as the single varying factor — any accuracy difference between `N_ant` experiments is attributable to angular resolution, not to dataset or model differences.

### 3.5 Normalisation

Each heatmap is divided by its maximum amplitude before storage, producing values in [0, 1]. This removes an otherwise-confounding effect: by the radar range equation, received amplitude falls off roughly as `1/d²`, so without per-sample normalisation the network would learn range instead of shape. Normalisation forces the classifier to use the **pattern** of reflections rather than their absolute energy.

---

## 4. Model Selection — From SVM to CNN

### 4.1 The SVM Starting Point

The initial model considered for shape classification was a support vector machine (SVM) with hand-crafted features. This choice was motivated by the usual reasons to prefer classical methods on small datasets: interpretable features, calibrated probabilities via Platt scaling, strong performance on tens-of-samples-per-class regimes, and training costs measured in seconds rather than hours.

For this task, the hand-crafted feature set would have to capture the spatial layout of each shape's scatterers. Plausible candidates include:

- Moment-based descriptors of the Range-Angle heatmap (central moments, Hu moments) to capture overall mass distribution and elongation.
- Peak statistics: number of local maxima above a threshold, mean and variance of peak amplitudes, distances between top-`k` peaks.
- Radial profile of the heatmap around its centre of mass — a 1-D summary that is often discriminative for radially-symmetric versus elongated shapes.
- Angular histogram of peak locations — where, in angle, most of the energy is concentrated.

An SVM with an RBF kernel over such a feature vector would be a standard and respectable baseline.

### 4.2 Why Hand-Crafted Features Fall Short Here

The SVM approach was deprioritised in favour of a CNN for three reasons specific to Range-Angle heatmaps:

1. **Feature design is the task.** The features listed above are, in effect, guesses at what is discriminative. They are reasonable guesses — but the entire point of the experiment is to learn how much angular resolution is required to tell shapes apart, not to pre-commit to a fixed set of descriptors that may themselves fail at low resolution. A Moment-based descriptor that works well at `N_ant = 32` may lose all discriminative power at `N_ant = 4`, because the underlying heatmap has become smeared in the angle dimension. A learned feature extractor can adapt to that degradation; a hand-crafted one cannot.
2. **Orientation is not a nuisance variable, it is the data.** The dataset contains every shape at 36 orientations. For an SVM to achieve orientation invariance it must either (a) be trained on explicit features already invariant to rotation (harder to design than it sounds for anisotropic inputs like Range-Angle maps, where the two axes have different physical meaning), or (b) rely on the training set to learn invariance, which defeats the point of SVMs on small data. A CNN can, in principle, learn the orientation orbit directly from examples, and the convolutional structure shares parameters across the 2-D input in a way that helps.
3. **Small sample complexity cuts both ways.** The usual argument for SVMs — that they shine with few samples — only holds when the features are already good. With a poor feature set, SVMs plateau at the feature set's implicit ceiling and cannot trade compute for accuracy the way a deeper model can. The 2,160 samples per `N_ant` is small by modern standards but is not so small that a modest CNN overfits catastrophically, particularly with batch normalisation, dropout, and a compact global-average-pooled head.

A second, broader reason is architectural. A Range-Angle heatmap is a 2-D image; CNNs are designed to exploit the spatial correlations of 2-D inputs. An SVM treats its feature vector as an unordered bag, which discards precisely the locality information that a corner-bearing shape's heatmap exhibits.

### 4.3 Alternatives Considered and Deferred

Three other architecture families were considered and deferred. A **3-D CNN** over the full range-Doppler-angle cube was rejected because the Doppler axis is not informative for static shapes (Section 2.3), so it would waste parameters. A **PointNet** over the CFAR-detected point cloud was rejected because CFAR itself introduces threshold-sensitive parameters that would confound the antenna-count sweep, and because sub-threshold information useful for discriminating smooth shapes would be lost. A **CNN + LSTM sequence model** is appropriate for future posture-classification work where temporal evolution matters, but is unnecessary for single-frame shape classification and would add a great deal of complexity with no payoff at this stage.

The chosen 2-D CNN is therefore both the simplest sensible architecture for the input representation and the one whose inductive bias most closely matches the structure of the task.

---

## 5. CNN Architecture

### 5.1 Overview

The classifier is a four-block 2-D CNN (`ShapeCNN`) followed by global average pooling and a small fully-connected head. Input is a single-channel Range-Angle heatmap of size 512 × 256; output is a five-way logit vector. Total parameter count is on the order of 10⁵, which is deliberately modest for the dataset size.

### 5.2 Feature Extractor

The feature extractor uses a coarse-to-fine schedule of convolution kernel sizes. Each block is `Conv → BatchNorm → ReLU → MaxPool(2)`:

| Block | Kernel | Filters | Output spatial size | Role                                   |
|-------|--------|---------|---------------------|----------------------------------------|
| 1     | 7 × 7  | 16      | 256 × 128           | Broad shape silhouette, large receptive field |
| 2     | 5 × 5  | 32      | 128 × 64            | Mid-scale structure (edges, spacing)   |
| 3     | 3 × 3  | 64      | 64 × 32             | Local peak shapes and corner features  |
| 4     | 3 × 3  | 128     | 32 × 16             | High-level compositions of the above    |

The 7 × 7 kernel in the first block is deliberate: the raw heatmap is large (512 × 256) relative to the feature of interest (the shape footprint spans perhaps 20–80 pixels depending on range and size), so the first layer needs a big enough receptive field to see the overall silhouette in one view. Subsequent blocks shrink the kernel as the feature maps shrink. Batch normalisation in every block stabilises training across the antenna-count sweep, where input statistics vary substantially between `N_ant = 4` and `N_ant = 32`.

### 5.3 Global Average Pooling and Classifier Head

The final feature map (128 × 32 × 16) is collapsed by global average pooling to a 128-dimensional vector, which is then fed to a two-layer MLP: `Linear(128 → 64) → ReLU → Dropout(0.4) → Linear(64 → 5)`.

GAP is preferred over `Flatten → Linear` for two reasons. It has no spatial parameters, so the parameter count is independent of input size — this is important because changing `N_ant` in the MATLAB generator may change the exact angle-axis dimensions. And it acts as a strong regulariser against overfitting on a small dataset, because the classifier must rely on which features are present rather than where they are on the feature map.

### 5.4 Regularisation Choices

The network uses three forms of regularisation. L2 weight decay (`1e-4`) provides baseline control over weight magnitudes. Dropout (0.4) in the classifier head reduces co-adaptation of the learned features. And horizontal flip augmentation at the input (applied stochastically with probability 0.5 per sample during training) doubles the effective rotational coverage — a rectangle oriented at `+θ` flipped horizontally becomes a rectangle at `-θ`, which is a valid synthetic sample for every shape in the set.

---

## 6. Training Methodology

### 6.1 Loss and Optimiser

Cross-entropy loss is used with the Adam optimiser (`lr = 3e-4`, `weight_decay = 1e-4`). Gradient clipping to a maximum L2 norm of 1.0 is applied before each step. Cross-entropy is the natural choice for a five-way single-label problem; Adam is chosen over SGD+momentum because the dataset is small enough that the adaptive per-parameter learning rates of Adam reach a competitive solution faster, and because the training budget is 40–60 epochs per run — too few for SGD's slow-and-steady regime to fully play out.

### 6.2 Batch Size

Training uses a batch size of 256; evaluation uses 512. The batch choice is the largest that reliably fits in GPU memory at the 512 × 256 input resolution. Larger batches reduce the variance of the gradient estimate, which is what we want here because the dataset is small and per-batch loss is otherwise noisy. Evaluation can use an even larger batch because no gradients are stored.

### 6.3 Learning-Rate Schedule

The schedule is cosine annealing from `3e-4` down to `1e-6` over the full epoch budget, with no warm restarts. This replaces an earlier `ReduceLROnPlateau` schedule that turned out to be poorly suited to the noisy validation loss produced by a small validation set: the plateau detector was firing on noise, producing learning-rate drops that were more a property of the sampling variance than of genuine convergence. Cosine annealing is deterministic and schedule-driven, so it produces more comparable training curves across seeds and across antenna counts.

### 6.4 Validation Smoothing and Early Stopping

Validation loss and accuracy are further smoothed by an exponential moving average (`α = 0.3`). Early stopping triggers on the EMA-smoothed validation loss with a patience of 10 epochs. Without smoothing, the raw validation loss oscillates by several percentage points between adjacent epochs, simply because the validation set is 216 samples and one mis-classified Rectangle is already a ~0.5% swing. Early stopping on the raw signal would therefore react to sampling noise; early stopping on the EMA responds only to trends.

### 6.5 Multi-Seed Protocol

Every `(N_ant, seed)` pair is trained independently. The default seed set is `{0, 1, 2, 3, 4}`. For each seed, the PyTorch, NumPy, and Python random-number generators are re-seeded before data loading and before model initialisation, so that both the DataLoader shuffle and the network's initial weights are controlled by the seed. Aggregate statistics (mean and sample standard deviation across seeds) are reported in the final comparison plot; individual per-seed accuracies are also retained in `results_per_seed.csv` for the writeup.

The rationale for this protocol is simple: with only ~216 test samples and with network performance that depends on initialisation, a single-seed test accuracy is a sample from a high-variance distribution, not a point estimate of the true accuracy. A preliminary 3-seed run at `N_ant = 16` produced test accuracies of 51.9%, 78.8%, and 75.0% — a 27-point spread. Reporting any single one of these as "the accuracy at `N_ant = 16`" would be misleading. The full sweep therefore averages five seeds per antenna count, so that conclusions about the optimal `N_ant` rest on statistically defensible means rather than on which seed happened to land where.

---

## 7. Experimental Protocol

### 7.1 Antenna-Count Sweep

The headline experiment trains the same `ShapeCNN` architecture on each of six antenna datasets (`N_ant ∈ {4, 8, 10, 12, 16, 32}`) with five seeds per dataset, for 30 total training runs. Each run records training/validation loss and accuracy per epoch, a confusion matrix on the held-out test set, and a single `best_val_ema_loss` checkpoint.

From these runs, three aggregate figures are produced:

1. **Overall accuracy vs antenna count**: mean test accuracy with ±1 standard-deviation error bars, plus a secondary axis showing the corresponding angular resolution in degrees.
2. **Per-class accuracy by antenna count**: grouped bar chart showing how each of the five shape classes behaves as `N_ant` changes.
3. **Per-antenna confusion matrix**: the best-seed confusion matrix at each `N_ant`, normalised by true-class counts.

### 7.2 Metrics

Primary metric is **test accuracy** (fraction of correctly predicted samples). Secondary metrics are **per-class recall** (how often the model recovers each true class) and the off-diagonal entries of the confusion matrix (which pairs of classes are most frequently confused). Precision and F1 are less informative here because the test set is class-balanced by construction, so recall and precision are numerically close.

### 7.3 Ablations

Two ablations are planned but not yet complete. The first removes the horizontal-flip augmentation to measure how much of the observed accuracy is attributable to that single design choice. The second replaces cosine annealing with a constant learning rate to confirm that the scheduler contributes beyond trivial convergence behaviour. Both ablations share the same seed set and evaluation protocol as the main sweep.

---

## 8. Inference on CAD Objects

The trained model is not confined to scatterer-synthesised data. Real objects designed in Tinkercad (or any CAD package) can be passed through an STL-to-heatmap pipeline that reuses the same MATLAB forward model:

```
STL file  →  scatterer extraction (vertex sampling)
          →  place at chosen range + orientation
          →  synthesise IF signal + FFT
          →  Range-Angle heatmap
          →  PyTorch CNN  →  predicted shape
```

This pipeline is the bridge to real-world inference. Because the forward model is the same one used to generate the training set, a CAD object of known ground-truth shape provides a clean end-to-end test: any misclassification is attributable either to scatterer sampling artefacts from the STL or to genuine model error, both of which are debuggable without a hardware setup.

The STL test set in the current repository covers Circle, Square, and Triangle (at 400 mm). Rectangle and Oval are straightforward to add by parameterising the existing STL generator.

---

## 9. Results

*This section will be populated once the full multi-seed antenna-count sweep completes. The figures and tables below indicate the intended structure.*

### 9.1 Overall Accuracy vs Antenna Count

Mean test accuracy (±1 std across 5 seeds) as a function of `N_ant`.

| `N_ant` | Angular Resolution | Mean Test Accuracy | Std Dev |
|--------:|-------------------:|-------------------:|--------:|
|       4 |              ~28.6° | TBD                | TBD     |
|       8 |              ~14.3° | TBD                | TBD     |
|      10 |              ~11.5° | TBD                | TBD     |
|      12 |               ~9.5° | TBD                | TBD     |
|      16 |               ~7.2° | TBD                | TBD     |
|      32 |               ~3.6° | TBD                | TBD     |

Companion figure: `results/accuracy_vs_antennas.png`.

### 9.2 Per-Class Accuracy by Antenna Count

Per-shape recall at each antenna count, averaged across seeds, to support the analysis in Section 10.

| Class     | `N=4` | `N=8` | `N=10` | `N=12` | `N=16` | `N=32` |
|-----------|------:|------:|-------:|-------:|-------:|-------:|
| Circle    | TBD   | TBD   | TBD    | TBD    | TBD    | TBD    |
| Square    | TBD   | TBD   | TBD    | TBD    | TBD    | TBD    |
| Rectangle | TBD   | TBD   | TBD    | TBD    | TBD    | TBD    |
| Triangle  | TBD   | TBD   | TBD    | TBD    | TBD    | TBD    |
| Oval      | TBD   | TBD   | TBD    | TBD    | TBD    | TBD    |

### 9.3 Confusion Matrices

One confusion matrix per `N_ant` (best seed). Companion figures: `results/confusion_N{4,8,10,12,16,32}.png`.

### 9.4 Preliminary Observation

A preliminary three-seed run at `N_ant = 16` (seeds 0, 1, 2) produced a test-accuracy mean of 68.6% with a sample standard deviation of 14.6 percentage points. Per-class recall at the best seed was Circle 100%, Square 93%, Oval 84%, Triangle 72%, Rectangle 45%. This observation motivates the analysis in Section 10 and will be re-reported from the full five-seed run once available.

---

## 10. Discussion

This section addresses two questions the experimental design is set up to answer: why the network finds some shapes easier to classify than others, and why per-shape classification strength varies with the number of receive antennas.

### 10.1 Why Shape-Level Difficulty Varies

The ordering observed in the preliminary data — Circle > Square > Oval > Triangle > Rectangle — is not accidental. It is predicted by the three-axis taxonomy introduced in Section 3.2 (symmetry order, angularity, aspect ratio), as follows.

**Circle is the easiest class.** Its scatterer signature is invariant to orientation: a circle rotated by 90° is indistinguishable from a circle at 0°. In feature-space terms, the orientation orbit of Circle collapses to a single point. The network therefore only has to learn one representative pattern, and any data augmentation or pose variation contributes useful gradient signal. Additionally, a uniform scatterer ring produces a distinctive low-amplitude, evenly-spread range-angle footprint that cannot be confused with any corner-bearing class.

**Square is nearly as easy.** Its 4-fold rotational symmetry means its orientation orbit is only 90° long — effectively four repeats of the same signature in 360°. This shrinks the orientation-invariance the network must learn by a factor of four. Additionally, a square at 45° still looks "square-ish" because the four corner reflectors remain visible, just at rotated positions. Squares are confusable with small rectangles (aspect ratio near 1) but otherwise occupy a distinct region of feature space.

**Oval is moderately difficult.** It is a smooth elongated shape with 2-fold symmetry. It is distinguishable from Circle by its elongation (non-circular range-angle footprint) and from the angular classes (Square, Rectangle, Triangle) by the absence of corner reflectors. But its signature does depend on orientation: an oval at 0° (major axis parallel to the radar boresight) looks like a near-point target, whereas an oval at 90° spreads across many angle bins. The network must learn both endpoints of that continuum, which is why Oval plateaus below Circle and Square.

**Triangle is harder.** A triangle has three strong corner reflectors, which should make it easy in principle. The difficulty is orientation sensitivity: a triangle at some orientations presents three clearly-separated corners, while at others only two corners dominate the radar return (because the third is occluded or far from the radar), producing a two-corner signature that can look like the dominant corners of a rectangle. Triangles also have 3-fold rotational symmetry, which is an unusual orbit size compared to the other classes (2 or 4) and may be harder for the convolutional features to encode cleanly.

**Rectangle is the hardest class.** This is the main shape-level research finding. The problem with rectangles is that they are the most *acoustically ambiguous* shape in the set:

- A rectangle with aspect ratio close to 1 is almost indistinguishable from a Square.
- A rectangle viewed end-on (along its long axis) produces a near-point signature that looks like a small Circle or a short Oval.
- A rectangle viewed broadside produces an elongated angular footprint that can look like a long Oval, but with corner reflectors that could also be read as the dominant corners of a Triangle.
- Rectangle has the weakest rotational symmetry (2-fold), meaning its 180° orientation orbit includes pose configurations that span the largest region of feature space of any class.

In other words, every other class has at least one feature that disambiguates it cleanly from its neighbours: Circle has no corners, Square has four corners at equal spacing, Oval has smooth elongation, Triangle has three corners. Rectangle shares at least one of these features with each of the other four classes, depending on orientation and aspect ratio, and the current training set does not have enough redundancy along any single axis to let the model separate it out cleanly.

### 10.2 Why Per-Shape Strength Varies With Antenna Count

The angular resolution `θ_res ≈ 2/N` radians is the knob that the sweep controls. Its impact on the classifier is not uniform across classes, because each class depends on different spatial features of the heatmap.

**Circle is nearly insensitive to `N_ant`.** Its discriminative feature is that the range-angle energy is distributed uniformly around a centre, with no strong localised peaks. This "smooth everywhere" pattern remains recognisable even at `N = 4` (28° resolution), because it does not require resolving any specific corner. Circle accuracy is expected to be near-ceiling across every antenna count.

**Square and Triangle improve with `N_ant` up to a saturation point.** Their signatures depend on resolving individual corner reflectors. At `N = 4`, corner positions smear across many degrees and two adjacent corners blur into a single blob. At `N = 8–12`, corners become separately resolvable at typical ranges, and classification improves rapidly. Above `N ≈ 16`, further resolution gains do not help: the corners are already cleanly separated, and additional angular bandwidth resolves sub-scatterer fine structure that the training set does not encode. We therefore expect Square and Triangle to rise sharply from `N = 4` to `N = 12–16` and then plateau.

**Oval benefits most from high `N_ant`.** Its signature is the full angular extent of a smooth curve, which is fundamentally an angular-resolution question: how accurately can the radar trace out the continuous curve of a rotated oval? At low `N_ant` the curve is under-sampled and looks like a short, round blob; at high `N_ant` the elongation is explicit. Oval's per-class accuracy is therefore expected to rise monotonically across the full sweep, with less early saturation than Square or Triangle.

**Rectangle does not cleanly improve with `N_ant`.** This is the counterintuitive prediction. More antennas help Rectangle's corner-resolution story (in the same way as Square), but they do *not* help the orientation- and aspect-ambiguity problem described in Section 10.1. A rectangle that looks like a square at a particular orientation and aspect ratio will still look like a square at `N = 32` — with sharper edges. The fundamental confusion is geometric, not resolution-bound, so we expect Rectangle accuracy to flatten or even drift down at high `N_ant` as the model increasingly over-fits to resolution-specific features that happen to overlap between Rectangle and its neighbour classes.

**Aggregate accuracy therefore saturates.** Because the easy classes (Circle) are already saturated and the hard class (Rectangle) does not respond to `N_ant`, the overall accuracy-vs-`N_ant` curve is expected to rise briskly at low `N_ant` and then flatten somewhere around `N ≈ 12–16`. This has a direct practical implication: the marginal benefit of doubling antenna count from 16 to 32 is probably small, which matters for cost and complexity if the system is ever built in hardware.

### 10.3 Limitations of the Current Methodology

Four limitations of the current setup are worth flagging for the next phase:

1. **Random data splits may leak orientation.** The 80/10/10 split is by individual sample, so for a given (size, range, shape) combination, the training set may contain the same object at nearby orientations to the test examples. This makes the task closer to "recognise this specific orbit point" than to "generalise across orientation." An orientation-stratified split (holding out a contiguous wedge of orientations for test) is the correct next step and will almost certainly reduce headline accuracy.
2. **Aspect ratio is lumped into "size."** The three size values vary overall scale but not aspect ratio. A rectangle with aspect 1.0 is a square; a rectangle with aspect 3.0 is an elongated bar. These corner cases are exactly where the class-level confusion happens, and the current dataset does not stratify them.
3. **SNR is fixed.** All synthetic data is generated at a single SNR. Real radar varies substantially across range and material, and a classifier that depends on a specific noise floor will not transfer.
4. **No real-hardware validation.** The CAD/STL pipeline lets the model run on simulated CAD objects, but the forward model is identical between train and test, so success there does not close the sim-to-real gap. A separate validation on a physical FMCW radar (e.g. TI IWR1642) remains the definitive test.

---

## 11. Summary and Next Steps

### 11.1 Work Completed

| Component                         | Location                                     | Status    |
|-----------------------------------|----------------------------------------------|-----------|
| Synthetic data generator          | `src/matlab/data_generation/*.m`             | Complete  |
| STL-to-heatmap inference pipeline | `src/matlab/inference/*.m`                   | Complete  |
| Training script (single antenna)  | `src/python/training/radar_obj_classifier_cnn_single_antenna.py`   | Complete  |
| Training script (antenna sweep)   | `src/python/training/radar_obj_classifier_cnn_multiple_antenna.py` | Complete  |
| Multi-seed evaluation protocol    | Integrated into both training scripts        | Complete  |
| Antenna-count sweep results       | `results/`                                   | **Pending** (full sweep running) |
| Orientation-stratified split      | —                                            | Planned   |

### 11.2 Immediate Next Steps

1. **Run the full five-seed sweep** across all six `N_ant` values and populate Section 9 with the resulting tables and figures.
2. **Re-split the data along orientation** (hold out four orientations for test, four for validation) and re-run the sweep under the stratified protocol. Compare headline accuracies to quantify the orientation leakage in the current split.
3. **Per-orientation accuracy heatmaps.** For each shape, plot recall as a function of true orientation (36 bins). This will either confirm or refute the Rectangle-orientation-ambiguity hypothesis in Section 10.1.
4. **Rectangle-specific stratification.** Add aspect ratio as an explicit generator axis (e.g. 1.2, 1.5, 2.0, 3.0) and retrain. Expect accuracy to improve on elongated rectangles and drop on near-square rectangles, which would confirm the diagnosis.
5. **CAD inference tests.** Run the trained model on the STL-derived heatmaps already in `data/heatmaps/` as a first sim-to-real check.

### 11.3 Open Questions for the Next Advisor Meeting

- Does the advisor prefer orientation stratification or a larger dataset with random splits for the final writeup? The former is more honest; the latter is easier to report as a single number.
- Is access to a real 77-GHz FMCW radar feasible this semester, or should sim-to-real validation be left as an explicit future-work item?
- Would the advisor like the report to include a brief SVM baseline run (for comparison), or is the conceptual argument in Section 4 sufficient?

---

*Report last updated: Spring 2026. All code, data, and figures referenced herein are in the project repository.*
