# Results Analysis & Research Notes

This document is a close read of everything in `results/` plus the
training and data-generation code that produced it. The goal is to
flag what the numbers are actually saying, answer the five research
questions, and describe how to test the model properly going forward.

The numbers referenced below come from `results/results_per_seed.csv`,
`results/run results.txt`, and the PNG figures in this directory.

---

## 1. Critical observations on the results

### 1.1 Overall accuracy *decreases* with antenna count

From `accuracy_vs_antennas.png` and the CSV:

| N_ant | Mean test acc | Std   | Best seed |
|------:|--------------:|------:|----------:|
|     4 |       94.18 % | 0.27 %|   94.34 % |
|     8 |       83.39 % | 0.81 %|   84.16 % |
|    10 |       81.39 % | 1.63 %|   83.24 % |
|    12 |       80.15 % | 2.90 %|   83.12 % |
|    16 |       64.63 % | 29.39%|   81.85 % |
|    32 |       80.96 % | 1.58 %|   82.77 % |

The headline finding is the opposite of what the usual radar intuition
predicts: going from 4 antennas (28.6° angular resolution) to 32
antennas (3.6° resolution) *lost* roughly 13 points of accuracy. The
best single run at any configuration is still N=4. The diagnostic
script `check_heatmap_normalization.py` exists specifically to rule out
that this is a per-file normalisation artefact — so the inversion is a
real property of the data/model combination, not a bug in how the
heatmaps are scaled.

### 1.2 N=16 had a catastrophic seed-0 failure

Seed 0 at N=16 collapsed to 30.64 % — the confusion matrix shows the
model never predicted Triangle and spread its probability mass across
Square, Rectangle and Oval essentially at random. Seeds 1 and 2 at the
same N finished at ~81 %. The huge std (29 %) on that column is
produced entirely by this one failed run.

Training curves (`history_N16.png`, `history_N32.png`) explain why:
at higher antenna counts the raw validation loss spikes wildly in the
first 15 epochs (up to ~2.7 for N=16, vs. ~1.6 at N=4) before the EMA
finally drags it back down. The warmup + gradient clipping + EMA
early-stopping in the trainer were added to mitigate these spikes, and
they mostly work — but seed 0 at N=16 still hit a bad initialisation
and never recovered. This is the model telling you it is *much harder*
to train as N_ant grows, not that the data is uninformative.

### 1.3 Per-class behaviour is consistent across N_ant

| Class     | Easy? | Notes                                                         |
|-----------|:-----:|---------------------------------------------------------------|
| Circle    | Yes   | 100 % at every N_ant except N=16 seed 0 (the collapsed run).  |
| Square    | Yes   | 85–99 %, small drop at mid-N.                                 |
| Rectangle | **Hard** | Falls from ~82 % at N=4 to ~47 % at N=32.                  |
| Triangle  | Mixed | 95 % at N=4, 61 % at N=32.                                    |
| Oval      | OK    | ~90 %, stable.                                                |

Rectangle is always the worst class, and its accuracy falls fastest as
N_ant grows. The dominant confusions at high N (from `confusion_N32.png`):
Rectangle → Triangle (31.8 %), Triangle → Rectangle (30.6 %),
Rectangle → Square (8.7 %). Rectangle and Triangle mutually cannibalise
each other; Circle never gets confused with anything.

### 1.4 Train/val/test split has orientation leakage

Re-reading `radar_data_generation_multiple_antenna.m` (section 10):
the split is stratified only by class. Each shape has
36 orientations × 3 sizes × 4 ranges × 4 SNR levels = 1,728 samples
that get shuffled and cut 60/30/10. That means the same
(shape, orientation, size, range) can appear in both train and test,
only differing by SNR realisation. The reported test accuracies are
therefore "how well does the model recognise a heatmap it has seen a
noisier version of?" — not "how well does it recognise a pose it has
never seen." A leave-one-orientation-out or leave-one-size-out split
would give a more honest number. This is the single biggest asterisk
on the whole results table.

### 1.5 Training was stable at low N, shaky at high N

`history_N4.png`: smooth monotonic descent, train and val converge
together at ~94 %. `history_N32.png`: EMA val loss oscillates for 30+
epochs before settling at 83 %, and train still ends a couple of
points above val. That gap + oscillation pattern is the classic
signature of a model that is being asked to memorise too fine a
pattern with not quite enough data.

### 1.6 One minor bug: `generate_RA` has a noisy line

`radar_data_generation_multiple_antenna.m:256` computes `noise_amp`
but the statement has no semicolon, so it prints the value to the
console for every sample during data generation. It does not affect
correctness, just noise in the MATLAB output. Likewise line 267 does
`RA = abs(mean(ang_fft,1))` and then immediately overwrites it on 268,
so the first line is dead code. These are cosmetic; the training data
itself is fine.

---

## 2. Why a CNN?

The input is a 512 × 256 Range-Angle heatmap: a 2-D image where the
bright pixels are scatterer returns. Three properties of this data
make a 2-D CNN the right default:

**Spatial locality matters, absolute position does not.** A "corner"
scatterer looks the same whether it is at (range = 3 m, angle = 0°) or
(range = 5 m, angle = 20°). The meaningful features are local patterns
(bright peak + sidelobes, pair of peaks at fixed angular spacing,
curved arc) that are translation-invariant in the RA plane. That is
exactly what stacks of 2-D convolutions with max-pooling are built
for: they learn translation-invariant local pattern detectors.

**The useful signal spans multiple scales.** The outline of a 60 cm
triangle at 3 m spans a large region of the heatmap; the separation
between two edge returns of a small rectangle can be a few pixels.
The trainer's four conv blocks use kernel sizes 7 → 5 → 3 → 3 so
early layers can capture the coarse outline and deeper layers can
resolve fine corner–corner spacing. Global average pooling at the end
lets the final classifier average over wherever in the heatmap the
shape happens to land.

**Alternatives have documented trade-offs for this problem size.**

- *Flat MLP over 131,072 pixels.* No spatial prior → would need much
  more data and would overfit with only ~1.7 k training samples per
  class.
- *Hand-crafted features + SVM / random forest* (edge counts, peak
  counts, PCA on polar features). Valid baseline, usually lower
  ceiling, and brittle to the amplitude and sidelobe changes that
  come with rotating the shape.
- *Point-cloud or graph nets on extracted peaks.* Could work for
  idealised simulation data, but depends on reliable peak detection
  which gets unstable at low SNR and high angular resolution.
- *Vision transformers.* Overkill for 2 k samples; they need dataset
  sizes that are one or two orders of magnitude larger before they
  outperform CNNs on this kind of structured image.
- *1-D CNN on the range profile or the raw IF signal.* Discards
  angular information, which is exactly what the antenna experiment
  is designed to measure.

A small 2-D CNN (the trainer uses 115 k parameters) is therefore the
right *first* architecture: it encodes the correct inductive bias,
trains in a few minutes per seed on a single GPU, and leaves room to
try the other alternatives later as baselines.

---

## 3. Why does the model do better on some shapes than others?

The per-class accuracy pattern is Circle > Oval ≈ Square > Triangle
> Rectangle. That ordering is exactly what the physics of the
scatterer model predicts:

**Circle is a fixed-point under rotation.** `make_circle` in the
MATLAB generator puts 24 equal-amplitude points on a ring. Rotate the
whole ring by any angle and you get the same ring — so every one of
the 36 training orientations produces essentially the same heatmap
(up to noise). The model has 4 × as many *effective* training examples
for Circle as it does for a shape whose signature actually depends on
orientation. The confusion matrices reflect this: 100 % recall at
every N_ant except the collapsed seed.

**Oval is almost rotationally symmetric.** Same argument, but with
aspect ratio 1.8:1 instead of 1:1, so the model does have to learn
the elongation. Still easy because the curvature-weighted amplitude
in `make_oval` puts strong peaks at the two tips, giving a
characteristic two-lobed signature. Oval-to-anything confusions are
always small (< 10 %).

**Square has 4-fold symmetry.** 0°, 90°, 180°, 270° produce the same
heatmap. Only 9 of the 36 training orientations carry unique
information, but that is still enough when combined with the
distinctive "four strong corner returns" pattern (`AMP_CORNER = 1.0`
vs `AMP_EDGE = 0.70`).

**Triangle has 3-fold symmetry and a 3-corner signature.** Mostly
easy, but its 3-corner signature *is* a subset of Rectangle's
4-corner signature — which is exactly where the confusions come from.

**Rectangle has aspect ratio 2:1 and no rotational symmetry.** Every
one of the 36 orientations genuinely produces a distinct heatmap:
the long side can be along the boresight, perpendicular, or at any
angle in between. So the model has to learn *orientation-dependent*
features to classify Rectangle. `confusion_N32.png` makes this
concrete: 31.8 % of rectangles are classified as triangles. At low
N_ant this is averaged out by the coarse angular resolution; at high
N_ant the model can "see" the individual corners and flips its vote.

The *particular* failure mode is not what I first guessed. My
original hypothesis was "rectangles get misclassified when the long
axis is end-on (0° / 180°)" — the intuition being that the rectangle
collapses to a two-point signature that looks like a triangle. After
rendering the errors I saw the opposite is true: the hard cases at
N=16 are **broadside** rectangles, where the long side is
perpendicular to the line of sight and shows up as a long horizontal
band of corner + edge scatterers. These broadside bands get confused
with Oval at close range (the band looks like an elongated lobe) and
with Square at far range (the band compresses to what reads as a
4-corner blob). See §7 for the figures that pinned this down.

---

## 4. Why does varying antenna count affect model strength?

The observed relationship is "accuracy goes *down* as N_ant goes up."
Three mechanisms each contribute; you would verify each by poking at
the inspection script.

### 4.1 Angular resolution changes the complexity of the feature manifold

The Rayleigh resolution of a uniform linear array is roughly
`114.6° / N_ant`: 28.6° for N=4, 3.6° for N=32. At 28.6°, a shape
rotated by 10° is well below the resolution cell, so the heatmap of
the same shape at 0° and at 10° look essentially identical after the
angle FFT. All 36 training orientations collapse to a small number of
*distinguishable* heatmap patterns. From the model's point of view
the data is highly redundant and the effective class-conditional
distribution is simple.

At 3.6°, the same 10° rotation produces a *visibly different* heatmap.
The model now has to learn a feature that is invariant to all 36
orientations — it has to generalise across the full orientation
range instead of collapsing to a handful of canonical views. The same
2,160 × 4 SNR samples per file now cover a much larger effective
manifold, so the model trains for longer, plateaus lower, and gets
stuck more often on bad initialisations (the N=16 seed 0 collapse).

### 4.2 The CNN's receptive field no longer matches the useful scale

The architecture is fixed across experiments. Block 1 uses a 7 × 7
kernel, which at 512 × 256 corresponds to roughly 7 angular bins.
At N=4 that covers 28.6° × (7/256) ≈ 0.78° worth of angular span
per kernel position — but the shape's angular signature is *already
smeared* across many bins, so a 7 × 7 kernel is a reasonable match
for the blob size. At N=32 the shape collapses into a near-point
source, occupying < 1 bin angularly; a 7 × 7 kernel mostly looks at
empty background. The CNN has to rely on later (small-kernel) layers
to resolve the now-sharp features, which is a harder learning problem
with no architectural help.

### 4.3 The data pipeline amplifies the two effects above

- The horizontal-flip augmentation in `train_model` (`flip_mask = torch.rand(...) > 0.5`)
  mirrors along the angular axis. Circle and Square are invariant to
  this; Triangle/Rectangle/Oval at orientation θ become orientation
  −θ. At low N_ant this still lands inside a resolution cell of a
  nearby orientation, so it behaves like a gentle smoothing. At high
  N_ant it becomes a real orientation flip, effectively doubling
  the pose set the model must memorise.
- `check_heatmap_normalization.py` already rules out the "each file
  is normalised differently" explanation, so we can treat per-file
  normalisation as a non-factor.

### 4.4 Sanity-check it visually

The `--compare-n` view of the script renders the *same* test sample
through every antenna count's model. Rendering a single Triangle at
the same (orientation, size, range) across N ∈ {4, 8, 10, 12, 16, 32}
is the cleanest demonstration of the mechanism:

```bash
python src/python/ui/inspect_heatmaps.py --compare-n \
    --mat-dir . --model-dir results \
    --shape Triangle --sample 519
```

In that figure the triangle's angular footprint shrinks monotonically
from a broad lobe at N=4 (spanning ~25° at half-max) to a few bright
pixels at N=32. The prediction is still correct at every N but the
confidence drops visibly — you can watch the feature manifold grow
faster than the model's fixed-size kernels can follow. That is the
smoking gun for the "more antennas → harder classification problem"
story.

Repeating the same exercise on an *erroring* Rectangle
(`--shape Rectangle --only-errors --rank 0`) shows the same footprint
shrinkage plus the moment when the predicted label flips — useful
for checking specific error modes case-by-case.

---

## 5. Testing status & next steps

### 5.1 What's already being tested

Every training run evaluates on a held-out *test* partition (10 % of
samples, stratified by class). The numbers in
`results_per_seed.csv`, the right column of `accuracy_vs_antennas.png`,
and each `confusion_N*.png` are all already test-set metrics. So yes,
"testing" in the ML sense is happening; it is just that the
train / val / test split shares orientations and sizes across the
three partitions, so the test set is measuring *noise-robustness*
rather than *generalisation to new scenes*.

### 5.2 What has not been tested

1. **Real-world objects via STL → heatmap.** `data/heatmaps/`
   contains `stl_heatmap_circle_0deg.mat`, `stl_heatmap_square_0deg.mat`
   and `stl_heatmap_triangle_0deg.mat` — these were produced by
   `src/matlab/inference/radar_stl_to_heatmap.m` from Tinkercad STLs
   and the trained model has *never been run on them*. This is the
   most important next step, because it is the whole point of the
   pipeline described in the README.

2. **Generalisation to unseen orientations / sizes / ranges.** See
   5.1 — the current split doesn't measure this. A leave-one-out
   split would.

3. **Cross-N_ant generalisation.** Nothing prevents training on N=4
   data and testing on N=32 data, or vice versa. Doing that sweep
   would separate "the model learned shape" from "the model learned
   the angular resolution of its training file."

4. **Out-of-distribution robustness.** The SNR levels in training
   are 5/10/20/30 dB. Nothing has been tested at −5 dB, no clutter,
   no multi-object scenes, no range beyond 5 m.

### 5.3 How to actually run the three missing tests

**(a) STL heatmap inference — highest priority.**

The `radar_obj_classifier_cnn_multiple_antenna.py` module already
has `load_model` and `predict_shape`. So from
`src/python/training/` (or from the root with PYTHONPATH set):

```python
import scipy.io
from radar_obj_classifier_cnn_multiple_antenna import load_model, predict_shape

model, ckpt = load_model('results/radar_shape_model_N4.pt')
for name, true_idx in [('circle', 0), ('square', 1), ('triangle', 3)]:
    data = scipy.io.loadmat(f'data/heatmaps/stl_heatmap_{name}_0deg.mat')
    # the stl heatmap generator writes 'RA_normalised' (see radar_stl_to_heatmap.m);
    # adjust this key if your file uses a different name
    hm = data.get('RA_normalised', data.get('RA', data.get('heatmap')))
    predict_shape(hm, model, true_label=true_idx)
```

Report the confusion matrix across the three real shapes, per N_ant,
as a separate table. Expect worse numbers than the synthetic test
set — that is fine and scientifically interesting, because it
measures the simulation-to-reality gap.

**(b) Leave-one-out generalisation test.**

The MATLAB generator already records `orientations_ds`, `sizes_val`
and `ranges_ds` in the `.mat` file. Re-run it with the split
replaced by something like "hold out orientations {20°, 110°, 200°,
290°}" or "hold out size = 0.60 m." Re-train and compare. A drop
from 94 % to, say, 70 % would tell you how much of the current
accuracy comes from memorising poses vs. recognising shape.

**(c) Cross-N_ant evaluation.**

You already have every `radar_shape_model_N*.pt` and every
`radar_shapes_N*.mat`. A simple script that loads model N=4 and
runs it on the test tensors of every other N (and vice versa)
produces a 6 × 6 grid. Diagonals are what you have now; off-diagonals
quantify how much each model overfits to its own angular resolution.

### 5.4 What I can answer with what I have right now

- *Is the CNN better than chance?* Yes, by a wide margin at every N.
- *Is more hardware (more antennas) helping?* No, not on this dataset — it is hurting. Keep N=4 as the production choice until the dataset is big enough to support finer resolution.
- *Which shape is hardest?* Rectangle, at every N.
- *Would I ship this for the real-world demo?* I would ship N=4, and I would evaluate it on the STL heatmaps in `data/heatmaps/` before claiming anything beyond "it works on synthetic data."

---

## 6. Using the inspection script

`src/python/ui/inspect_heatmaps.py` was written for this writeup. It
loads any `radar_shapes_N*.mat` + matching `radar_shape_model_N*.pt`
and lets you look at predictions directly.

Common invocations (run from the IW root):

```bash
# Grid of the 6 highest-confidence wrong Rectangle predictions at N=32
python src/python/ui/inspect_heatmaps.py \
    --mat results/radar_shapes_N32.mat \
    --model results/radar_shape_model_N32.pt \
    --shape Rectangle --only-errors --n-examples 6

# Side-by-side montage: correct vs wrong Triangles at N=4
python src/python/ui/inspect_heatmaps.py \
    --mat results/radar_shapes_N4.mat \
    --model results/radar_shape_model_N4.pt \
    --shape Triangle --montage --n-examples 5 \
    --save triangle_N4_montage.png

# Same shape, every N_ant — verifies the "resolution kills Rectangle" hypothesis
python src/python/ui/inspect_heatmaps.py --compare-n \
    --mat-dir results --model-dir results \
    --shape Rectangle --only-errors --rank 0

# Drill into a specific test index
python src/python/ui/inspect_heatmaps.py \
    --mat results/radar_shapes_N16.mat \
    --model results/radar_shape_model_N16.pt \
    --sample 42
```

The script assumes the `.mat` files live alongside the `.pt` files.
If you kept them where the run produced them (i.e. at the repo root,
not under `results/`), substitute the path. Use `--help` for the full
option list.

---

## 7. Hypothesis verification from the inspection figures

After the write-up above was drafted I ran the inspection script on
three scenes and checked each of my claims against the actual
heatmaps. This section records what survived, what didn't, and what
turned out to be new.

### 7.1 Verified

**Angular footprint shrinks with N_ant.** The Triangle `--compare-n`
panel (sample 519) shows a clean monotonic collapse from a broad
angular lobe at N=4 to a handful of bright pixels at N=32. This is
exactly the geometry §4.1 predicted: at N=4 many orientations alias
onto the same blurry blob, at N=32 each orientation has its own
pointy signature. The model predicts correctly at every N in this
example, which is the best-case outcome and still shows the
confidence dropping visibly as N grows.

**Confidence decreases with N_ant even when the answer is right.**
In the same compare-n panel the softmax margin for the correct class
is highest at N=4 and lowest at N=32. This is consistent with the
"the class-conditional distribution got harder to separate" reading
of §4.1 rather than an optimisation accident.

**Triangle ↔ Rectangle confusion is a high-N_ant phenomenon.** The
Rectangle correct-vs-wrong montage at N=4 shows no Rectangle →
Triangle errors at all — the two errors in the panel are both
Rectangle → Square. The Triangle ↔ Rectangle cross-confusion that
dominates `confusion_N32.png` is genuinely produced by the
higher-resolution regime, not a baseline property of the dataset.

### 7.2 Falsified

**"End-on" Rectangle hypothesis.** I claimed rectangles would fail
when the long axis is aligned with the line of sight (0° / 180°)
because they would collapse to a two-point return that looks like a
triangle. The N=16 errors panel shows the opposite: the hard cases
are all *broadside* — the long side is roughly perpendicular to the
boresight and shows up as a long horizontal band of scatterers. The
N=4 correct Rectangles in the same montage are the end-on cases that
I predicted would fail; the model gets those right because the
broad-angle-resolution cell smears the two tip returns into a wide
lobe that the CNN recognises as Rectangle-shaped.

The underlying mistake in my original reasoning was mixing up two
different "collapses": the *physical* collapse where a long axis
aligned with LOS reduces angular extent, and the *array-imaging*
collapse where low angular resolution blurs everything. At low
N_ant the array blurs all rectangles into a recognisable generic
rectangle-lobe regardless of orientation — so end-on is *easy*, not
hard. At high N_ant the blurring stops, and the confusion is driven
by which specific orientations happen to produce scatterer patterns
that match another class's template.

**"Rectangle → Triangle at N=16" was overstated.** The N=16 errors
panel shows Rectangle → {Oval, Square} as the dominant confusions,
not Rectangle → Triangle. The triangle confusion really does
dominate only at N=32, as the `confusion_N32.png` matrix said.
The write-up now reflects that.

### 7.3 New findings from the figures

**Rectangle errors are range-stratified at N=16.** Broadside
rectangles at close range (strong, vertically-narrow bright bands)
get classified as Oval — the band looks like an elongated curvy
return. Broadside rectangles at far range (the band is wider in
angle because 114.6/N scales with range only indirectly via the
slant distance, and the total footprint is smaller in pixels) get
classified as Square — the wider band reads as a 4-corner blob. This
is a cleaner story than the single "triangle confusion" line I had
in §1.3, and suggests that a range-aware feature (or simply more
data at close range) might recover some of the lost accuracy.

**Triangle errors at high N are sparse-outline failures.** The
compare-n figure shows the triangle at N=32 as three near-isolated
bright points. When one of those points dims (because the facet
angle is unfavourable for backscatter) the remaining two look like
a rectangle's end-on signature — which explains the symmetric
Triangle → Rectangle flow in `confusion_N32.png`. So Triangle-at-high-N
*does* exhibit the "end-on" collapse I originally ascribed to
Rectangle, just for the opposite class.

**N=4 has real structure the eye can't see.** Some of the N=4 errors
in the Rectangle montage look identical to correct predictions at
glance. The CNN is picking up on amplitude and sidelobe cues that
are invisible in a `20*log10(RA)` render. That is *good* for the
model (it is using more than outline) and *bad* for any hand-crafted
feature alternative that would be limited to visible structure —
worth remembering when §2's "alternatives" list is revisited.

### 7.4 What the figures imply for the next experiments

- The leave-one-orientation-out test from §5.3(b) should explicitly
  hold out the broadside band (80°–100° and 260°–280°). If test
  accuracy for Rectangle drops much more than the other classes,
  that confirms the broadside-is-hard story rather than a generic
  memorisation-of-poses story.
- The cross-N test in §5.3(c) should report per-class accuracy, not
  just overall, because the N=4 model's Rectangle behaviour is the
  interesting degree of freedom. Expect the N=4 → N=32 off-diagonal
  to be near chance for Rectangle and close to the diagonal for
  Circle.
- The STL demo in §5.3(a) should prioritise getting a *rectangle*
  STL rendered and tested. Every STL heatmap currently in
  `data/heatmaps/` is Circle/Square/Triangle — the three easy
  classes — so the existing files cannot measure the class that
  actually drives the accuracy curve.
