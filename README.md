# Diabetic Retinopathy Screening

MATLAB R2026a competition pipeline for five-grade fundus classification:
`Mild`, `Moderate`, `No_DR`, `Proliferate_DR`, and `Severe`.

Referable DR is `Moderate`, `Severe`, or `Proliferate_DR`. The retained
operating thresholds were selected for maximum validation sensitivity subject
to specificity >= 85%.

## Repository layout

```
.
├── src/           E1, E3 Retina Walker, preprocessing, training, and evaluation
├── scripts/       retained E1 evaluation entry point
├── tests/         E1/E3 synthetic and bounded real-image checks
├── artifacts/     retained E1/E3 calibrated thresholds; generated output root
├── data/          canonical split metadata and high-resolution fundus images
├── models/        E1/E3 checkpoints and required E1 initializer
├── evaluate_e3_split.m
└── run_V3.m       E1/E3 training dispatcher
```

## Required local files

These large inputs are intentionally excluded from Git:

- `data/DR_V4_RESNET101_SCREENING.mat` containing the fixed
  `imdsTrain`, `imdsValidation`, `imdsTest`, and `classNames` variables.
- `data/downloads/DR1/dr_unified_v2/dr_unified_v2/` containing the canonical
  high-resolution JPG source. The saved split is remapped to this tree by basename.
- `models/AG_V4_5_ReducedAug.mat` containing the required E1 initialization network.
- `models/V3_E1_HighResFOV448.mat` containing final E1.
- `models/V3_E3_RetinaWalker_best.mat` containing final E3.
- `models/V3_E3_RetinaWalker_latest.mat` containing the E3 resume state.

All code resolves paths from the repository root; there are no machine-specific
absolute paths.

## Verification

Requirements: MATLAB R2026a, Deep Learning Toolbox, Statistics and Machine
Learning Toolbox, and a supported ResNet-101 installation/checkpoint.

Run the synthetic unit and Retina Walker checks:

```matlab
run('tests/run_tests.m')
run('tests/test_e3_components.m')
run('tests/test_e3_explicit_split_metrics.m')
```

Run the bounded real-image E3 pipeline smoke test:

```matlab
run('tests/test_e3_pipeline_smoke.m')
```

## Training and evaluation

```matlab
metricsE1 = run_V3("E1");
metricsE3 = run_V3("E3");

run('scripts/evaluate_E1.m')
metrics = evaluate_e3_split('data/downloads/DR1/dr_unified_v2/dr_unified_v2/test');
```

E1 is global 448x448 ResNet-101 classification. E3 starts from E1 and adds
vessel-guided eight-patch retinal inspection plus global/local fusion. Existing
code supports E1/E3 training, checkpoint loading, and inference primitives. A
single-image frontend API is intentionally not introduced by this cleanup.

Validation-locked operating points are retained at:

- `artifacts/results/V3/V3_E1_HighResFOV448/selected_threshold.txt` (`0.164`)
- `artifacts/results/V3/V3_E3_RetinaWalker/selected_threshold.txt` (`0.061`)

To retrain on added labels, regenerate `data/DR_V4_RESNET101_SCREENING.mat`
with the authoritative split and keep matching originals under the high-resolution
source tree. E1 retrains from `AG_V4_5_ReducedAug.mat`. E3 resumes only when its
saved `latest` checkpoint matches the exact configuration and ordered split;
move old E3 resume checkpoints aside before starting a genuinely new dataset run.
