# Diabetic Retinopathy Screening

MATLAB R2026a research pipeline for five-grade fundus classification:
`Mild`, `Moderate`, `No_DR`, `Proliferate_DR`, and `Severe`.

The clinical operating target is maximum referable-DR sensitivity
(`Moderate` or worse) subject to specificity >= 85%. This is research code,
not a clinically validated medical device.

## Repository layout

```
.
├── src/           reusable evaluation, loss, analysis, and logging functions
├── experiments/   historical training experiments (never used by tests)
├── scripts/       evaluation and diagnostic entry points
├── tests/         synthetic checks; no retinal images
├── artifacts/     tracked metrics, figures, logs, and historical baselines
├── docs/          research report
├── data/          ignored local datasets and datastore split file
└── models/        ignored local checkpoints
```

## Required local files

These large/private inputs are intentionally excluded from Git:

- `data/DR_V4_RESNET101_SCREENING.mat` containing the fixed
  `imdsTrain`, `imdsValidation`, `imdsTest`, and `classNames` variables.
- `models/V4_1_base.mat` for historical comparisons.
- `models/AG_V4_5_ReducedAug.mat` containing `netTrained` for frozen-model
  evaluation.
- `artifacts/baselines/DR_V41_FINAL_TEST_RESULTS.mat` is optional and only
  enables the V4.1 full-test comparison.

All code resolves paths from the repository root; there are no machine-specific
absolute paths.

## Verification

Requirements: MATLAB R2026a, Deep Learning Toolbox, Statistics and Machine
Learning Toolbox, and a supported ResNet-101 installation/checkpoint.

Run the synthetic unit checks (no model and no images), then verify compatibility
against exactly five saved test predictions (one per class; no images):

```matlab
run('tests/run_tests.m')
run('tests/run_saved_subset_test.m')
```

Run a deterministic, stratified 10-image test-set smoke check:

```matlab
addpath('scripts')
metrics = evaluate_final_test(10);
```

The smoke path never trains and never writes official result artifacts.
`evaluate_final_test(Inf)` is deliberately explicit because it evaluates the
entire test set and can overwrite the official final-result artifact; do not run
it for tuning.

## V3 clinical-boundary sprint

V3 keeps the current repository layout (no new `ANTIGRAVITY*` tree). Generated
audits/results remain under `artifacts/`, checkpoints under ignored `models/`,
and the report is `docs/FINAL_V3_REPORT.md`.

```matlab
addpath('scripts'); audit_v3_images
addpath('experiments')
run_V3("E0"); run_V3("E1"); run_V3("E2"); run_V3("E3"); run_V3("E4")
```

E1-E3 require the fixed-split datastore and champion checkpoint listed above.
The high-resolution source is expected under
`data/downloads/DR1/dr_unified_v2/dr_unified_v2/`; V3 remaps the authoritative
saved split by basename and never trusts the download's separate split folders.

## Research result

The frozen `AG_V4_5_ReducedAug` model at validation-locked threshold `0.1199`
reported test ROC-AUC `0.8564`, PR-AUC `0.6914`, sensitivity `69.84%`, and
specificity `85.18%`. See `docs/RESEARCH_REPORT.md` and `artifacts/` for the
preserved experiment record.
