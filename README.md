# Diabetic Retinopathy Screening & Classification System

An automated 5-class diabetic retinopathy (DR) screening and classification pipeline developed in **MATLAB R2026a** with Deep Learning Toolbox and GPU acceleration.

---

## 1. Project Goal & Clinical Screening Definition

The objective is to classify fundus images into 5 DR stages with a focus on clinical screening:
1. **Mild** (Non-referable)
2. **Moderate** (Referable)
3. **No_DR** (Non-referable)
4. **Proliferate_DR** (Referable)
5. **Severe** (Referable)

### Clinical Decision Rule
* **Referable DR**: $\text{Moderate} \lor \text{Proliferate\_DR} \lor \text{Severe}$
* **Non-Referable**: $\text{Mild} \lor \text{No\_DR}$

The primary clinical objective is **maximizing Referable DR Sensitivity** while maintaining **Specificity $\ge 85.0\%$**.

---

## 2. Dataset & Stratified Splits

* **Dataset:** Kaggle `sovitrath/diabetic-retinopathy-2015-data-colored-resized`
* **Resolution:** $224 \times 224 \times 3$
* **Stratified Splits (Fixed `rng(42)`):**
  * **Train ($N = 24,588$):** Mild: 1710, Moderate: 3704, No_DR: 18067, Proliferate_DR: 496, Severe: 611
  * **Validation ($N = 5,270$):** Mild: 367, Moderate: 794, No_DR: 3872, Proliferate_DR: 106, Severe: 131
  * **Test ($N = 5,268$):** Mild: 366, Moderate: 794, No_DR: 3871, Proliferate_DR: 106, Severe: 131

---

## 3. Results Summary: V4.1 Baseline vs ANTIGRAVITY Refinement

All model decisions and operating thresholds were selected **strictly on validation data**. The untouched test set was evaluated once on the final frozen model.

### Untouched Test Set Benchmark (Single Frozen Evaluation)

| Metric | V4.1 Test Benchmark | `AG_V4_5_ReducedAug` Final Test | Improvement |
| :--- | :---: | :---: | :---: |
| **ROC-AUC** | 0.8519 | **0.8564** | **$+0.0045$** |
| **PR-AUC** | 0.6838 | **0.6914** | **$+0.0076$** |
| **Sensitivity** | 69.74% (FN=312) | **69.84%** (FN=311) | **$+0.10\%$ ($-1$ FN)** |
| **Specificity** | 85.46% (FP=616) | **85.18%** (FP=628) | Meets $\ge 85\%$ mandate |
| **5-Class Accuracy** | 77.01% | **77.16%** | **$+0.15\%$** |
| **Clinical Accuracy** | 82.38% | 82.18% | $-0.20\%$ |
| **PPV** | 53.86% | 53.41% | $-0.45\%$ |
| **NPV** | 92.07% | 92.07% | $0.00\%$ |

### Clinical Confusion Matrix (Test Set N=5,268)
```
                      Predicted Referable    Predicted Non-Referable
True Referable:               720                      311
True Non-Referable:           628                     3609
```

---

## 4. Key Scientific Findings

1. **Sub-Pixel Resampling Blur:** Dataset images are natively $224 \times 224$. Random scaling ($0.95-1.05$) and translation ($\pm 8$ px) caused continuous sub-pixel bilinear interpolation blur that degraded 1–2 pixel microaneurysms. Disabling translation and scale while keeping mild rotation ($\pm 5^\circ$) and horizontal reflection dropped validation false negatives by 23 cases and boosted validation sensitivity to **69.25%**.
2. **Moderate DR Representation Bottleneck:** Deep diagnostics revealed that $80\%-85\%$ of Moderate DR false negatives had $P(\text{No\_DR}) \ge 0.80$ due to severe class imbalance (18,067 No_DR vs 3,704 Moderate in training). Controlled fine-tuning at $\text{LR} = 3 \times 10^{-5}$ improved Moderate vs No_DR AUC from **0.8069** to **0.8108 – 0.8128**.
3. **Focal Loss Limitation:** Multi-class Focal Loss ($\gamma = 2.0$) heavily suppressed majority gradients, distorting probability calibration and lowering ROC-AUC to 0.8428.

---

## 5. Repository Structure

```
.
├── README.md                          <-- Project overview and benchmark results
├── .gitignore                         <-- Git ignore rules
└── ANTIGRAVITY/
    ├── checkpoints/                   <-- Model checkpoints (MATLAB .mat dlnetwork)
    ├── figures/                       <-- ROC & PR curves, error distributions
    │   ├── final_test_roc_pr_curves.png
    │   └── moderate_nodr_distribution.png
    ├── logs/                          <-- Experiment logs and tracking
    │   ├── AG_README.txt
    │   ├── AG_experiment_log.mat
    │   └── AG_experiment_log.txt
    ├── reports/                       <-- In-depth research reports
    │   └── RESEARCH_REPORT.md
    ├── results/                       <-- Lightweight validation and test score files (.mat)
    │   ├── AG_V4_2_LR5e5_results.mat
    │   ├── AG_V4_3_LR2e5_results.mat
    │   ├── AG_V4_4_Focal_results.mat
    │   ├── AG_V4_5_ReducedAug_results.mat
    │   ├── AG_V4_6_LR1e5_RedAug_results.mat
    │   └── AG_FINAL_TEST_RESULTS.mat
    └── scripts/                       <-- MATLAB scripts and functions
        ├── AG_V4_2_LR5e5.m
        ├── AG_V4_3_LR2e5.m
        ├── AG_V4_4_Focal.m
        ├── AG_V4_5_ReducedAug.m
        ├── AG_V4_6_LR1e5_RedAug.m
        ├── AG_FINAL_TEST_EVALUATION.m
        ├── ag_evaluate_model.m
        ├── ag_focal_loss.m
        ├── ag_log_experiment.m
        ├── ag_moderate_nodr_analysis.m
        └── ag_deep_diagnostic.m
```

---

## 6. How to Run

1. Open MATLAB R2026a and set current folder to `ANTIGRAVITY`.
2. Ensure dataset is placed at `C:\Users\Abhij\Downloads\archive (1)\colored_images\colored_images`.
3. To evaluate the champion model on validation data:
   ```matlab
   addpath('scripts');
   load('checkpoints/AG_V4_5_ReducedAug.mat', 'netTrained');
   load('../DR_V4_RESNET101_SCREENING.mat', 'imdsValidation', 'classNames');
   augVal = augmentedImageDatastore([224 224 3], imdsValidation);
   scores = minibatchpredict(netTrained, augVal);
   metrics = ag_evaluate_model(scores, imdsValidation.Labels, classNames);
   ```

---

## 7. Clinical Disclaimer

This is a research/experimental diabetic-retinopathy screening model developed for dataset-level evaluation. It is not approved as a medical device, is not clinically validated, and is not for diagnostic use.
