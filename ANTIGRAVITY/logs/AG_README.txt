================================================================================
PROJECT ANTIGRAVITY — DIABETIC RETINOPATHY SCREENING EXPERIMENTAL BRANCH
================================================================================

1. PROJECT PURPOSE
------------------
The objective of this research branch is to build and refine a 5-class diabetic
retinopathy classifier for clinical screening.
Classes: Mild, Moderate, No_DR, Proliferate_DR, Severe.
Clinical Referral Definition:
  - REFERABLE DR:     Moderate OR Proliferate_DR OR Severe
  - NON-REFERABLE DR: Mild OR No_DR
Primary clinical objective: Strong referable-DR detection (Sensitivity) while
maintaining approximately >= 85% Specificity. The primary objective is NOT simply
5-class argmax accuracy.

2. V4.1 BASELINE METRICS (LOCKED BENCHMARK)
--------------------------------------------
Architecture: ResNet-101 (~42.56M parameters), trained for 10 epochs.
Validation Benchmark (at threshold 0.1460):
  - Accuracy:          76.22%
  - ROC-AUC:           0.8440
  - PR-AUC:            0.6623
  - Sensitivity:       67.02%
  - Specificity:       85.07%
  - PPV:               52.19%
  - NPV:               91.38%
  - Clinical Accuracy: 81.54%
  - Confusion Matrix (Clinical): TP=691, TN=3606, FP=633, FN=340

Locked Test Benchmark (at threshold 0.1460):
  - 5-class Accuracy:  77.01%
  - ROC-AUC:           0.8519
  - PR-AUC:            0.6838
  - Sensitivity:       69.74%
  - Specificity:       85.46%
  - PPV:               53.86%
  - NPV:               92.07%
  - Clinical Accuracy: 82.38%
  - Clinical Confusion: TP=719, TN=3621, FP=616, FN=312

3. DATASET LOCATION & SPLIT
----------------------------
Dataset root:
  C:\Users\Abhij\Downloads\archive (1)\colored_images\colored_images
Stratified Split (Fixed with rng(42), exactly preserved):
  - Train:      24,588 images (Mild: 1710, Moderate: 3704, No_DR: 18067, Proliferate_DR: 496, Severe: 611)
  - Validation:  5,270 images (Mild: 367,  Moderate: 794,  No_DR: 3872,  Proliferate_DR: 106, Severe: 131)
  - Test:        5,268 images (Mild: 366,  Moderate: 794,  No_DR: 3871,  Proliferate_DR: 106, Severe: 131)

4. EXPERIMENTAL RULES
---------------------
- RULE 1: TEST SET IS STRICTLY LOCKED. No evaluation on test set for tuning or model selection.
- RULE 2: VALIDATION-ONLY MODEL SELECTION. Operating threshold is selected strictly on validation
  by maximizing sensitivity subject to specificity >= 85%.
- RULE 3: PRESERVE THE DATA SPLIT. Never modify, reshuffle, or regenerate the split.
- RULE 4: NO BLIND SWEEPS. Every experiment must have a scientific hypothesis.
- RULE 5: ISOLATION & PRESERVATION. Original V1-V4.1 files are strictly read-only and never modified.
  All experimental code, checkpoints, logs, and results reside inside ANTIGRAVITY\.

5. LIST OF EXPERIMENTS PERFORMED
---------------------------------
Baseline: V4.1 (ResNet-101, 10 epochs SGDM lr=1e-4) -> Sensitivity 67.02%, Specificity 85.07%, ROC-AUC 0.8440, PR-AUC 0.6623.
================================================================================
