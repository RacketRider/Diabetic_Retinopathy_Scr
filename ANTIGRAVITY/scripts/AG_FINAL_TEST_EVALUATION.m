% AG_FINAL_TEST_EVALUATION.m
% ONE-TIME FINAL TEST EVALUATION OF FROZEN CANDIDATE: AG_V4_5_ReducedAug
%
% CRITICAL PROTOCOL:
% - Model, checkpoint, and threshold (0.1199) were 100% frozen using validation data only.
% - This is the single, final untouched test evaluation.
% - No further tuning or modifications are permitted based on this output.
clear; clc;

addpath('C:\Users\Abhij\Documents\DR_SIH\ANTIGRAVITY\scripts');

fprintf('====================================================================\n');
fprintf('STARTING ONE-TIME FINAL UNTOUCHED TEST EVALUATION\n');
fprintf('Candidate Model: AG_V4_5_ReducedAug (ResNet-101)\n');
fprintf('Locked Validation Threshold: 0.1199\n');
fprintf('====================================================================\n');

% Load test datastore and class names
load('C:\Users\Abhij\Documents\DR_SIH\DR_V4_RESNET101_SCREENING.mat', 'imdsTest', 'classNames');

% Load frozen candidate checkpoint
load('C:\Users\Abhij\Documents\DR_SIH\ANTIGRAVITY\checkpoints\AG_V4_5_ReducedAug.mat', 'netTrained');

% Create test datastore (no augmentation)
augTest = augmentedImageDatastore([224 224 3], imdsTest);

% Predict on untouched test set
fprintf('Predicting on untouched test set (N = %d images)...\n', length(imdsTest.Files));
tStart = tic;
testScores = minibatchpredict(netTrained, augTest);
tPredict = toc(tStart);
fprintf('Test prediction completed in %.1f seconds.\n', tPredict);

YTest = imdsTest.Labels;
if ~iscategorical(YTest)
    YTest = categorical(YTest, classNames);
end

% Compute Referable probability: columns [2, 4, 5] (Moderate, Proliferate_DR, Severe)
pRefTest = sum(testScores(:, [2, 4, 5]), 2);
trueRefTest = (YTest == 'Moderate') | (YTest == 'Proliferate_DR') | (YTest == 'Severe');

% ROC-AUC and PR-AUC
[~, ~, ~, roc_auc_test] = perfcurve(trueRefTest, pRefTest, true);
[~, ~, ~, pr_auc_test] = perfcurve(trueRefTest, pRefTest, true, 'xCrit', 'reca', 'yCrit', 'prec');

% Apply FROZEN validation threshold: 0.1199
thFrozen = 0.1199;
predRefTest = (pRefTest >= thFrozen);

TP = sum(predRefTest & trueRefTest);
TN = sum(~predRefTest & ~trueRefTest);
FP = sum(predRefTest & ~trueRefTest);
FN = sum(~predRefTest & trueRefTest);

sensTest = TP / (TP + FN);
specTest = TN / (TN + FP);
ppvTest = TP / (TP + FP);
npvTest = TN / (TN + FN);
clinAccTest = (TP + TN) / (TP + TN + FP + FN);

% 5-class argmax accuracy
[~, argmaxIdx] = max(testScores, [], 2);
predLabelsTest = categorical(classNames(argmaxIdx), classNames);
testAcc5Class = mean(predLabelsTest == YTest);

% Clinical FN breakdown
fnMask = (~predRefTest & trueRefTest);
fnModerate = sum(fnMask & (YTest == 'Moderate'));
fnProlif = sum(fnMask & (YTest == 'Proliferate_DR'));
fnSevere = sum(fnMask & (YTest == 'Severe'));

% Clinical FP breakdown
fpMask = (predRefTest & ~trueRefTest);
fpMild = sum(fpMask & (YTest == 'Mild'));
fpNoDR = sum(fpMask & (YTest == 'No_DR'));

% 5-class confusion matrix
cmTest = confusionmat(YTest, predLabelsTest, 'Order', classNames);

% Per-class recall
classes = categories(YTest);
recallPerClass = zeros(length(classes), 1);
for c = 1:length(classes)
    recallPerClass(c) = mean(predLabelsTest(YTest == classes{c}) == classes{c});
end

fprintf('\n====================================================================\n');
fprintf('FINAL TEST EVALUATION RESULTS: AG_V4_5_ReducedAug\n');
fprintf('====================================================================\n');
fprintf('Validation-Locked Threshold: %.4f\n', thFrozen);
fprintf('5-Class Accuracy:            %.2f%%\n', testAcc5Class*100);
fprintf('ROC-AUC:                     %.4f\n', roc_auc_test);
fprintf('PR-AUC:                      %.4f\n', pr_auc_test);
fprintf('Sensitivity:                 %.2f%% (TP = %d, FN = %d)\n', sensTest*100, TP, FN);
fprintf('Specificity:                 %.2f%% (TN = %d, FP = %d)\n', specTest*100, TN, FP);
fprintf('PPV:                         %.2f%%\n', ppvTest*100);
fprintf('NPV:                         %.2f%%\n', npvTest*100);
fprintf('Clinical Accuracy:           %.2f%%\n', clinAccTest*100);
fprintf('\nClinical Confusion Matrix:\n');
fprintf('                     Predicted Referable    Predicted NonReferable\n');
fprintf('True Referable:      %d                     %d\n', TP, FN);
fprintf('True NonReferable:   %d                     %d\n', FP, TN);
fprintf('\nClinical FN Breakdown:\n');
fprintf('  Moderate:        %d / 794 (%.1f%%)\n', fnModerate, 100*fnModerate/794);
fprintf('  Proliferate_DR:  %d / 106 (%.1f%%)\n', fnProlif, 100*fnProlif/106);
fprintf('  Severe:          %d / 131 (%.1f%%)\n', fnSevere, 100*fnSevere/131);
fprintf('\nClinical FP Breakdown:\n');
fprintf('  Mild:   %d / 366\n', fpMild);
fprintf('  No_DR:  %d / 3871\n', fpNoDR);
fprintf('====================================================================\n\n');

% Compare against LOCKED V4.1 TEST BENCHMARK
load('C:\Users\Abhij\Documents\DR_SIH\DR_V41_FINAL_TEST_RESULTS.mat', ...
    'sensitivityTestV41', 'specificityTestV41', 'aucTestV41', 'aucPRTestV41', ...
    'clinicalAccuracyTestV41', 'ppvTestV41', 'npvTestV41', 'testAccuracyV41', ...
    'TPTestV41', 'TNTestV41', 'FPTestV41', 'FNTestV41');

fprintf('====================================================================\n');
fprintf('FINAL HEAD-TO-HEAD COMPARISON ON UNTOUCHED TEST SET\n');
fprintf('====================================================================\n');
fprintf('Metric                  V4.1 Test Benchmark    AG_V4_5 Final Test    Net Improvement\n');
fprintf('Sensitivity:            %.2f%% (FN=%d)          %.2f%% (FN=%d)         %+.2f%% (%+d FN)\n', ...
    sensitivityTestV41*100, FNTestV41, sensTest*100, FN, (sensTest - sensitivityTestV41)*100, FN - FNTestV41);
fprintf('Specificity:            %.2f%% (FP=%d)          %.2f%% (FP=%d)         %+.2f%% (%+d FP)\n', ...
    specificityTestV41*100, FPTestV41, specTest*100, FP, (specTest - specificityTestV41)*100, FP - FPTestV41);
fprintf('ROC-AUC:                %.4f                 %.4f                %+.4f\n', ...
    aucTestV41, roc_auc_test, roc_auc_test - aucTestV41);
fprintf('PR-AUC:                 %.4f                 %.4f                %+.4f\n', ...
    aucPRTestV41, pr_auc_test, pr_auc_test - aucPRTestV41);
fprintf('Clinical Accuracy:      %.2f%%                %.2f%%               %+.2f%%\n', ...
    clinicalAccuracyTestV41*100, clinAccTest*100, (clinAccTest - clinicalAccuracyTestV41)*100);
fprintf('PPV:                    %.2f%%                %.2f%%               %+.2f%%\n', ...
    ppvTestV41*100, ppvTest*100, (ppvTest - ppvTestV41)*100);
fprintf('NPV:                    %.2f%%                %.2f%%               %+.2f%%\n', ...
    npvTestV41*100, npvTest*100, (npvTest - npvTestV41)*100);
fprintf('5-Class Accuracy:       %.2f%%                %.2f%%               %+.2f%%\n', ...
    testAccuracyV41*100, testAcc5Class*100, (testAcc5Class - testAccuracyV41)*100);
fprintf('====================================================================\n');

% Save final test results
finalSavePath = 'C:\Users\Abhij\Documents\DR_SIH\ANTIGRAVITY\results\AG_FINAL_TEST_RESULTS.mat';
save(finalSavePath, 'testScores', 'YTest', 'classNames', 'thFrozen', ...
    'roc_auc_test', 'pr_auc_test', 'sensTest', 'specTest', 'ppvTest', 'npvTest', ...
    'clinAccTest', 'testAcc5Class', 'TP', 'TN', 'FP', 'FN', 'cmTest', ...
    'fnModerate', 'fnProlif', 'fnSevere', 'fpMild', 'fpNoDR', '-v7.3');
fprintf('Saved final test results to: %s\n', finalSavePath);

% Generate test ROC and PR figures
fig = figure('Visible', 'off', 'Position', [100 100 1000 450]);
subplot(1, 2, 1);
[Xroc, Yroc] = perfcurve(trueRefTest, pRefTest, true);
plot(Xroc, Yroc, 'b-', 'LineWidth', 2); hold on;
plot([0 1], [0 1], 'k--');
xlabel('False Positive Rate (1 - Specificity)'); ylabel('True Positive Rate (Sensitivity)');
title(sprintf('Test ROC Curve (AUC = %.4f)', roc_auc_test));
grid on;

subplot(1, 2, 2);
[Xpr, Ypr] = perfcurve(trueRefTest, pRefTest, true, 'xCrit', 'reca', 'yCrit', 'prec');
plot(Xpr, Ypr, 'r-', 'LineWidth', 2);
xlabel('Recall (Sensitivity)'); ylabel('Precision (PPV)');
title(sprintf('Test Precision-Recall Curve (AUC = %.4f)', pr_auc_test));
grid on;

figPath = 'C:\Users\Abhij\Documents\DR_SIH\ANTIGRAVITY\figures\final_test_roc_pr_curves.png';
saveas(fig, figPath);
fprintf('Saved final test figure to: %s\n', figPath);
