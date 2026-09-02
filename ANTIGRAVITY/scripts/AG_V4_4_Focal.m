% AG_V4_4_Focal.m
% Experiment 3: Phase 2 Loss Function Refinement with Focal Loss (gamma = 2.0)
clear; clc;

addpath('C:\Users\Abhij\Documents\DR_SIH\ANTIGRAVITY\scripts');

fprintf('====================================================================\n');
fprintf('STARTING EXPERIMENT: AG_V4_4_Focal\n');
fprintf('Hypothesis: Focal Loss (gamma=2.0) downweights easy No_DR retinas,\n');
fprintf('forcing the gradient to focus on hard Moderate DR vs No_DR discrimination.\n');
fprintf('====================================================================\n');

% Load baseline datastores
load('C:\Users\Abhij\Documents\DR_SIH\DR_V4_RESNET101_SCREENING.mat', 'imdsTrain', 'imdsValidation', 'classNames');

% Load AG_V4_2_LR5e5 checkpoint as base
load('C:\Users\Abhij\Documents\DR_SIH\ANTIGRAVITY\checkpoints\AG_V4_2_LR5e5.mat', 'netTrained');
currentNet = netTrained;

% Training Augmentation (identical to baseline)
imageAugmenter = imageDataAugmenter( ...
    'RandRotation', [-10 10], ...
    'RandXReflection', true, ...
    'RandXTranslation', [-8 8], ...
    'RandYTranslation', [-8 8], ...
    'RandScale', [0.95 1.05]);

augTrain = augmentedImageDatastore([224 224 3], imdsTrain, ...
    'DataAugmentation', imageAugmenter);

% Validation Datastore (no augmentation)
augVal = augmentedImageDatastore([224 224 3], imdsValidation);

% Training options
lr = 2e-5;
epochs = 2;
batchSize = 32;
gamma = 2.0;

focalLossFcn = @(Y, T) ag_focal_loss(Y, T, gamma);

opts = trainingOptions('sgdm', ...
    'InitialLearnRate', lr, ...
    'Momentum', 0.9, ...
    'L2Regularization', 1e-4, ...
    'MiniBatchSize', batchSize, ...
    'MaxEpochs', epochs, ...
    'Shuffle', 'every-epoch', ...
    'ExecutionEnvironment', 'gpu', ...
    'Verbose', true, ...
    'VerboseFrequency', 100, ...
    'Plots', 'none');

% Train network
fprintf('Beginning training for %d epochs with Focal Loss (gamma=%.1f) at LR=%g...\n', epochs, gamma, lr);
tStart = tic;
netTrained = trainnet(augTrain, currentNet, focalLossFcn, opts);
trainTime = toc(tStart);
fprintf('Training completed in %.1f minutes (%.1f seconds).\n', trainTime/60, trainTime);

% Evaluate on validation set
fprintf('Evaluating on validation set...\n');
valScores = minibatchpredict(netTrained, augVal);
YVal = imdsValidation.Labels;
metrics = ag_evaluate_model(valScores, YVal, classNames);

% Compare against V4.1 and AG_V4_2
load('C:\Users\Abhij\Documents\DR_SIH\ANTIGRAVITY\checkpoints\V4_1_base.mat', 'valScoresV41', 'YValV41', 'classNamesV4');
baselineMetrics = ag_evaluate_model(valScoresV41, YValV41, classNamesV4);
load('C:\Users\Abhij\Documents\DR_SIH\ANTIGRAVITY\results\AG_V4_2_LR5e5_results.mat', 'metrics');
v42Metrics = metrics;

fprintf('\n--- COMPARISON WITH V4.1 BASELINE & AG_V4_2 ---\n');
fprintf('Metric               V4.1 Base      AG_V4_2 (5e-5)    AG_V4_4 (Focal)   Diff vs Base\n');
fprintf('ROC-AUC:             %.4f         %.4f            %.4f            %+.4f\n', ...
    baselineMetrics.rocAuc, v42Metrics.rocAuc, metrics.rocAuc, metrics.rocAuc - baselineMetrics.rocAuc);
fprintf('PR-AUC:              %.4f         %.4f            %.4f            %+.4f\n', ...
    baselineMetrics.prAuc, v42Metrics.prAuc, metrics.prAuc, metrics.prAuc - baselineMetrics.prAuc);
fprintf('Sensitivity:         %.2f%%        %.2f%%           %.2f%%           %+.2f%%\n', ...
    baselineMetrics.sensitivity*100, v42Metrics.sensitivity*100, metrics.sensitivity*100, (metrics.sensitivity - baselineMetrics.sensitivity)*100);
fprintf('Specificity:         %.2f%%        %.2f%%           %.2f%%           %+.2f%%\n', ...
    baselineMetrics.specificity*100, v42Metrics.specificity*100, metrics.specificity*100, (metrics.specificity - baselineMetrics.specificity)*100);
fprintf('Clinical Accuracy:   %.2f%%        %.2f%%           %.2f%%           %+.2f%%\n', ...
    baselineMetrics.clinicalAccuracy*100, v42Metrics.clinicalAccuracy*100, metrics.clinicalAccuracy*100, (metrics.clinicalAccuracy - baselineMetrics.clinicalAccuracy)*100);
fprintf('Mod vs No_DR AUC:    %.4f         %.4f            %.4f            %+.4f\n', ...
    baselineMetrics.modNoRocAuc, v42Metrics.modNoRocAuc, metrics.modNoRocAuc, metrics.modNoRocAuc - baselineMetrics.modNoRocAuc);
fprintf('Clinical FN:         %d           %d              %d              %+d\n', ...
    baselineMetrics.FN, v42Metrics.FN, metrics.FN, metrics.FN - baselineMetrics.FN);

% Decision rule
isImproved = (metrics.rocAuc >= v42Metrics.rocAuc) && ...
             (metrics.prAuc >= v42Metrics.prAuc) && ...
             (metrics.sensitivity >= v42Metrics.sensitivity);
if isImproved
    decision = 'ACCEPTED';
    fprintf('\n>>> RESULT: ACCEPTED as NEW best candidate!\n');
else
    decision = 'REJECTED';
    fprintf('\n>>> RESULT: REJECTED (Did not outperform previous best).\n');
end

% Save checkpoint
checkpointPath = 'C:\Users\Abhij\Documents\DR_SIH\ANTIGRAVITY\checkpoints\AG_V4_4_Focal.mat';
save(checkpointPath, 'netTrained', 'opts', 'trainTime', '-v7.3');
fprintf('Saved checkpoint to: %s\n', checkpointPath);

% Save results
resultsPath = 'C:\Users\Abhij\Documents\DR_SIH\ANTIGRAVITY\results\AG_V4_4_Focal_results.mat';
save(resultsPath, 'metrics', 'valScores', 'YVal', 'classNames', 'trainTime', 'decision', '-v7.3');
fprintf('Saved results to: %s\n', resultsPath);

% Log experiment
expLog = struct();
expLog.ID = 'AG_V4_4_Focal';
expLog.dateTime = char(datetime('now'));
expLog.architecture = 'ResNet-101';
expLog.parameterCount = 42558661;
expLog.initialCheckpoint = 'AG_V4_2_LR5e5.mat';
expLog.learningRate = lr;
expLog.optimizer = 'SGDM';
expLog.batchSize = batchSize;
expLog.epochs = epochs;
expLog.loss = 'Focal Loss (gamma=2.0)';
expLog.classWeights = 'None';
expLog.augmentation = 'Rot[-10,10], XRefl, Trans[-8,8], Scale[0.95,1.05]';
expLog.inputSize = [224 224 3];
expLog.valAccuracy = metrics.valAccuracy;
expLog.valRocAuc = metrics.rocAuc;
expLog.valPrAuc = metrics.prAuc;
expLog.threshold = metrics.threshold;
expLog.valSensitivity = metrics.sensitivity;
expLog.valSpecificity = metrics.specificity;
expLog.valPPV = metrics.ppv;
expLog.valNPV = metrics.npv;
expLog.valClinicalAccuracy = metrics.clinicalAccuracy;
expLog.decision = decision;
expLog.notes = sprintf('Train time: %.1fm. FN=%d (Mod=%d, Prolif=%d, Sev=%d). Mod-NoDR AUC: %.4f', ...
    trainTime/60, metrics.FN, metrics.fnModerate, metrics.fnProlif, metrics.fnSevere, metrics.modNoRocAuc);
ag_log_experiment(expLog);

fprintf('\nEXPERIMENT AG_V4_4_Focal COMPLETE.\n');
