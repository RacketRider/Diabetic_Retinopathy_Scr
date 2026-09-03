% AG_V4_2_LR5e5.m
% Experiment 1: Phase 1 LR Refinement from V4.1 Checkpoint with LR = 5e-5
clear; clc;

projectRoot = fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(projectRoot, 'src'));
paths = ag_project_paths();

fprintf('====================================================================\n');
fprintf('STARTING EXPERIMENT: AG_V4_2_LR5e5\n');
fprintf('Hypothesis: Gentler fine-tuning (LR=5e-5) can refine V4.1 representations\n');
fprintf('without destabilizing the learned features.\n');
fprintf('====================================================================\n');

% Load baseline datastores
data = load(paths.datastore, 'imdsTrain', 'imdsValidation', 'classNames');
imdsTrain = data.imdsTrain;
imdsValidation = data.imdsValidation;
classNames = data.classNames;

% Load baseline V4.1 checkpoint
baseline = load(fullfile(paths.models, 'V4_1_base.mat'), 'trainedNetV41');
trainedNetV41 = baseline.trainedNetV41;

% Training Augmentation (identical to V4.1)
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
lr = 5e-5;
epochs = 3;
batchSize = 32;

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
    'ValidationData', augVal, ...
    'ValidationFrequency', 200, ...
    'Plots', 'none');

% Train network
fprintf('Beginning training for %d epochs...\n', epochs);
tStart = tic;
netTrained = trainnet(augTrain, trainedNetV41, "crossentropy", opts);
trainTime = toc(tStart);
fprintf('Training completed in %.1f minutes (%.1f seconds).\n', trainTime/60, trainTime);

% Evaluate on validation set
fprintf('Evaluating on validation set...\n');
valScores = minibatchpredict(netTrained, augVal);
YVal = imdsValidation.Labels;
metrics = ag_evaluate_model(valScores, YVal, classNames);

% Compare against V4.1
baseline = load(fullfile(paths.models, 'V4_1_base.mat'), 'valScoresV41', 'YValV41', 'classNamesV4');
valScoresV41 = baseline.valScoresV41;
YValV41 = baseline.YValV41;
classNamesV4 = baseline.classNamesV4;
baselineMetrics = ag_evaluate_model(valScoresV41, YValV41, classNamesV4);

fprintf('\n--- COMPARISON WITH V4.1 BASELINE ---\n');
fprintf('Metric               V4.1 Baseline    AG_V4_2_LR5e5       Diff\n');
fprintf('ROC-AUC:             %.4f           %.4f              %+.4f\n', baselineMetrics.rocAuc, metrics.rocAuc, metrics.rocAuc - baselineMetrics.rocAuc);
fprintf('PR-AUC:              %.4f           %.4f              %+.4f\n', baselineMetrics.prAuc, metrics.prAuc, metrics.prAuc - baselineMetrics.prAuc);
fprintf('Sensitivity:         %.2f%%          %.2f%%             %+.2f%%\n', baselineMetrics.sensitivity*100, metrics.sensitivity*100, (metrics.sensitivity - baselineMetrics.sensitivity)*100);
fprintf('Specificity:         %.2f%%          %.2f%%             %+.2f%%\n', baselineMetrics.specificity*100, metrics.specificity*100, (metrics.specificity - baselineMetrics.specificity)*100);
fprintf('Clinical Accuracy:   %.2f%%          %.2f%%             %+.2f%%\n', baselineMetrics.clinicalAccuracy*100, metrics.clinicalAccuracy*100, (metrics.clinicalAccuracy - baselineMetrics.clinicalAccuracy)*100);
fprintf('Mod vs No_DR AUC:    %.4f           %.4f              %+.4f\n', baselineMetrics.modNoRocAuc, metrics.modNoRocAuc, metrics.modNoRocAuc - baselineMetrics.modNoRocAuc);
fprintf('Clinical FN:         %d             %d                %+d\n', baselineMetrics.FN, metrics.FN, metrics.FN - baselineMetrics.FN);

% Decision rule
isImproved = (metrics.rocAuc >= baselineMetrics.rocAuc) && ...
             (metrics.prAuc >= baselineMetrics.prAuc) && ...
             (metrics.sensitivity > baselineMetrics.sensitivity);
if isImproved
    decision = 'ACCEPTED';
    fprintf('\n>>> RESULT: ACCEPTED as candidate improvement!\n');
else
    decision = 'REJECTED';
    fprintf('\n>>> RESULT: REJECTED (Did not demonstrate superior operating point over V4.1).\n');
end

% Save checkpoint
if ~isfolder(paths.models), mkdir(paths.models); end
if ~isfolder(paths.results), mkdir(paths.results); end
checkpointPath = fullfile(paths.models, 'AG_V4_2_LR5e5.mat');
save(checkpointPath, 'netTrained', 'opts', 'trainTime', '-v7.3');
fprintf('Saved checkpoint to: %s\n', checkpointPath);

% Save results
resultsPath = fullfile(paths.results, 'AG_V4_2_LR5e5_results.mat');
save(resultsPath, 'metrics', 'valScores', 'YVal', 'classNames', 'trainTime', 'decision', '-v7.3');
fprintf('Saved results to: %s\n', resultsPath);

% Log experiment
expLog = struct();
expLog.ID = 'AG_V4_2_LR5e5';
expLog.dateTime = char(datetime('now'));
expLog.architecture = 'ResNet-101';
expLog.parameterCount = 42558661;
expLog.initialCheckpoint = 'V4_1_base.mat';
expLog.learningRate = lr;
expLog.optimizer = 'SGDM';
expLog.batchSize = batchSize;
expLog.epochs = epochs;
expLog.loss = 'crossentropy';
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

fprintf('\nEXPERIMENT AG_V4_2_LR5e5 COMPLETE.\n');
