% AG_V4_6_LR1e5_RedAug.m
% Experiment 5: Final polish from AG_V4_5_ReducedAug at LR = 1e-5 with Reduced Augmentation
clear; clc;

projectRoot = fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(projectRoot, 'src'));
paths = ag_project_paths();

fprintf('====================================================================\n');
fprintf('STARTING EXPERIMENT: AG_V4_6_LR1e5_RedAug\n');
fprintf('Hypothesis: Final refinement at LR=1e-5 continuing from our best candidate\n');
fprintf('AG_V4_5_ReducedAug will finalize convergence in the optimal lesion-preserving subspace.\n');
fprintf('====================================================================\n');

% Load baseline datastores
data = load(paths.datastore, 'imdsTrain', 'imdsValidation', 'classNames');
imdsTrain = data.imdsTrain;
imdsValidation = data.imdsValidation;
classNames = data.classNames;

% Load AG_V4_5_ReducedAug checkpoint as base
checkpoint = load(fullfile(paths.models, 'AG_V4_5_ReducedAug.mat'), 'netTrained');
netTrained = checkpoint.netTrained;
currentNet = netTrained;

% Reduced Augmentation: Mild rotation [-5 5], reflection only. No translation, no scaling.
imageAugmenter = imageDataAugmenter( ...
    'RandRotation', [-5 5], ...
    'RandXReflection', true);

augTrain = augmentedImageDatastore([224 224 3], imdsTrain, ...
    'DataAugmentation', imageAugmenter);

% Validation Datastore (no augmentation)
augVal = augmentedImageDatastore([224 224 3], imdsValidation);

% Training options
lr = 1e-5;
epochs = 2;
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
    'Plots', 'none');

% Train network
fprintf('Beginning training for %d epochs at LR=%g with Reduced Augmentation...\n', epochs, lr);
tStart = tic;
netTrained = trainnet(augTrain, currentNet, "crossentropy", opts);
trainTime = toc(tStart);
fprintf('Training completed in %.1f minutes (%.1f seconds).\n', trainTime/60, trainTime);

% Evaluate on validation set
fprintf('Evaluating on validation set...\n');
valScores = minibatchpredict(netTrained, augVal);
YVal = imdsValidation.Labels;
metrics = ag_evaluate_model(valScores, YVal, classNames);

% Compare against V4.1 and AG_V4_5
baseline = load(fullfile(paths.models, 'V4_1_base.mat'), 'valScoresV41', 'YValV41', 'classNamesV4');
valScoresV41 = baseline.valScoresV41;
YValV41 = baseline.YValV41;
classNamesV4 = baseline.classNamesV4;
baselineMetrics = ag_evaluate_model(valScoresV41, YValV41, classNamesV4);
previous = load(fullfile(paths.results, 'AG_V4_5_ReducedAug_results.mat'), 'metrics');
v45Metrics = previous.metrics;

fprintf('\n--- COMPARISON WITH V4.1 BASELINE & AG_V4_5 ---\n');
fprintf('Metric               V4.1 Base      AG_V4_5 (Best)    AG_V4_6 (LR1e-5)  Diff vs Base\n');
fprintf('ROC-AUC:             %.4f         %.4f            %.4f            %+.4f\n', ...
    baselineMetrics.rocAuc, v45Metrics.rocAuc, metrics.rocAuc, metrics.rocAuc - baselineMetrics.rocAuc);
fprintf('PR-AUC:              %.4f         %.4f            %.4f            %+.4f\n', ...
    baselineMetrics.prAuc, v45Metrics.prAuc, metrics.prAuc, metrics.prAuc - baselineMetrics.prAuc);
fprintf('Sensitivity:         %.2f%%        %.2f%%           %.2f%%           %+.2f%%\n', ...
    baselineMetrics.sensitivity*100, v45Metrics.sensitivity*100, metrics.sensitivity*100, (metrics.sensitivity - baselineMetrics.sensitivity)*100);
fprintf('Specificity:         %.2f%%        %.2f%%           %.2f%%           %+.2f%%\n', ...
    baselineMetrics.specificity*100, v45Metrics.specificity*100, metrics.specificity*100, (metrics.specificity - baselineMetrics.specificity)*100);
fprintf('Clinical Accuracy:   %.2f%%        %.2f%%           %.2f%%           %+.2f%%\n', ...
    baselineMetrics.clinicalAccuracy*100, v45Metrics.clinicalAccuracy*100, metrics.clinicalAccuracy*100, (metrics.clinicalAccuracy - baselineMetrics.clinicalAccuracy)*100);
fprintf('Mod vs No_DR AUC:    %.4f         %.4f            %.4f            %+.4f\n', ...
    baselineMetrics.modNoRocAuc, v45Metrics.modNoRocAuc, metrics.modNoRocAuc, metrics.modNoRocAuc - baselineMetrics.modNoRocAuc);
fprintf('Clinical FN:         %d           %d              %d              %+d\n', ...
    baselineMetrics.FN, v45Metrics.FN, metrics.FN, metrics.FN - baselineMetrics.FN);

% Decision rule
isImproved = (metrics.rocAuc >= v45Metrics.rocAuc) && ...
             (metrics.prAuc >= v45Metrics.prAuc) && ...
             (metrics.sensitivity >= v45Metrics.sensitivity);
if isImproved
    decision = 'ACCEPTED';
    fprintf('\n>>> RESULT: ACCEPTED as NEW best candidate!\n');
else
    decision = 'REJECTED';
    fprintf('\n>>> RESULT: REJECTED (Did not outperform previous best AG_V4_5).\n');
end

% Save checkpoint
if ~isfolder(paths.models), mkdir(paths.models); end
if ~isfolder(paths.results), mkdir(paths.results); end
checkpointPath = fullfile(paths.models, 'AG_V4_6_LR1e5_RedAug.mat');
save(checkpointPath, 'netTrained', 'opts', 'trainTime', '-v7.3');
fprintf('Saved checkpoint to: %s\n', checkpointPath);

% Save results
resultsPath = fullfile(paths.results, 'AG_V4_6_LR1e5_RedAug_results.mat');
save(resultsPath, 'metrics', 'valScores', 'YVal', 'classNames', 'trainTime', 'decision', '-v7.3');
fprintf('Saved results to: %s\n', resultsPath);

% Log experiment
expLog = struct();
expLog.ID = 'AG_V4_6_LR1e5_RedAug';
expLog.dateTime = char(datetime('now'));
expLog.architecture = 'ResNet-101';
expLog.parameterCount = 42558661;
expLog.initialCheckpoint = 'AG_V4_5_ReducedAug.mat';
expLog.learningRate = lr;
expLog.optimizer = 'SGDM';
expLog.batchSize = batchSize;
expLog.epochs = epochs;
expLog.loss = 'crossentropy';
expLog.classWeights = 'None';
expLog.augmentation = 'Rot[-5,5], XRefl only (No translation/scale)';
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

fprintf('\nEXPERIMENT AG_V4_6_LR1e5_RedAug COMPLETE.\n');
