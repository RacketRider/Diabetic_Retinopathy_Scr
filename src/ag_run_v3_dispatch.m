function metrics = ag_run_v3_dispatch(experimentName)
% RUN_V3 Run the retained E1 or E3 pipeline without changing the fixed split.

projectRoot = fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(projectRoot, 'src'));
paths = ag_project_paths();
experimentName = upper(string(experimentName));
rng(42, 'twister');

switch experimentName
    case "E1"
        fprintf('\n=======================================================\n');
        fprintf('  RUNNING EXPERIMENT: %s\n', experimentName);
        fprintf('=======================================================\n');
        requireFile(paths.datastore, 'fixed train/validation split');
        fprintf('[PROGRESS] Loading fixed split & preparing high-resolution datastores...\n');
        data = load(paths.datastore, 'imdsTrain', 'imdsValidation', 'classNames');
        classNames = string(data.classNames(:));
        targetSize = 448;
        [trainHigh, validationHigh, mapping] = ag_prepare_v3_datastores( ...
            data.imdsTrain, data.imdsValidation, targetSize, paths.highResolutionSource);
        fprintf('[PROGRESS] Datastores prepared: %d train, %d val images.\n', ...
            numel(trainHigh.Files), numel(validationHigh.Files));
        augmenter = imageDataAugmenter('RandRotation', [-5 5], 'RandXReflection', true);
        checkpointPath = fullfile(paths.models, 'AG_V4_5_ReducedAug.mat');
        requireFile(checkpointPath, 'E1 initialization checkpoint');
        fprintf('[PROGRESS] Loading initial checkpoint: %s\n', checkpointPath);
        net = ag_build_v3_network(ag_load_dlnetwork_checkpoint(checkpointPath), targetSize, classNames, false);
        config = makeConfig('V3_E1_HighResFOV448', targetSize, true, false, false);
        config.initialCheckpoint = 'AG_V4_5_ReducedAug.mat';
        [netTrained, trainingInfo] = trainSeverity(net, trainHigh, validationHigh, augmenter, config);
        fprintf('[PROGRESS] Computing validation predictions for %s...\n', config.name);
        validationData = augmentedImageDatastore([targetSize targetSize 3], validationHigh);
        scores = minibatchpredict(netTrained, validationData, 'MiniBatchSize', config.batchSize);
        fprintf('[PROGRESS] Saving model checkpoint & evaluating metrics...\n');
        if ~isfolder(paths.models), mkdir(paths.models); end
        save(fullfile(paths.models, config.name + ".mat"), 'netTrained', 'trainingInfo', ...
            'config', 'mapping', '-v7.3');
        metrics = ag_save_v3_evaluation(config.name, scores, validationHigh.Labels, ...
            classNames, config, [], trainingInfo);

    case "E3"
        metrics = ag_run_e3(paths);

    otherwise
        error('ag:v3:UnknownExperiment', 'Retained pipeline must be E1 or E3.');
end
end

function config = makeConfig(name, inputResolution, fovCrop, rdrHead, hardMining)
config = struct('name', string(name), 'seed', 42, 'inputResolution', inputResolution, ...
    'fovCrop', fovCrop, 'inputChannels', 'RGB', 'rdrHead', rdrHead, ...
    'ordinalHead', false, 'hardMining', hardMining, 'hardRepeatFactor', 2, ...
    'lambdaRDR', double(rdrHead), 'optimizer', 'sgdm', 'learningRate', 2e-5, ...
    'momentum', 0.9, 'l2Regularization', 1e-4, 'batchSize', 8, 'epochs', 4, ...
    'augmentation', 'horizontal reflection; random rotation [-5,5] degrees', ...
    'selectionObjective', 'max RDR sensitivity subject to specificity >= 0.85');
end

function [netTrained, info] = trainSeverity(net, trainData, validationData, augmenter, config)
fprintf('\n-------------------------------------------------------\n');
fprintf('  [E1] Starting Training: %s\n', config.name);
fprintf('  Epochs: %d | Batch Size: %d | Initial LR: %g\n', config.epochs, config.batchSize, config.learningRate);
fprintf('-------------------------------------------------------\n\n');

augTrain = augmentedImageDatastore([config.inputResolution config.inputResolution 3], trainData, ...
    'DataAugmentation', augmenter);
augValidation = augmentedImageDatastore([config.inputResolution config.inputResolution 3], validationData);
options = trainingOptions('sgdm', 'InitialLearnRate', config.learningRate, ...
    'Momentum', config.momentum, 'L2Regularization', config.l2Regularization, ...
    'MiniBatchSize', config.batchSize, 'MaxEpochs', config.epochs, ...
    'Shuffle', 'every-epoch', 'ExecutionEnvironment', 'auto', 'Verbose', true, ...
    'VerboseFrequency', 10, 'Plots', 'training-progress', 'ValidationData', augValidation, ...
    'ValidationFrequency', 50);
[netTrained, info] = trainnet(augTrain, net, 'crossentropy', options);
end

function requireFile(path, description)
if ~isfile(path)
    error('ag:v3:MissingPrerequisite', 'Missing %s: %s', description, path);
end
end


