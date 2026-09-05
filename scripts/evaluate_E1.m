% VALIDATION-ONLY SCRIPT
% NO TRAINING OCCURS IN THIS FILE

clear; clc;

projectRoot = fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(projectRoot, 'src'));
paths = ag_project_paths();
experimentName = "V3_E1_HighResFOV448";
modelPath = fullfile(paths.models, experimentName + ".mat");

if ~isfile(modelPath)
    error('ag:v3evaluate:MissingModel', 'Missing saved E1 model: %s', modelPath);
end
fprintf('[VALIDATION ONLY] Loading existing V3_E1_HighResFOV448 model...\n');
checkpoint = load(modelPath, 'netTrained', 'trainingInfo', 'config', 'mapping');
requiredFields = {'netTrained', 'trainingInfo', 'config', 'mapping'};
if ~all(isfield(checkpoint, requiredFields)) || ~isa(checkpoint.netTrained, 'dlnetwork')
    error('ag:v3evaluate:InvalidModel', 'Saved E1 model is missing required checkpoint data: %s', modelPath);
end

config = checkpoint.config;
requiredConfigFields = {'name', 'seed', 'inputResolution', 'fovCrop', 'inputChannels', ...
    'rdrHead', 'ordinalHead', 'hardMining', 'batchSize'};
if ~all(isfield(config, requiredConfigFields)) || string(config.name) ~= experimentName || ...
        config.seed ~= 42 || config.inputResolution ~= 448 || ~logical(config.fovCrop) || ...
        string(config.inputChannels) ~= "RGB" || logical(config.rdrHead) || ...
        logical(config.ordinalHead) || logical(config.hardMining) || config.batchSize ~= 8
    error('ag:v3evaluate:ConfigMismatch', 'Saved checkpoint does not contain the original E1 configuration.');
end

if ~isfile(paths.datastore)
    error('ag:v3evaluate:MissingSplit', 'Missing fixed train/validation split: %s', paths.datastore);
end
fprintf('[VALIDATION ONLY] Reconstructing original validation set...\n');
rng(config.seed, 'twister');
data = load(paths.datastore, 'imdsTrain', 'imdsValidation', 'classNames');
requiredDataFields = {'imdsTrain', 'imdsValidation', 'classNames'};
if ~all(isfield(data, requiredDataFields))
    error('ag:v3evaluate:InvalidSplit', 'Fixed split file is missing required validation metadata: %s', paths.datastore);
end

classNames = string(data.classNames(:));
targetSize = config.inputResolution;
[~, validationHigh, reconstructedMapping] = ag_prepare_v3_datastores( ...
    data.imdsTrain, data.imdsValidation, targetSize, paths.highResolutionSource);
requiredMappingFields = {'trainFiles', 'validationFiles', 'targetSize', 'sourceRoot'};
if ~all(isfield(checkpoint.mapping, requiredMappingFields)) || ...
        ~isequal(string(checkpoint.mapping.trainFiles(:)), string(reconstructedMapping.trainFiles(:))) || ...
        ~isequal(string(checkpoint.mapping.validationFiles(:)), string(reconstructedMapping.validationFiles(:))) || ...
        checkpoint.mapping.targetSize ~= reconstructedMapping.targetSize || ...
        string(checkpoint.mapping.sourceRoot) ~= string(reconstructedMapping.sourceRoot)
    error('ag:v3evaluate:SplitMismatch', 'Original E1 validation split could not be reproduced exactly.');
end

validationData = augmentedImageDatastore([targetSize targetSize 3], validationHigh);
fprintf('[VALIDATION ONLY] Running validation inference...\n');
fprintf('[VALIDATION ONLY] No training will be performed.\n');
scores = minibatchpredict(checkpoint.netTrained, validationData, ...
    'MiniBatchSize', config.batchSize);
pReferable = [];
metrics = ag_save_v3_evaluation(config.name, scores, validationHigh.Labels, ...
    classNames, config, pReferable, checkpoint.trainingInfo);
