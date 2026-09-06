function metrics = ag_run_e3(paths)
% AG_RUN_E3 Global 448px vision plus deterministic vessel-guided local inspection.

config = ag_e3_config();
rng(config.seed, 'twister');
requireFile(paths.datastore, 'canonical V3 split');
requireFile(fullfile(paths.models, config.initialCheckpoint), 'E1 initialization checkpoint');
checkpointPaths = struct( ...
    'latest', fullfile(paths.models, config.name + "_latest.mat"), ...
    'best', fullfile(paths.models, config.name + "_best.mat"), ...
    'final', fullfile(paths.models, config.name + ".mat"));
rejectMismatchedInitialization(checkpointPaths, config);
data = load(paths.datastore, 'imdsTrain', 'imdsValidation', 'imdsTest', 'classNames');
required = {'imdsTrain','imdsValidation','imdsTest','classNames'};
if ~all(isfield(data, required))
    error('ag:e3data:InvalidSplit', 'Canonical split is missing train/validation/test/classNames.');
end
classNames = string(data.classNames(:));
assertCanonicalSplit(data.imdsTrain, data.imdsValidation, data.imdsTest);
[trainHigh, validationHigh, mapping] = ag_prepare_v3_datastores( ...
    data.imdsTrain, data.imdsValidation, config.inputResolution, paths.highResolutionSource);
[testHigh, ~, testMapping] = ag_prepare_v3_datastores( ...
    data.imdsTest, data.imdsValidation, config.inputResolution, paths.highResolutionSource);
mapping.testFiles = testMapping.trainFiles;
mapping.testCount = numel(mapping.testFiles);

config.useGPU = gpuDeviceCount('available') > 0;
if config.useGPU
    gpu = gpuDevice; gpuName = string(gpu.Name);
else
    gpuName = "CPU";
end
printHeader(config, gpuName, numel(trainHigh.Files), numel(validationHigh.Files), numel(testHigh.Files));
cacheDir = fullfile(paths.artifacts, 'cache', 'E3_walker');
trainCache = ag_build_e3_walker_cache(trainHigh.Files, config, ...
    fullfile(cacheDir, 'train.mat'), "train", trainHigh.Labels);
validationCache = ag_build_e3_walker_cache(validationHigh.Files, config, ...
    fullfile(cacheDir, 'validation.mat'), "validation", validationHigh.Labels);
testCache = ag_build_e3_walker_cache(testHigh.Files, config, ...
    fullfile(cacheDir, 'test.mat'), "test", testHigh.Labels);
allMethods = unique([trainCache.vesselMethod; validationCache.vesselMethod; testCache.vesselMethod]);
config.actualVesselMethod = strjoin(allMethods, ', ');
config.E3.actualVesselMethod = config.actualVesselMethod;
allFovMethods = unique([trainCache.fovMethod; validationCache.fovMethod; testCache.fovMethod]);
config.actualFovMaskMethod = strjoin(allFovMethods, ', ');
config.E3.actualFovMaskMethod = config.actualFovMaskMethod;
config.resumeSignature = ag_e3_resume_signature(config);

trainData = observationSet(trainHigh, trainCache, classNames);
validationData = observationSet(validationHigh, validationCache, classNames);
testData = observationSet(testHigh, testCache, classNames);
validateAlignment(trainData, config, "train");
validateAlignment(validationData, config, "validation");
validateAlignment(testData, config, "test");

checkpoint = ag_load_dlnetwork_checkpoint(fullfile(paths.models, config.initialCheckpoint));
fprintf('[PASS] E1 checkpoint load\n');
model = ag_build_e3_model(checkpoint, config, classNames);
walkerDebugDir = fullfile(paths.v3Figures, char(config.name), 'walker_debug');
visualReport = ag_visualize_e3_walker(trainData, config, walkerDebugDir, 16);
ag_e3_preflight(model, trainData, validationData, config, paths.models);
[model, trainingInfo] = ag_train_e3(model, trainData, validationData, config, ...
    checkpointPaths, mapping, classNames);

fprintf('[E3] Evaluating best checkpoint on canonical validation split...\n');
validationPrediction = ag_e3_predict(model, validationData, config, ...
    ["full" "globalOnly" "random"]);
metrics = ag_save_v3_evaluation(config.name, validationPrediction.scores.full, ...
    validationData.labels, classNames, config, [], trainingInfo);
fprintf('[E3] Evaluating once on canonical test split at the validation-selected threshold...\n');
testPrediction = ag_e3_predict(model, testData, config, "full");
metrics = ag_finalize_e3_evaluation(metrics, validationPrediction, testPrediction, ...
    validationData, testData, model, config, trainingInfo, visualReport, paths);
printResults(metrics);
end

function output = observationSet(imds, cache, classNames)
if ~all(cache.completed) || ~isequal(cache.files, cellstr(string(imds.Files(:))))
    error('ag:e3data:CacheMismatch', 'Walker cache does not match datastore file order.');
end
output = struct('files', {imds.Files(:)}, 'labels', imds.Labels(:), ...
    'classNames', classNames(:), 'coords', double(cache.coords), ...
    'randomCoords', double(cache.randomCoords), 'walkerScores', cache.scores, ...
    'samplingType', cache.samplingType, 'fovMethod', cache.fovMethod, ...
    'fovFraction', cache.fovFraction);
end

function validateAlignment(data, config, splitName)
count = numel(data.files);
if numel(data.labels) ~= count || size(data.coords, 3) ~= count || ...
        size(data.coords, 1) ~= config.numPatches || size(data.coords, 2) ~= 2 || ...
        size(data.randomCoords, 3) ~= count
    error('ag:e3data:Alignment', '%s global/local/target observations are misaligned.', splitName);
end
fprintf('[PASS] E3 %s alignment: %d global = %d walker sets = %d targets (%d patches/image)\n', ...
    splitName, count, size(data.coords, 3), numel(data.labels), config.numPatches);
end

function assertCanonicalSplit(trainData, validationData, testData)
train = canonicalStems(trainData.Files); validation = canonicalStems(validationData.Files); test = canonicalStems(testData.Files);
if ~isempty(intersect(train, validation)) || ~isempty(intersect(train, test)) || ~isempty(intersect(validation, test))
    error('ag:e3data:SplitLeakage', 'Canonical train/validation/test split overlap detected.');
end
end

function stems = canonicalStems(files)
stems = strings(numel(files), 1);
for i = 1:numel(files), stems(i) = lower(portableStem(files{i})); end
end

function printHeader(config, gpuName, trainCount, validationCount, testCount)
fprintf(['\n====================================================================\n' ...
    'ANTIGRAVITY V3 - E3\nGLOBAL VISION + RETINA WALKER\n' ...
    '====================================================================\n' ...
    'Global backbone       : ResNet-101 (E1 initialization)\n' ...
    'Local backbone        : Shared ResNet-101 stem through %s\n' ...
    'Input resolution      : %dx%d\nWalker patches/image  : %d\n' ...
    'Vessel patches        : %d\nExploration patches   : %d\n' ...
    'Local patch size      : %dx%d\nAggregation           : %s\n' ...
    'Fusion dimension      : %d\nTrain samples         : %d\n' ...
    'Validation samples    : %d\nTest samples          : %d\nGPU                    : %s\n' ...
    '====================================================================\n'], ...
    config.localFeatureLayer, config.inputResolution, config.inputResolution, ...
    config.numPatches, config.numVesselPatches, config.numExplorationPatches, ...
    config.patchSize(1), config.patchSize(2), config.aggregation, config.fusionDim, ...
    trainCount, validationCount, testCount, gpuName);
end

function printResults(metrics)
fprintf(['\n====================================================================\nE3 RESULTS (VALIDATION)\n' ...
    '====================================================================\n' ...
    'Overall Accuracy             : %.2f %%\nBalanced Accuracy            : %.2f %%\n' ...
    'Macro F1                     : %.4f\nReferable Sensitivity        : %.2f %%\n' ...
    'Referable Specificity        : %.2f %%\nReferable ROC-AUC            : %.4f\n' ...
    'Moderate Recall              : %.2f %%\nModerate -> No_DR FN         : %d\n' ...
    'Mean inference time/image    : %.2f ms\n' ...
    '====================================================================\n'], ...
    100 * metrics.overallAccuracy, 100 * metrics.balancedAccuracy, metrics.macroF1, ...
    100 * metrics.sensitivity, 100 * metrics.specificity, metrics.rocAuc, ...
    100 * metrics.moderateRecall, metrics.moderateToNoDR, ...
    1000 * metrics.e3.meanInferenceTime);
end

function requireFile(path, description)
if ~isfile(path), error('ag:e3:MissingPrerequisite', 'Missing %s: %s', description, path); end
end

function rejectMismatchedInitialization(checkpointPaths, config)
paths = struct2cell(checkpointPaths);
for i = 1:numel(paths)
    if ~isfile(paths{i}), continue; end
    saved = load(paths{i}, 'config');
    if isfield(saved, 'config') && isfield(saved.config, 'initialCheckpoint') && ...
            string(saved.config.initialCheckpoint) ~= string(config.initialCheckpoint)
        error('ag:e3train:InitializationMismatch', ...
            ['E3 checkpoint initialization mismatch:\nsaved  = %s\ncurrent = %s\n' ...
            'Move or rename incompatible checkpoint: %s\nWalker caches are unaffected.'], ...
            string(saved.config.initialCheckpoint), string(config.initialCheckpoint), paths{i});
    end
end
end
