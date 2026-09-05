% Bounded real-data E3 smoke test: 16 train + 16 validation images, no training update.
projectRoot = fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(projectRoot, 'src'));
paths = ag_project_paths();
split = load(paths.datastore, 'imdsTrain', 'imdsValidation', 'classNames');
classNames = string(split.classNames(:));
[trainHigh, validationHigh] = ag_prepare_v3_datastores( ...
    split.imdsTrain, split.imdsValidation, 448, paths.highResolutionSource);
trainIndices = stratified(trainHigh.Labels, classNames, 16);
validationIndices = stratified(validationHigh.Labels, classNames, 16);
trainFiles = trainHigh.Files(trainIndices); validationFiles = validationHigh.Files(validationIndices);

cfg = ag_e3_config();
cfg.cacheSaveFrequency = 8;
cfg.useGPU = gpuDeviceCount('available') > 0;
cfg.resumeSignature = ag_e3_resume_signature(cfg);
assert(contains(cfg.resumeSignature, cfg.initialCheckpoint));
temporaryRoot = fullfile(tempdir, 'ag_e3_smoke');
if isfolder(temporaryRoot), rmdir(temporaryRoot, 's'); end
mkdir(temporaryRoot);
trainCache = ag_build_e3_walker_cache(trainFiles, cfg, fullfile(temporaryRoot, 'train.mat'));
validationCache = ag_build_e3_walker_cache(validationFiles, cfg, fullfile(temporaryRoot, 'validation.mat'));
reusedCache = ag_build_e3_walker_cache(trainFiles, cfg, fullfile(temporaryRoot, 'train.mat'));
assert(isequal(reusedCache.coords, trainCache.coords));
modelVariantCfg = cfg;
modelVariantCfg.initialCheckpoint = "model-identity-must-not-affect-walker-cache.mat";
modelIndependentCache = ag_build_e3_walker_cache( ...
    trainFiles, modelVariantCfg, fullfile(temporaryRoot, 'train.mat'));
assert(isequal(modelIndependentCache.cacheKey, trainCache.cacheKey));
fprintf('[PASS] walker cache identity is model-independent\n');
trainData = makeData(trainFiles, trainHigh.Labels(trainIndices), trainCache, classNames);
validationData = makeData(validationFiles, validationHigh.Labels(validationIndices), validationCache, classNames);
checkpoint = ag_load_dlnetwork_checkpoint(fullfile(paths.models, cfg.initialCheckpoint));
fprintf('[PASS] E1 checkpoint load\n');
model = ag_build_e3_model(checkpoint, cfg, classNames);
assert(model.globalFeatureDim == 2048 && model.localFeatureDim == 256);
assert(model.fusionFeatureDim == 512 && model.fusionNet.Layers(end).OutputSize == 5);
assertResumeRejectsWrongInitialization(model, trainData, validationData, cfg, classNames, temporaryRoot);
visual = ag_visualize_e3_walker(trainData, cfg, fullfile(temporaryRoot, 'walker_debug'), 16);
assert(visual.status == "PASS");
report = ag_e3_preflight(model, trainData, validationData, cfg, temporaryRoot);
assert(report.status == "PASS" && isfinite(report.loss));
assert(~isfile(fullfile(temporaryRoot, 'E3_preflight_serialization.mat')));
cfg.actualVesselMethod = "fibermetric-green-dark";
cfg.failureExamplesPerType = 1;
validationPrediction = ag_e3_predict(model, validationData, cfg, ...
    ["full" "globalOnly" "random"]);
globalOnlyPrediction = ag_e3_predict(model, validationData, cfg, "globalOnly");
assert(isequal(size(globalOnlyPrediction.scores.globalOnly), [16 5]));
testPrediction = ag_e3_predict(model, validationData, cfg, "full");
metrics = ag_evaluate_model_v3(validationPrediction.scores.full, validationData.labels, classNames);
smokePaths = struct('v3Results', fullfile(temporaryRoot, 'results'), ...
    'v3Figures', fullfile(temporaryRoot, 'figures'), ...
    'models', fullfile(temporaryRoot, 'models'));
trainingInfo = struct('Epoch', 1, 'TrainingLoss', report.loss, ...
    'ValidationBalancedAccuracy', metrics.balancedAccuracy, 'Stage', "smoke");
metrics = ag_finalize_e3_evaluation(metrics, validationPrediction, testPrediction, ...
    validationData, validationData, model, cfg, trainingInfo, visual, smokePaths);
assert(isfield(metrics, 'test') && isfield(metrics, 'ablations'));
assert(isfile(fullfile(smokePaths.v3Results, cfg.name, 'results.mat')));
assert(isfile(fullfile(smokePaths.v3Figures, cfg.name, 'training_curves.png')));
fprintf('E3_REAL_PIPELINE_SMOKE_PASS global=%d local=%d fusion=%d\n', ...
    model.globalFeatureDim, model.localFeatureDim, model.fusionFeatureDim);

function indices = stratified(labels, classNames, count)
indices = zeros(0, 1); perClass = ceil(count / numel(classNames));
for c = 1:numel(classNames)
    matches = find(string(labels) == classNames(c));
    indices = [indices; matches(1:perClass)]; %#ok<AGROW>
end
indices = indices(1:count);
end

function data = makeData(files, labels, cache, classNames)
data = struct('files', {files(:)}, 'labels', labels(:), 'classNames', classNames(:), ...
    'coords', double(cache.coords), 'randomCoords', double(cache.randomCoords), ...
    'walkerScores', cache.scores, 'samplingType', cache.samplingType);
end

function assertResumeRejectsWrongInitialization(model, trainData, validationData, cfg, classNames, temporaryRoot)
checkpointPaths = struct('latest', fullfile(temporaryRoot, 'stale_latest.mat'), ...
    'best', fullfile(temporaryRoot, 'stale_best.mat'), ...
    'final', fullfile(temporaryRoot, 'stale_final.mat'));
mapping = struct('trainFiles', {trainData.files}, ...
    'validationFiles', {validationData.files}, 'testFiles', {validationData.files});
stale = struct('model', struct(), 'trainingInfo', struct(), 'config', cfg, ...
    'mapping', mapping, 'classNames', classNames, ...
    'velocity', struct(), 'bestValidationMetric', 0, 'epoch', 1, 'iteration', 1);
stale.config.initialCheckpoint = "wrong_initializer.mat";
stale.config.resumeSignature = ag_e3_resume_signature(stale.config);
save(checkpointPaths.latest, '-struct', 'stale', '-v7.3');
rejected = false;
try
    ag_train_e3(model, trainData, validationData, cfg, checkpointPaths, mapping, classNames);
catch ME
    rejected = strcmp(ME.identifier, 'ag:e3train:InitializationMismatch') && ...
        contains(ME.message, 'saved  = wrong_initializer.mat') && ...
        contains(ME.message, 'current = V3_E1_HighResFOV448.mat');
end
assert(rejected, 'An E3 checkpoint with the wrong initializer was not explicitly rejected.');
delete(checkpointPaths.latest);
fprintf('[PASS] resume signature rejects initializer mismatch\n');
end
