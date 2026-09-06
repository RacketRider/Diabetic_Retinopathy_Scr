function report = ag_e3_preflight(model, trainData, validationData, cfg, checkpointDirectory)
% AG_E3_PREFLIGHT Bounded real-image forward/backward/inference/serialization smoke test.

trainSubset = subsetData(trainData, stratifiedIndices(trainData.labels, trainData.classNames, 16));
validationSubset = subsetData(validationData, stratifiedIndices(validationData.labels, validationData.classNames, 16));
smokeCfg = cfg; smokeCfg.batchSize = 1;
[globalBatch, patchBatch, targets, labels] = ag_e3_minibatch(trainSubset, 1, smokeCfg, false, false);
assert(size(globalBatch, 1) == cfg.inputResolution && size(globalBatch, 2) == cfg.inputResolution && ...
    size(globalBatch, 3) == 3 && size(globalBatch, 4) == 1);
assert(size(patchBatch, 1) == cfg.patchSize(1) && size(patchBatch, 2) == cfg.patchSize(2) && ...
    size(patchBatch, 3) == 3 && size(patchBatch, 4) == cfg.numPatches);
assert(size(targets, 2) == numel(labels));
fprintf(['E3 datastore validation\n-----------------------\nGlobal predictors : %d observations\n' ...
    'Walker predictors : %d observations\nTargets           : %d observations\n' ...
    'Walker patches     : %d/image\nStatus             : PASS\n'], ...
    size(globalBatch, 4), size(patchBatch, 4) / cfg.numPatches, size(targets, 2), cfg.numPatches);

if cfg.useGPU
    globalBatch = gpuArray(globalBatch); patchBatch = gpuArray(patchBatch); targets = gpuArray(targets);
end
[loss, gradients] = dlfeval(@ag_e3_model_gradients, model, ...
    dlarray(globalBatch, 'SSCB'), dlarray(patchBatch, 'SSCB'), ...
    dlarray(targets, 'CB'), cfg, "B");
assert(isfinite(double(gather(extractdata(loss)))));
assert(allFinite(gradients.global) && allFinite(gradients.local) && allFinite(gradients.fusion));
[stageALoss, stageAGradients] = dlfeval(@ag_e3_model_gradients, model, ...
    dlarray(globalBatch, 'SSCB'), dlarray(patchBatch, 'SSCB'), ...
    dlarray(targets, 'CB'), cfg, "A");
assert(isfinite(double(gather(extractdata(stageALoss)))) && isempty(stageAGradients.global));
assert(allFinite(stageAGradients.local) && allFinite(stageAGradients.fusion));
[updatedFusion, ~] = sgdmupdate(model.fusionNet, gradients.fusion, [], cfg.learningRate, cfg.momentum);
before = gather(extractdata(model.fusionNet.Learnables.Value{1}));
after = gather(extractdata(updatedFusion.Learnables.Value{1}));
assert(any(before ~= after, 'all'));
fprintf(['[PASS] model forward\n[PASS] loss finite\n' ...
    '[PASS] Stage A gradients finite\n[PASS] Stage B gradients finite\n[PASS] optimizer step\n']);
if cfg.useGPU, fprintf('[PASS] GPU execution\n'); else, fprintf('[PASS] CPU execution (no compatible GPU detected)\n'); end

prediction = ag_e3_predict(model, validationSubset, smokeCfg, "full");
metrics = ag_evaluate_model_v3(prediction.scores.full, validationSubset.labels, validationSubset.classNames);
assert(isstruct(metrics) && isfield(metrics, 'balancedAccuracy'));
fprintf('[PASS] validation prediction\n[PASS] metrics struct\n');

if ~isfolder(checkpointDirectory), mkdir(checkpointDirectory); end
preflightPath = fullfile(checkpointDirectory, 'E3_preflight_serialization.mat');
config = cfg; %#ok<NASGU>
save(preflightPath, 'model', 'config', '-v7.3');
reloaded = load(preflightPath, 'model', 'config');
assert(isfield(reloaded.model, 'globalNet') && isa(reloaded.model.fusionNet, 'dlnetwork'));
delete(preflightPath);
fprintf('[PASS] checkpoint serialization\n');
report = struct('status', "PASS", 'sampleCount', numel(validationSubset.files), ...
    'loss', double(gather(extractdata(loss))), 'gpu', cfg.useGPU, ...
    'globalInputSize', size(globalBatch), 'patchInputSize', size(patchBatch));
end

function tf = allFinite(gradientTable)
tf = true;
for i = 1:height(gradientTable)
    value = gradientTable.Value{i};
    tf = tf && (isempty(value) || all(gather(isfinite(extractdata(value))), 'all'));
end
end

function indices = stratifiedIndices(labels, classNames, count)
indices = zeros(0, 1); perClass = ceil(count / numel(classNames));
for c = 1:numel(classNames)
    matches = find(string(labels) == classNames(c));
    indices = [indices; matches(1:min(perClass, numel(matches)))]; %#ok<AGROW>
end
indices = indices(1:min(count, numel(indices)));
end

function output = subsetData(data, indices)
output = data;
output.files = data.files(indices);
output.labels = data.labels(indices);
output.coords = data.coords(:, :, indices);
output.randomCoords = data.randomCoords(:, :, indices);
end
