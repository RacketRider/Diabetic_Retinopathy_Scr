function prediction = ag_e3_predict(model, data, cfg, modes)
% AG_E3_PREDICT Run E3 inference and optional inference-only branch ablations.

if nargin < 4, modes = ["full" "globalOnly"]; end
modes = string(modes(:));
validModes = ["full" "globalOnly" "random"];
if any(~ismember(modes, validModes))
    error('ag:e3predict:InvalidMode', 'Modes must be full, globalOnly, or random.');
end
count = numel(data.files);
for mode = modes'
    scoreByMode.(char(mode)) = zeros(count, numel(data.classNames), 'single'); %#ok<AGROW>
end
fullSeconds = 0;
for first = 1:cfg.batchSize:count
    indices = first:min(first + cfg.batchSize - 1, count);
    [globalBatch, patchBatch] = ag_e3_minibatch(data, indices, cfg, false, false);
    [dlGlobal, dlPatches] = toDevice(globalBatch, patchBatch, cfg.useGPU);
    synchronize(cfg.useGPU); timer = tic;
    globalFeatures = predict(model.globalNet, dlGlobal);
    fullLocal = [];
    if any(modes == "full")
        fullLocal = aggregateLocal(predict(model.localNet, dlPatches), cfg.numPatches, numel(indices));
        logits = fusedPredict(model, globalFeatures, fullLocal, numel(indices));
        scoreByMode.full(indices, :) = rowProbabilities(logits);
    end
    synchronize(cfg.useGPU); fullSeconds = fullSeconds + toc(timer);
    if any(modes == "globalOnly")
        if isempty(fullLocal)
            localDim = model.localFeatureDim;
            zeroLocal = zeros(localDim, numel(indices), 'single');
            if cfg.useGPU, zeroLocal = gpuArray(zeroLocal); end
            zeroLocal = dlarray(zeroLocal, 'CB');
        else
            zeroLocal = fullLocal * 0;
        end
        scoreByMode.globalOnly(indices, :) = rowProbabilities( ...
            fusedPredict(model, globalFeatures, zeroLocal, numel(indices)));
    end
    if any(modes == "random")
        [~, randomPatchBatch] = ag_e3_minibatch(data, indices, cfg, false, true);
        if cfg.useGPU, randomPatchBatch = gpuArray(randomPatchBatch); end
        randomLocal = aggregateLocal(predict(model.localNet, ...
            dlarray(randomPatchBatch, 'SSCB')), cfg.numPatches, numel(indices));
        scoreByMode.random(indices, :) = rowProbabilities( ...
            fusedPredict(model, globalFeatures, randomLocal, numel(indices)));
    end
    batchNumber = ceil(first / cfg.batchSize);
    if mod(batchNumber, 100) == 0 || indices(end) == count
        fprintf('[E3 INFERENCE] %d/%d\n', indices(end), count);
    end
end
prediction = struct('scores', scoreByMode, 'labels', data.labels, ...
    'meanInferenceTime', fullSeconds / count);
end

function [dlGlobal, dlPatches] = toDevice(globalBatch, patchBatch, useGPU)
if useGPU
    globalBatch = gpuArray(globalBatch); patchBatch = gpuArray(patchBatch);
end
dlGlobal = dlarray(globalBatch, 'SSCB');
dlPatches = dlarray(patchBatch, 'SSCB');
end

function local = aggregateLocal(features, patchCount, batchSize)
features = reshape(stripdims(features), [], patchCount, batchSize);
local = dlarray(mean(features, 2), 'CB');
end

function logits = fusedPredict(model, globalFeatures, localFeatures, batchSize)
globalFeatures = dlarray(reshape(stripdims(globalFeatures), [], batchSize), 'CB');
logits = predict(model.fusionNet, cat(1, globalFeatures, localFeatures));
end

function scores = rowProbabilities(logits)
scores = gather(extractdata(softmax(logits)))';
end

function synchronize(useGPU)
if useGPU, wait(gpuDevice); end
end
