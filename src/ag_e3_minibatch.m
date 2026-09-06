function [globalBatch, patchBatch, targets, batchLabels] = ag_e3_minibatch(data, indices, cfg, augment, useRandom)
% AG_E3_MINIBATCH Read aligned global images, walker patches, and one-hot targets.

if nargin < 4, augment = false; end
if nargin < 5, useRandom = false; end
indices = indices(:)';
batchSize = numel(indices); patchCount = cfg.numPatches;
globalBatch = zeros(cfg.inputResolution, cfg.inputResolution, 3, batchSize, 'single');
patches = zeros(cfg.patchSize(1), cfg.patchSize(2), 3, patchCount, batchSize, 'single');
targets = zeros(numel(data.classNames), batchSize, 'single');
batchLabels = data.labels(indices);
for b = 1:batchSize
    sourceIndex = indices(b);
    image = preprocessFundus(data.files{sourceIndex}, cfg.inputResolution);
    if useRandom
        coords = double(data.randomCoords(:, :, sourceIndex));
    else
        coords = double(data.coords(:, :, sourceIndex));
    end
    if augment && rand() < 0.5
        image = fliplr(image);
        coords(:, 1) = cfg.inputResolution + 1 - coords(:, 1);
    end
    if augment
        gain = 0.9 + 0.2 * rand();
        bias = -0.03 + 0.06 * rand();
        image = min(max(image * gain + bias, 0), 1);
    end
    globalBatch(:, :, :, b) = image;
    patches(:, :, :, :, b) = ag_e3_extract_patches(image, coords, cfg.patchSize);
    classIndex = find(data.classNames == string(batchLabels(b)), 1);
    if isempty(classIndex)
        error('ag:e3data:UnknownLabel', 'Unknown label at observation %d.', sourceIndex);
    end
    targets(classIndex, b) = 1;
end
patchBatch = reshape(patches, cfg.patchSize(1), cfg.patchSize(2), 3, patchCount * batchSize);
if size(globalBatch, 4) ~= batchSize || size(patchBatch, 4) ~= patchCount * batchSize || ...
        size(targets, 2) ~= batchSize || any(sum(targets, 1) ~= 1)
    error('ag:e3data:Alignment', 'Global/local/target minibatch alignment failed.');
end
end
