function cache = ag_build_e3_walker_cache(files, cfg, cacheFile, splitName, labels)
% AG_BUILD_E3_WALKER_CACHE Cache coordinates/metadata only; source patches remain on disk.

files = cellstr(string(files(:)));
if nargin < 4 || strlength(string(splitName)) == 0
    [~, splitName] = fileparts(cacheFile);
end
if nargin < 5
    labels = strings(0, 1);
end
cacheDirectory = fileparts(cacheFile);
if ~isfolder(cacheDirectory), mkdir(cacheDirectory); end
saveEvery = max(1, cfg.cacheSaveFrequency);
cacheKey = makeCacheKey(cfg);
if isfile(cacheFile)
    saved = load(cacheFile, 'cache');
    if isfield(saved, 'cache') && isfield(saved.cache, 'cacheKey') && ...
            isequal(saved.cache.cacheKey, cacheKey) && isequal(saved.cache.files, files)
        cache = upgradeCache(saved.cache, numel(files));
        if cache.fovMaskVersion < 2
            cache = migrateFovMetadata(cache, files, cfg, cacheFile, saveEvery);
        end
        if all(cache.completed) && all(isfinite(cache.fovFraction))
            fprintf('[E3 CACHE] Reusing %d walker records: %s\n', numel(files), cacheFile);
            printSummary(cache);
            return;
        end
        if any(~cache.completed)
            fprintf('[E3 CACHE] Resuming at record %d/%d: %s\n', ...
                find(~cache.completed, 1), numel(files), cacheFile);
        end
    else
        fprintf('[E3 CACHE] Configuration/source list changed; rebuilding %s\n', cacheFile);
        cache = initializeCache(files, cfg, cacheKey);
    end
else
    cache = initializeCache(files, cfg, cacheKey);
end

for i = find(~cache.completed(:))'
    image = preprocessFundus(files{i}, cfg.walkerResolution);
    try
        walker = ag_e3_compute_walker(image, cfg, i);
    catch ME
        save(cacheFile, 'cache', '-v7.3');
        label = "unavailable";
        if numel(labels) >= i, label = string(labels(i)); end
        diagnosticPath = saveFailureDiagnostic(files{i}, image, cacheFile, ...
            splitName, i, label, ME);
        fprintf(2, '\n[E3 WALKER FAILURE]\n');
        fprintf(2, 'Split      : %s\n', string(splitName));
        fprintf(2, 'Index      : %d/%d\n', i, numel(files));
        fprintf(2, 'File       : %s\n', files{i});
        fprintf(2, 'Dimensions : %s\n', mat2str(size(image)));
        fprintf(2, 'Label      : %s\n', label);
        fprintf(2, 'Identifier : %s\n', ME.identifier);
        fprintf(2, 'Error      : %s\n', ME.message);
        fprintf(2, 'Diagnostic : %s\n', diagnosticPath);
        rethrow(ME);
    end
    cache.coords(:, :, i) = uint16(walker.coords);
    cache.randomCoords(:, :, i) = uint16(walker.randomCoords);
    cache.scores(:, i) = single(walker.scores);
    cache.samplingType(:, i) = walker.samplingType;
    cache.vesselMethod(i) = walker.vesselMethod;
    cache.fovMethod(i) = walker.fovMethod;
    cache.fovFraction(i) = walker.fovFraction;
    cache.fovThreshold(i) = walker.fovThreshold;
    cache.tier1FovFraction(i) = walker.fovMetadata.tier1Fraction;
    cache.tier1FovThreshold(i) = walker.fovMetadata.tier1Threshold;
    cache.completed(i) = true;
    if mod(i, saveEvery) == 0
        save(cacheFile, 'cache', '-v7.3');
        fprintf('[E3 CACHE] %d/%d coordinates cached.\n', i, numel(files));
    end
end

% Metadata did not exist in the original cache. Recompute masks only; retain
% every completed vessel/walker coordinate exactly as saved.
legacy = find(cache.completed & ~isfinite(cache.fovFraction));
if ~isempty(legacy)
    fprintf('[E3 CACHE] Backfilling FOV metadata for %d retained records.\n', numel(legacy));
    for n = 1:numel(legacy)
        i = legacy(n);
        image = preprocessFundus(files{i}, cfg.walkerResolution);
        [~, fov] = ag_e3_retina_mask(image);
        cache.fovMethod(i) = fov.method;
        cache.fovFraction(i) = fov.fraction;
        cache.fovThreshold(i) = fov.threshold;
        cache.tier1FovFraction(i) = fov.tier1Fraction;
        cache.tier1FovThreshold(i) = fov.tier1Threshold;
        if mod(n, saveEvery) == 0
            save(cacheFile, 'cache', '-v7.3');
        end
    end
end
save(cacheFile, 'cache', '-v7.3');
printSummary(cache);
end

function cache = initializeCache(files, cfg, cacheKey)
count = numel(files); patchCount = cfg.numPatches;
cache = struct('version', 2, 'cacheKey', cacheKey, 'files', {files}, ...
    'coords', zeros(patchCount, 2, count, 'uint16'), ...
    'randomCoords', zeros(patchCount, 2, count, 'uint16'), ...
    'scores', zeros(patchCount, count, 'single'), ...
    'samplingType', strings(patchCount, count), ...
    'vesselMethod', strings(count, 1), 'fovMethod', strings(count, 1), ...
    'fovFraction', NaN(count, 1), 'fovThreshold', NaN(count, 1), ...
    'tier1FovFraction', NaN(count, 1), 'tier1FovThreshold', NaN(count, 1), ...
    'fovMaskVersion', 2, 'completed', false(count, 1));
end

function cache = upgradeCache(cache, count)
cache.version = 2;
if ~isfield(cache, 'fovMethod'), cache.fovMethod = strings(count, 1); end
if ~isfield(cache, 'fovFraction'), cache.fovFraction = NaN(count, 1); end
if ~isfield(cache, 'fovThreshold'), cache.fovThreshold = NaN(count, 1); end
if ~isfield(cache, 'tier1FovFraction'), cache.tier1FovFraction = NaN(count, 1); end
if ~isfield(cache, 'tier1FovThreshold'), cache.tier1FovThreshold = NaN(count, 1); end
if ~isfield(cache, 'fovMaskVersion'), cache.fovMaskVersion = 1; end
end

function cache = migrateFovMetadata(cache, files, cfg, cacheFile, saveEvery)
completed = find(cache.completed);
invalidated = 0;
fprintf('[E3 CACHE] Validating FOV geometry for %d retained records.\n', numel(completed));
for n = 1:numel(completed)
    i = completed(n);
    oldMethod = cache.fovMethod(i);
    if strlength(oldMethod) == 0, oldMethod = "adaptive-gray"; end
    image = preprocessFundus(files{i}, cfg.walkerResolution);
    [~, fov] = ag_e3_retina_mask(image);
    if oldMethod ~= fov.method
        cache.completed(i) = false;
        invalidated = invalidated + 1;
    end
    cache.fovMethod(i) = fov.method;
    cache.fovFraction(i) = fov.fraction;
    cache.fovThreshold(i) = fov.threshold;
    cache.tier1FovFraction(i) = fov.tier1Fraction;
    cache.tier1FovThreshold(i) = fov.tier1Threshold;
    if mod(n, saveEvery) == 0
        save(cacheFile, 'cache', '-v7.3');
    end
end
cache.fovMaskVersion = 2;
save(cacheFile, 'cache', '-v7.3');
fprintf('[E3 CACHE] Retained %d coordinates; invalidated %d implausible masks.\n', ...
    numel(completed) - invalidated, invalidated);
end

function diagnosticPath = saveFailureDiagnostic(file, image, cacheFile, splitName, index, label, ME)
sourceImage = imread(file);
preprocessedImage = image;
grayscaleImage = rgb2gray(image);
positive = grayscaleImage(grayscaleImage > 0 & isfinite(grayscaleImage));
if isempty(positive)
    tier1Threshold = NaN;
    tier1Mask = false(size(grayscaleImage));
else
    tier1Threshold = max(0.015, min(0.15, 0.35 * graythresh(positive)));
    tier1Mask = grayscaleImage > tier1Threshold;
    radius = max(2, round(min(size(grayscaleImage)) * 0.01));
    tier1Mask = imclose(tier1Mask, strel("disk", radius, 0));
    tier1Mask = imfill(tier1Mask, "holes");
    tier1Mask = bwareafilt(tier1Mask, 1);
end
tier1FovFraction = nnz(tier1Mask) / numel(tier1Mask);
[histogramCounts, histogramEdges] = histcounts(grayscaleImage(isfinite(grayscaleImage)), 256);
diagnostic = struct('split', string(splitName), 'cacheIndex', index, ...
    'sourceFile', string(file), 'label', string(label), ...
    'preprocessedDimensions', size(image), 'exceptionIdentifier', string(ME.identifier), ...
    'exceptionMessage', string(ME.message), 'tier1Threshold', tier1Threshold, ...
    'tier1FovFraction', tier1FovFraction);
[~, stem] = fileparts(file);
diagnosticDirectory = fullfile(fileparts(cacheFile), 'diagnostics');
if ~isfolder(diagnosticDirectory), mkdir(diagnosticDirectory); end
diagnosticPath = fullfile(diagnosticDirectory, ...
    sprintf('%s_%06d_%s.mat', char(splitName), index, stem));
save(diagnosticPath, 'diagnostic', 'sourceImage', 'preprocessedImage', ...
    'grayscaleImage', 'tier1Mask', 'histogramCounts', 'histogramEdges', '-v7.3');
end

function printSummary(cache)
vesselMethods = unique(cache.vesselMethod);
fprintf('[E3 CACHE] Complete: %d records | vessel method(s): %s\n', ...
    numel(cache.files), strjoin(vesselMethods, ', '));
fovMethods = unique(cache.fovMethod);
for method = reshape(fovMethods, 1, [])
    fprintf('[E3 CACHE] FOV %-22s : %d\n', method, nnz(cache.fovMethod == method));
end
fprintf('[E3 CACHE] FOV fraction range: %.6f to %.6f\n', ...
    min(cache.fovFraction), max(cache.fovFraction));
end
function key = makeCacheKey(cfg)
if (string(cfg.vesselMethod) == "auto" || string(cfg.vesselMethod) == "fibermetric") && ...
        exist('fibermetric', 'file') == 2
    resolvedMethod = "fibermetric-green-dark";
else
    resolvedMethod = "green-multiscale-bottomhat";
end
identity = struct('version', 1, 'patchSize', cfg.patchSize, ...
    'numPatches', cfg.numPatches, 'numVesselPatches', cfg.numVesselPatches, ...
    'numExplorationPatches', cfg.numExplorationPatches, ...
    'walkerResolution', cfg.walkerResolution, 'vesselMethod', string(cfg.vesselMethod), ...
    'resolvedVesselMethod', resolvedMethod, ...
    'minWalkerSeparation', cfg.minWalkerSeparation, ...
    'preprocessingVersion', string(cfg.preprocessingVersion), 'seed', cfg.seed);
key = jsonencode(identity);
end
