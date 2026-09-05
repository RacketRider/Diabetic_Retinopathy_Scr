function report = ag_visualize_e3_walker(data, cfg, outputDirectory, maxImages)
% AG_VISUALIZE_E3_WALKER Validate and render deterministic train-only walker examples.

if nargin < 4, maxImages = 16; end
if ~isfolder(outputDirectory), mkdir(outputDirectory); end
indices = stratifiedIndices(data.labels, data.classNames, maxImages);
minimumDistance = Inf; minimumCoverage = Inf;
for n = 1:numel(indices)
    i = indices(n);
    image = preprocessFundus(data.files{i}, cfg.inputResolution);
    walker = ag_e3_compute_walker(image, cfg, i);
    cachedCoords = double(data.coords(:, :, i));
    if ~isequal(cachedCoords, double(walker.coords))
        error('ag:e3visual:CacheMismatch', 'Walker cache mismatch for observation %d.', i);
    end
    linear = sub2ind(size(walker.fovMask), cachedCoords(:, 2), cachedCoords(:, 1));
    if ~all(walker.fovMask(linear))
        error('ag:e3visual:OutsideFOV', 'Walker center outside retinal FOV for observation %d.', i);
    end
    distances = pdist(cachedCoords);
    if ~isempty(distances), minimumDistance = min(minimumDistance, min(distances)); end
    coverage = patchCoverage(walker.fovMask, cachedCoords, cfg.patchSize);
    minimumCoverage = min(minimumCoverage, min(coverage));
    if min(coverage) < 0.9 || (~isempty(distances) && min(distances) < 0.9 * cfg.minWalkerSeparation)
        error('ag:e3visual:InvalidWalkerGeometry', ...
            'Walker preflight failed for observation %d (coverage %.2f, spacing %.1f).', ...
            i, min(coverage), min(distances));
    end
    saveOverlay(image, cachedCoords, walker.samplingType, cfg.patchSize, ...
        fullfile(outputDirectory, sprintf('%02d_walker.png', n)));
    if n <= min(4, numel(indices))
        patches = ag_e3_extract_patches(image, cachedCoords, cfg.patchSize);
        saveMontage(image, patches, walker.samplingType, ...
            fullfile(outputDirectory, sprintf('%02d_patch_montage.png', n)));
    end
end
report = struct('count', numel(indices), 'indices', indices, ...
    'minimumCenterDistance', minimumDistance, 'minimumPatchFOVCoverage', minimumCoverage, ...
    'status', "PASS");
fprintf('[PASS] walker visual preflight: %d images | min spacing %.1f px | min FOV coverage %.1f%%\n', ...
    report.count, report.minimumCenterDistance, 100 * report.minimumPatchFOVCoverage);
end

function indices = stratifiedIndices(labels, classNames, count)
indices = zeros(0, 1); perClass = ceil(count / numel(classNames));
for c = 1:numel(classNames)
    matches = find(string(labels) == classNames(c));
    indices = [indices; matches(1:min(perClass, numel(matches)))]; %#ok<AGROW>
end
indices = indices(1:min(count, numel(indices)));
end

function coverage = patchCoverage(mask, coords, patchSize)
coverage = zeros(size(coords, 1), 1);
for i = 1:size(coords, 1)
    patch = ag_e3_extract_patches(single(mask), coords(i, :), patchSize);
    coverage(i) = mean(patch(:, :, 1), 'all');
end
end

function saveOverlay(image, coords, types, patchSize, outputPath)
fig = figure('Visible', 'off', 'Position', [50 50 800 800]);
imshow(image); hold on;
plot(coords(:, 1), coords(:, 2), 'w-', 'LineWidth', 1);
for i = 1:size(coords, 1)
    color = [0.1 0.8 1.0]; if types(i) == "exploration", color = [1.0 0.75 0.1]; end
    rectangle('Position', [coords(i, 1) - patchSize(2)/2, coords(i, 2) - patchSize(1)/2, ...
        patchSize(2), patchSize(1)], 'EdgeColor', color, 'LineWidth', 1.2);
    text(coords(i, 1), coords(i, 2), sprintf('%d', i), 'Color', 'white', ...
        'FontWeight', 'bold', 'HorizontalAlignment', 'center');
end
title('E3 retina walker (line shows presentation order only)');
exportgraphics(fig, outputPath); close(fig);
end

function saveMontage(image, patches, types, outputPath)
fig = figure('Visible', 'off', 'Position', [20 20 2400 320]);
tiledlayout(1, size(patches, 4) + 1, 'Padding', 'compact', 'TileSpacing', 'compact');
nexttile; imshow(image); title('Global');
for i = 1:size(patches, 4)
    nexttile; imshow(patches(:, :, :, i)); title(sprintf('P%d %s', i, types(i)));
end
exportgraphics(fig, outputPath); close(fig);
end
