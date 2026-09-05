function walker = ag_e3_compute_walker(I, cfg, sampleIndex)
% AG_E3_COMPUTE_WALKER Deterministic vessel-guided and exploratory retinal sampling.

if nargin < 3, sampleIndex = 1; end
I = im2single(I);
if size(I, 3) == 1, I = repmat(I, 1, 1, 3); end
if size(I, 3) > 3, I = I(:, :, 1:3); end
if size(I, 1) ~= cfg.walkerResolution || size(I, 2) ~= cfg.walkerResolution
    I = imresize(I, [cfg.walkerResolution cfg.walkerResolution]);
end
validateConfig(cfg);
[fovMask, fovMetadata] = ag_e3_retina_mask(I);
margin = max(4, ceil(max(cfg.patchSize) / 2));
safeMask = imerode(fovMask, strel("disk", margin, 0));
if nnz(safeMask) < 20 * cfg.numPatches
    safeMask = imerode(fovMask, strel("disk", max(2, round(margin / 2)), 0));
end
if nnz(safeMask) < cfg.numPatches
    safeMask = fovMask;
end

[vesselResponse, vesselMethod] = vesselMap(I, fovMask, cfg.vesselMethod);
explorationResponse = lesionCandidateMap(I, fovMask);
separation = max(1, cfg.minWalkerSeparation);
[vesselCoords, vesselScores] = spatialMaxima(vesselResponse, safeMask, ...
    cfg.numVesselPatches, separation, zeros(0, 2));
[exploreCoords, exploreScores] = spatialMaxima(explorationResponse, safeMask, ...
    cfg.numExplorationPatches, separation, vesselCoords);
coords = [vesselCoords; exploreCoords];
scores = [vesselScores; exploreScores];
[coords, scores] = fillDistributed(coords, scores, safeMask, vesselResponse + explorationResponse, ...
    cfg.numPatches, max(1, round(separation / 2)));
if size(coords, 1) ~= cfg.numPatches
    error('ag:e3walker:InsufficientLocations', ...
        'Walker produced %d of %d required locations.', size(coords, 1), cfg.numPatches);
end
samplingType = [repmat("vessel", cfg.numVesselPatches, 1); ...
    repmat("exploration", cfg.numExplorationPatches, 1)];
if numel(samplingType) < cfg.numPatches
    samplingType(end + 1:cfg.numPatches, 1) = "fallback";
end
randomCoords = randomDistributed(safeMask, cfg.numPatches, separation, cfg.seed + sampleIndex);
walker = struct('coords', round(coords), 'scores', scores, ...
    'samplingType', samplingType, 'randomCoords', round(randomCoords), ...
    'vesselMethod', vesselMethod, 'fovMask', fovMask, ...
    'fovMethod', fovMetadata.method, 'fovFraction', fovMetadata.fraction, ...
    'fovThreshold', fovMetadata.threshold, 'fovMetadata', fovMetadata);
end

function validateConfig(cfg)
required = {'numPatches','numVesselPatches','numExplorationPatches','patchSize', ...
    'walkerResolution','vesselMethod','minWalkerSeparation','seed'};
if ~all(isfield(cfg, required)) || cfg.numPatches < 1 || ...
        cfg.numVesselPatches + cfg.numExplorationPatches ~= cfg.numPatches || ...
        numel(cfg.patchSize) ~= 2 || any(cfg.patchSize < 8)
    error('ag:e3walker:InvalidConfig', 'Invalid E3 walker configuration.');
end
end

function [response, method] = vesselMap(I, fovMask, requested)
G = adapthisteq(I(:, :, 2), 'ClipLimit', 0.01);
requested = lower(string(requested));
useFiber = requested == "fibermetric" || requested == "auto";
if useFiber && exist('fibermetric', 'file') == 2
    try
        response = fibermetric(G, [1 2 3 4 6], ObjectPolarity="dark");
        method = "fibermetric-green-dark";
    catch ME
        warning('ag:e3walker:FibermetricFallback', ...
            'fibermetric failed (%s); using deterministic bottom-hat fallback.', ME.message);
        [response, method] = fallbackVessels(G);
    end
else
    [response, method] = fallbackVessels(G);
end
response(~fovMask) = 0;
response = normalizeMap(response);
end

function [response, method] = fallbackVessels(G)
r1 = max(2, round(min(size(G)) / 90));
r2 = max(r1 + 2, round(min(size(G)) / 50));
response = max(imbothat(G, strel("disk", r1, 0)), ...
    imbothat(G, strel("disk", r2, 0)));
response = imgaussfilt(response, 0.8);
method = "green-multiscale-bottomhat";
end

function response = lesionCandidateMap(I, fovMask)
gray = rgb2gray(I);
r1 = max(3, round(min(size(gray)) / 75));
r2 = max(r1 + 2, round(min(size(gray)) / 40));
bright = max(imtophat(gray, strel("disk", r1, 0)), ...
    imtophat(gray, strel("disk", r2, 0)));
dark = max(imbothat(gray, strel("disk", r1, 0)), ...
    imbothat(gray, strel("disk", r2, 0)));
contrast = stdfilt(gray, true(5));
response = normalizeMap(bright) + normalizeMap(dark) + 0.5 * normalizeMap(contrast);
response(~fovMask) = 0;
response = normalizeMap(response);
end

function output = normalizeMap(input)
input = single(input);
lo = min(input(:)); hi = max(input(:));
if ~isfinite(lo) || ~isfinite(hi) || hi <= lo
    output = zeros(size(input), 'single');
else
    output = (input - lo) ./ (hi - lo);
end
end

function [coords, scores] = spatialMaxima(response, candidateMask, count, separation, existing)
coords = zeros(0, 2); scores = zeros(0, 1, 'single');
available = candidateMask;
available = suppressAround(available, existing, separation);
for i = 1:count
    values = response;
    values(~available) = -Inf;
    [score, index] = max(values(:));
    if ~isfinite(score), break; end
    [y, x] = ind2sub(size(response), index);
    coords(end + 1, :) = [x y]; %#ok<AGROW>
    scores(end + 1, 1) = score; %#ok<AGROW>
    available = suppressAround(available, [x y], separation);
end
end

function mask = suppressAround(mask, coords, radius)
if isempty(coords), return; end
[xx, yy] = meshgrid(1:size(mask, 2), 1:size(mask, 1));
for i = 1:size(coords, 1)
    mask((xx - coords(i, 1)).^2 + (yy - coords(i, 2)).^2 < radius^2) = false;
end
end

function [coords, scores] = fillDistributed(coords, scores, mask, response, count, separation)
while size(coords, 1) < count
    available = suppressAround(mask, coords, separation);
    candidates = find(available);
    if isempty(candidates), candidates = find(mask); end
    if isempty(candidates), break; end
    if isempty(coords)
        values = response(candidates);
    else
        [cy, cx] = ind2sub(size(mask), candidates);
        distances = Inf(numel(candidates), 1);
        for i = 1:size(coords, 1)
            distances = min(distances, hypot(cx - coords(i, 1), cy - coords(i, 2)));
        end
        values = distances + double(response(candidates));
    end
    [~, best] = max(values);
    [y, x] = ind2sub(size(mask), candidates(best));
    coords(end + 1, :) = [x y]; %#ok<AGROW>
    scores(end + 1, 1) = response(y, x); %#ok<AGROW>
end
end

function coords = randomDistributed(mask, count, separation, seed)
% RANDOMDISTRIBUTED
% Deterministically choose random retinal control locations.
%
% The requested separation is preferred, but unlike the real E3 walker,
% the random-patch ABALATION must never abort the experiment merely because
% a small/irregular retinal FOV cannot geometrically accommodate all points.
%
% Separation is therefore relaxed progressively for the random control only.

    [rows, cols] = find(mask);

    if numel(rows) < count
        error('ag:e3walker:InsufficientRandomMask', ...
            'Random retinal mask contains only %d valid pixels for %d patches.', ...
            numel(rows), count);
    end

    candidates = [cols, rows];

    % Preserve caller RNG state.
    previousState = rng;
    cleanupObj = onCleanup(@() rng(previousState));

    rng(seed, 'twister');

    % Use the same randomized candidate ordering during every relaxation.
    order = randperm(size(candidates, 1));
    candidates = candidates(order, :);

    % Prefer the configured separation, but progressively relax it.
    separationSchedule = unique(round([ ...
        separation, ...
        0.90 * separation, ...
        0.80 * separation, ...
        0.70 * separation, ...
        0.60 * separation, ...
        0.50 * separation, ...
        0]), 'stable');

    for currentSeparation = separationSchedule

        selected = zeros(count, 2);
        selectedCount = 0;
        minDistanceSquared = currentSeparation ^ 2;

        for i = 1:size(candidates, 1)

            candidate = candidates(i, :);

            if selectedCount == 0
                acceptable = true;
            else
                delta = selected(1:selectedCount, :) - candidate;
                distancesSquared = sum(delta.^2, 2);

                acceptable = all(distancesSquared >= minDistanceSquared);
            end

            if acceptable
                selectedCount = selectedCount + 1;
                selected(selectedCount, :) = candidate;

                if selectedCount == count
                    coords = selected;

                    if currentSeparation < separation
                        fprintf(['[E3 RANDOM CONTROL] Requested spacing %.1f px ' ...
                            'relaxed to %.1f px for this retinal FOV.\n'], ...
                            separation, currentSeparation);
                    end

                    return;
                end
            end
        end
    end

    % This should only be reachable if mask/candidate data are corrupted.
    error('ag:e3walker:RandomControlFailure', ...
        'Unable to choose %d unique retinal control locations.', count);
end
