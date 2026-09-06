% Focused synthetic E3 component checks; no retinal data or training is loaded.
projectRoot = fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(projectRoot, 'src'));

cfg = struct('numPatches', 8, 'numVesselPatches', 6, ...
    'numExplorationPatches', 2, 'patchSize', [128 128], ...
    'walkerResolution', 448, 'vesselMethod', "fallback", ...
    'minWalkerSeparation', 96, 'seed', 42);
[xx, yy] = meshgrid(1:448, 1:448);
fov = (xx - 224).^2 + (yy - 224).^2 <= 205^2;
I = zeros(448, 448, 3, 'single');
I(:, :, 1) = 0.38 * fov;
I(:, :, 2) = 0.52 * fov;
I(:, :, 3) = 0.28 * fov;
I(abs(yy - (180 + 0.25 * (xx - 224))) < 3 & fov) = 0.05;
I(abs(yy - (270 - 0.2 * (xx - 224))) < 4 & fov) = 0.08;
I(245:255, 300:310, :) = 0.95;

w1 = ag_e3_compute_walker(I, cfg, 7);
w2 = ag_e3_compute_walker(I, cfg, 7);
assert(isequal(w1.coords, w2.coords));
assert(isequal(size(w1.coords), [8 2]));
assert(sum(w1.samplingType == "vessel") == 6);
assert(sum(w1.samplingType == "exploration") == 2);
assert(all(isfinite(w1.scores)) && all(w1.scores >= 0));
assert(all(w1.coords(:, 1) >= 1 & w1.coords(:, 1) <= 448));
assert(all(w1.coords(:, 2) >= 1 & w1.coords(:, 2) <= 448));
assert(all(w1.fovMask(sub2ind([448 448], w1.coords(:, 2), w1.coords(:, 1)))));
patches = ag_e3_extract_patches(I, w1.coords, cfg.patchSize);
assert(isequal(size(patches), [128 128 3 8]));
assert(all(isfinite(patches), 'all'));
assert(isequal(size(w1.randomCoords), [8 2]));
assert(w1.fovMethod == "adaptive-gray" && w1.fovFraction >= 0.15);

dark = zeros(448, 448, 3, 'single');
dark(:, :, 1) = 0.045 * fov;
dark(:, :, 2) = 0.060 * fov;
dark(:, :, 3) = 0.030 * fov;
dark(170:249, 184:263, :) = 0.9;
darkWalker = ag_e3_compute_walker(dark, cfg, 8);
assert(darkWalker.fovMetadata.tier1Fraction < 0.15);
assert(darkWalker.fovMethod == "border-relative-gray");
assert(darkWalker.fovFraction > 0.60 && darkWalker.fovMask(224, 224));

partial = zeros(448, 448, 3, 'single');
partial(:, :, 1) = 0.045 * fov;
partial(:, :, 2) = 0.060 * fov;
partial(:, :, 3) = 0.030 * fov;
brightHalf = fov & xx < 224;
partial(repmat(brightHalf, 1, 1, 3)) = 0.9;
partialWalker = ag_e3_compute_walker(partial, cfg, 9);
assert(partialWalker.fovMetadata.tier1Fraction >= 0.15 && ...
    partialWalker.fovMetadata.tier1Fraction < 0.35);
assert(partialWalker.fovMethod == "border-relative-gray");
assert(partialWalker.fovFraction > 1.5 * partialWalker.fovMetadata.tier1Fraction);

emptyRejected = false;
try
    ag_e3_compute_walker(zeros(448, 448, 3, 'single'), cfg, 10);
catch ME
    emptyRejected = strcmp(ME.identifier, 'ag:e3walker:EmptyImage');
end
assert(emptyRejected, 'An empty image must not produce a retinal mask.');

disp('E3 synthetic component checks passed.');
