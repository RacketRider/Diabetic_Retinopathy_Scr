function [Icrop, meta] = cropFundusFOV(I)
% CROPFUNDUSFOV Remove non-retinal borders while preserving the full FOV.

if size(I, 3) == 1
    I = repmat(I, 1, 1, 3);
elseif size(I, 3) > 3
    I = I(:, :, 1:3);
end
I = im2uint8(I);
[height, width, ~] = size(I);
brightness = max(I, [], 3);
nonzeroValues = double(brightness(brightness > 0));
meta = struct('success', false, 'reason', "", 'threshold', NaN, ...
    'foregroundFraction', NaN, 'originalSize', [height width], ...
    'cropBox', [1 1 width height], 'cropSize', size(I));

if isempty(nonzeroValues)
    Icrop = I;
    meta.reason = "empty_image";
    return;
end

% Dataset audit showed real backgrounds near zero but wide camera variation.
% A per-image low percentile keeps the threshold conservative.
threshold = max(5, min(25, 0.5 * prctile(nonzeroValues, 5)));
mask = brightness > threshold;
radius = max(3, round(min(height, width) * 0.01));
mask = imclose(mask, strel("disk", radius, 0));
mask = imfill(mask, "holes");
mask = bwareafilt(mask, 1);
foregroundFraction = nnz(mask) / numel(mask);
meta.threshold = threshold;
meta.foregroundFraction = foregroundFraction;

if foregroundFraction < 0.15 || foregroundFraction > 0.98
    Icrop = I;
    meta.reason = "invalid_mask";
    return;
end

stats = regionprops(mask, "BoundingBox");
if isempty(stats)
    Icrop = I;
    meta.reason = "no_component";
    return;
end

bbox = stats(1).BoundingBox;
margin = 0.02 * max(bbox(3), bbox(4));
x1 = max(1, floor(bbox(1) + 0.5 - margin));
y1 = max(1, floor(bbox(2) + 0.5 - margin));
x2 = min(width, ceil(bbox(1) + bbox(3) - 0.5 + margin));
y2 = min(height, ceil(bbox(2) + bbox(4) - 0.5 + margin));

% Reject pathological crops rather than risking retinal information loss.
if x2 <= x1 || y2 <= y1 || (x2 - x1 + 1) * (y2 - y1 + 1) < 0.2 * width * height
    Icrop = I;
    meta.reason = "unsafe_crop";
    return;
end

Icrop = I(y1:y2, x1:x2, :);
meta.success = true;
meta.reason = "ok";
meta.cropBox = [x1 y1 x2 y2];
meta.cropSize = size(Icrop);
end
