function [mask, metadata] = ag_e3_retina_mask(I)
% AG_E3_RETINA_MASK Segment a plausible retinal FOV using bounded fallback tiers.

I = im2single(I);
if size(I, 3) == 1, I = repmat(I, 1, 1, 3); end
if size(I, 3) > 3, I = I(:, :, 1:3); end
gray = rgb2gray(I);
validGray = gray(isfinite(gray));
positive = gray(gray > 0 & isfinite(gray));
if isempty(positive)
    error('ag:e3walker:EmptyImage', 'Cannot walk an empty retinal image.');
end
imageRange = prctile(validGray, 98) - prctile(validGray, 2);

% Tier 1: preserve the original adaptive grayscale detector.
tier1Threshold = max(0.015, min(0.15, 0.35 * graythresh(positive)));
tier1Mask = cleanMask(gray > tier1Threshold);
tier1Fraction = nnz(tier1Mask) / numel(tier1Mask);
if tier1Fraction > 0.98
    tier1Mask = true(size(gray));
    tier1Fraction = 1;
end
[tier1Plausible, tier1CenterDistance] = plausibleMask(tier1Mask, imageRange, true);
grayReady = false;
if tier1Plausible && tier1Fraction < 0.35
    [grayBackground, grayHigh, grayDynamicRange] = borderLevels(gray);
    grayThreshold = grayBackground + 0.04 * grayDynamicRange;
    grayMask = cleanMask(gray > grayThreshold);
    grayFraction = nnz(grayMask) / numel(grayMask);
    [grayPlausible, grayCenterDistance] = plausibleMask(grayMask, imageRange, false);
    grayReady = true;
    % A low-area Tier-1 component is incomplete when a conservative
    % border-relative mask recovers substantially more connected retina.
    if grayDynamicRange > 1e-4 && grayPlausible && ...
            grayFraction > 1.5 * tier1Fraction
        tier1Plausible = false;
    end
end
if tier1Plausible
    mask = tier1Mask;
    metadata = resultMetadata("adaptive-gray", tier1Threshold, tier1Fraction, ...
        tier1Threshold, tier1Fraction, NaN, NaN, NaN, tier1CenterDistance);
    return;
end

% Tier 2: estimate the background from the border and scale to this image.
if ~grayReady
    [grayBackground, grayHigh, grayDynamicRange] = borderLevels(gray);
    grayThreshold = grayBackground + 0.04 * grayDynamicRange;
    grayMask = cleanMask(gray > grayThreshold);
    grayFraction = nnz(grayMask) / numel(grayMask);
    [grayPlausible, grayCenterDistance] = plausibleMask(grayMask, imageRange, false);
end
if grayDynamicRange > 1e-4 && grayPlausible
    mask = grayMask;
    metadata = resultMetadata("border-relative-gray", grayThreshold, grayFraction, ...
        tier1Threshold, tier1Fraction, grayBackground, grayHigh, grayDynamicRange, ...
        grayCenterDistance);
    return;
end

% Tier 3: retain color-channel evidence lost by grayscale conversion.
rgbIntensity = max(I(:, :, 1:3), [], 3);
validRgb = rgbIntensity(isfinite(rgbIntensity));
rgbImageRange = prctile(validRgb, 98) - prctile(validRgb, 2);
[rgbBackground, rgbHigh, rgbDynamicRange] = borderLevels(rgbIntensity);
rgbThreshold = rgbBackground + 0.04 * rgbDynamicRange;
rgbMask = cleanMask(rgbIntensity > rgbThreshold);
rgbFraction = nnz(rgbMask) / numel(rgbMask);
[rgbPlausible, rgbCenterDistance] = plausibleMask(rgbMask, rgbImageRange, false);
if rgbDynamicRange > 1e-4 && rgbPlausible
    mask = rgbMask;
    metadata = resultMetadata("border-relative-rgb", rgbThreshold, rgbFraction, ...
        tier1Threshold, tier1Fraction, rgbBackground, rgbHigh, rgbDynamicRange, ...
        rgbCenterDistance);
    return;
end

error('ag:e3walker:InvalidFOV', ...
    ['No plausible retinal FOV. Tier 1 %.1f%% (threshold %.6f), ' ...
     'Tier 2 %.1f%% (threshold %.6f), Tier 3 %.1f%% (threshold %.6f).'], ...
    100 * tier1Fraction, tier1Threshold, 100 * grayFraction, grayThreshold, ...
    100 * rgbFraction, rgbThreshold);
end

function mask = cleanMask(mask)
radius = max(2, round(min(size(mask)) * 0.01));
mask = imclose(mask, strel("disk", radius, 0));
mask = imfill(mask, "holes");
mask = bwareafilt(mask, 1);
end

function [background, highLevel, dynamicRange] = borderLevels(intensity)
[h, w] = size(intensity);
borderWidth = max(2, round(0.03 * min(h, w)));
top = intensity(1:borderWidth, :);
bottom = intensity(end-borderWidth+1:end, :);
left = intensity(:, 1:borderWidth);
right = intensity(:, end-borderWidth+1:end);
borderPixels = [top(:); bottom(:); left(:); right(:)];
borderPixels = borderPixels(isfinite(borderPixels));
validPixels = intensity(isfinite(intensity));
background = median(borderPixels);
highLevel = prctile(validPixels, 98);
dynamicRange = highLevel - background;
end

function [plausible, centerDistance] = plausibleMask(mask, imageRange, allowFullFrame)
fraction = nnz(mask) / numel(mask);
center = round((size(mask) + 1) / 2);
if any(mask(:))
    distances = bwdist(mask);
    centerDistance = distances(center(1), center(2)) / min(size(mask));
else
    centerDistance = Inf;
end
upperBound = 0.98;
if allowFullFrame, upperBound = 1; end
plausible = imageRange > 1e-4 && fraction >= 0.15 && ...
    fraction <= upperBound && centerDistance <= 0.10;
end

function metadata = resultMetadata(method, threshold, fraction, tier1Threshold, ...
        tier1Fraction, background, highLevel, dynamicRange, centerDistance)
metadata = struct('method', method, 'threshold', threshold, 'fraction', fraction, ...
    'tier1Threshold', tier1Threshold, 'tier1Fraction', tier1Fraction, ...
    'background', background, 'highLevel', highLevel, ...
    'dynamicRange', dynamicRange, 'centerDistanceFraction', centerDistance);
end
