function patches = ag_e3_extract_patches(I, coords, patchSize)
% AG_E3_EXTRACT_PATCHES Crop fixed-size RGB patches around [x y] centers with zero padding.

I = im2single(I);
if size(I, 3) == 1, I = repmat(I, 1, 1, 3); end
if size(I, 3) > 3, I = I(:, :, 1:3); end
if size(coords, 2) ~= 2 || numel(patchSize) ~= 2
    error('ag:e3patch:InvalidInput', 'coords must be K-by-2 and patchSize must be [H W].');
end
patchHeight = patchSize(1); patchWidth = patchSize(2);
patches = zeros(patchHeight, patchWidth, 3, size(coords, 1), 'single');
for i = 1:size(coords, 1)
    x1 = round(coords(i, 1) - (patchWidth - 1) / 2);
    y1 = round(coords(i, 2) - (patchHeight - 1) / 2);
    x2 = x1 + patchWidth - 1; y2 = y1 + patchHeight - 1;
    sourceX1 = max(1, x1); sourceX2 = min(size(I, 2), x2);
    sourceY1 = max(1, y1); sourceY2 = min(size(I, 1), y2);
    if sourceX2 < sourceX1 || sourceY2 < sourceY1
        error('ag:e3patch:OutsideImage', 'Walker coordinate %d lies outside the image.', i);
    end
    destinationX1 = sourceX1 - x1 + 1;
    destinationY1 = sourceY1 - y1 + 1;
    destinationX2 = destinationX1 + sourceX2 - sourceX1;
    destinationY2 = destinationY1 + sourceY2 - sourceY1;
    patches(destinationY1:destinationY2, destinationX1:destinationX2, :, i) = ...
        I(sourceY1:sourceY2, sourceX1:sourceX2, :);
end
end
