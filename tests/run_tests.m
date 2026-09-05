% Synthetic unit checks only: no model, training data, or test images are loaded.
projectRoot = fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(projectRoot, 'src'));

classNames = ["Severe"; "No_DR"; "Mild"; "Proliferate_DR"; "Moderate"];
labels = ["Mild"; "Mild"; "No_DR"; "No_DR"; "Moderate"; "Moderate"; ...
    "Proliferate_DR"; "Proliferate_DR"; "Severe"; "Severe"];
pReferable = [0.1; 0.6; 0.2; 0.4; 0.9; 0.4; 0.8; 0.7; 0.6; 0.3];
scores = zeros(numel(labels), numel(classNames));
for row = 1:numel(labels)
    trueColumn = find(classNames == labels(row), 1);
    if ismember(labels(row), ["Moderate", "Proliferate_DR", "Severe"])
        scores(row, trueColumn) = pReferable(row);
        scores(row, classNames == "No_DR") = 1 - pReferable(row);
    else
        scores(row, trueColumn) = 1 - pReferable(row);
        scores(row, classNames == "Moderate") = pReferable(row);
    end
end

v3 = ag_evaluate_model_v3(scores, labels, classNames);
assert(abs(v3.threshold - 0.601) < 1e-12);
assert(v3.raw05.TP == 4 && v3.raw05.FN == 2 && v3.raw05.TN == 3 && v3.raw05.FP == 1);
assert(v3.moderateToNoDR == 1 && v3.moderateToMild == 0);
assert(abs(v3.moderateRecall - 0.5) < 1e-12 && isfinite(v3.qwk));

[xx, yy] = meshgrid(1:120, 1:100);
mask = (xx - 60).^2 + (yy - 50).^2 <= 40^2;
synthetic = zeros(100, 120, 3, 'uint8');
synthetic(repmat(mask, 1, 1, 3)) = 100;
[cropped, cropMeta] = cropFundusFOV(synthetic);
assert(cropMeta.success && size(cropped, 1) >= 80 && size(cropped, 2) >= 80);
squared = padToSquare(cropped);
assert(size(squared, 1) == size(squared, 2));
processed = preprocessFundus(synthetic, 64);
assert(isequal(size(processed), [64 64 3]) && isa(processed, 'single'));

disp('All synthetic MATLAB checks passed.');
