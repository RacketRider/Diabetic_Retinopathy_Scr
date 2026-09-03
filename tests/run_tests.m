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

fixed = ag_evaluate_model(scores, labels, classNames, 0.5);
assert(fixed.TP == 4 && fixed.FN == 2 && fixed.TN == 3 && fixed.FP == 1);
assert(abs(fixed.sensitivity - 2/3) < 1e-12 && abs(fixed.specificity - 3/4) < 1e-12);
calibrated = ag_evaluate_model(scores, labels, classNames);
assert(abs(calibrated.threshold - 0.7) < 1e-12);
assert(calibrated.TP == 3 && calibrated.TN == 4 && calibrated.specificity == 1);

v3 = ag_evaluate_model_v3(scores, labels, classNames);
assert(abs(v3.threshold - 0.7) < 1e-12);
assert(v3.raw05.TP == 4 && v3.raw05.FN == 2 && v3.raw05.TN == 3 && v3.raw05.FP == 1);
assert(v3.moderateToNoDR == 1 && v3.moderateToMild == 0);
assert(abs(v3.moderateRecall - 0.5) < 1e-12 && isfinite(v3.qwk));

[decoded, pRdr] = ag_decode_v3_predictions(zeros(2, 6));
assert(all(abs(decoded - 0.2) < 1e-12, 'all') && all(pRdr == 0.5));
targets = zeros(6, 2, 'single');
targets(1, 1) = 1; targets(5, 2) = 1; targets(6, 2) = 1;
multitaskLoss = ag_v3_multitask_loss(dlarray(zeros(6, 2, 'single'), 'CB'), ...
    dlarray(targets, 'CB'), ones(5, 1, 'single'), 1);
assert(abs(double(extractdata(multitaskLoss)) - (log(5) + log(2))) < 1e-5);

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

Y = [0.8 0.2; 0.2 0.8];
T = eye(2);
expectedFocal = -((1 - 0.8)^2) * log(0.8);
assert(abs(ag_focal_loss(Y, T) - expectedFocal) < 1e-12);
assert(abs(ag_focal_loss(Y', T') - expectedFocal) < 1e-12);

disp('All synthetic MATLAB checks passed.');
