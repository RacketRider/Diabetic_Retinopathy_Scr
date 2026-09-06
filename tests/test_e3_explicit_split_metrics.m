% Synthetic fixed-threshold/diagnostic checks; no checkpoint or retinal image is loaded.
projectRoot = fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(projectRoot, 'src'));

classNames = ["Mild"; "Moderate"; "No_DR"; "Proliferate_DR"; "Severe"];
scores = [0.01 0.02 0.95 0.01 0.01; 0.05 0.10 0.75 0.05 0.05];
labels = ["No_DR"; "No_DR"];
metrics = ag_evaluate_model_v3(scores, labels, classNames, [], 0.061, true);

assert(metrics.threshold == 0.061);
assert(isempty(metrics.thresholdSweep));
assert(isnan(metrics.sensitivity) && metrics.specificity == 0.5);
assert(isnan(metrics.rocAuc) && isnan(metrics.prAuc) && isnan(metrics.modNoRocAuc));
assert(isequal(size(metrics.confusionMatrix), [5 5]));
assert(sum(metrics.confusionMatrix, 'all') == numel(labels));

disp('E3 explicit-split synthetic metric checks passed.');
