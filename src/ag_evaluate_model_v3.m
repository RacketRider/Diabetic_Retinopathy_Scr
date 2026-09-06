function metrics = ag_evaluate_model_v3(scores, labels, classNames, pRefOverride, fixedThreshold, allowDiagnostic)
% AG_EVALUATE_MODEL_V3 Unified severity and clinical-boundary evaluation.

if nargin < 4
    pRefOverride = [];
end
if nargin < 5
    fixedThreshold = [];
end
if nargin < 6
    allowDiagnostic = false;
end
if isa(scores, 'dlarray')
    scores = extractdata(scores);
end
scores = double(gather(scores));
classNames = string(classNames(:));
labels = categorical(string(labels(:)), classNames, classNames);
required = ["No_DR"; "Mild"; "Moderate"; "Severe"; "Proliferate_DR"];
if size(scores, 1) ~= numel(labels) || size(scores, 2) ~= numel(classNames) || ...
        numel(classNames) ~= 5 || ~all(ismember(required, classNames)) || any(isundefined(labels))
    error('ag:v3evaluate:InvalidShape', 'Scores, labels, and the five-class schema do not align.');
end
if any(~isfinite(scores), 'all') || any(scores < 0, 'all') || any(abs(sum(scores, 2) - 1) > 1e-3)
    error('ag:v3evaluate:InvalidProbabilities', 'Severity scores must be finite row probabilities.');
end

referableClasses = ["Moderate", "Severe", "Proliferate_DR"];
referableColumns = ismember(classNames, referableClasses);
if isempty(pRefOverride)
    pRef = sum(scores(:, referableColumns), 2);
else
    pRef = double(gather(pRefOverride(:)));
end
if numel(pRef) ~= numel(labels) || any(~isfinite(pRef)) || any(pRef < 0 | pRef > 1)
    error('ag:v3evaluate:InvalidRDR', 'Referable probabilities must contain one finite [0,1] value per image.');
end
trueRef = ismember(string(labels), referableClasses);
hasBothBoundaryClasses = any(trueRef) && any(~trueRef);
if ~hasBothBoundaryClasses && ~allowDiagnostic
    error('ag:v3evaluate:MissingBoundaryClass', 'Both referable and non-referable samples are required.');
end

if hasBothBoundaryClasses
    [~, ~, ~, rocAuc] = perfcurve(trueRef, pRef, true);
    [~, ~, ~, prAuc] = perfcurve(trueRef, pRef, true, 'xCrit', 'reca', 'yCrit', 'prec');
else
    rocAuc = NaN; prAuc = NaN;
end
if isempty(fixedThreshold)
    [optimized, thresholdSweep] = ag_optimize_rdr_threshold(trueRef, pRef, 0.85);
    if ~optimized.found
        error('ag:v3evaluate:NoClinicalThreshold', 'No threshold in 0:0.001:1 satisfies specificity >= 0.85.');
    end
    selected = optimized;
else
    if ~isscalar(fixedThreshold) || ~isfinite(fixedThreshold) || fixedThreshold < 0 || fixedThreshold > 1
        error('ag:v3evaluate:InvalidFixedThreshold', 'Fixed threshold must be a finite scalar in [0,1].');
    end
    selected = operatingPoint(trueRef, pRef, fixedThreshold);
    selected.found = true;
    thresholdSweep = table();
end
raw05 = operatingPoint(trueRef, pRef, 0.5);

[~, argmax] = max(scores, [], 2);
predictedLabels = categorical(classNames(argmax), classNames, classNames);
classOrder = categorical(classNames, classNames, classNames);
confusionMatrix = confusionmat(labels, predictedLabels, 'Order', classOrder);
accuracy = mean(predictedLabels == labels);

support = zeros(5, 1); precision = NaN(5, 1); recall = NaN(5, 1); f1 = NaN(5, 1);
for c = 1:5
    truth = labels == classNames(c);
    prediction = predictedLabels == classNames(c);
    support(c) = sum(truth);
    precision(c) = safeRatio(sum(truth & prediction), sum(prediction));
    recall(c) = safeRatio(sum(truth & prediction), sum(truth));
    f1(c) = safeRatio(2 * precision(c) * recall(c), precision(c) + recall(c));
end
perClass = table(classNames, support, precision, recall, f1, ...
    'VariableNames', {'Class','Support','Precision','Recall','F1'});
macroF1 = mean(f1, 'omitnan');
qwk = quadraticWeightedKappa(confusionMatrix, classNames);

moderate = labels == "Moderate";
moderateIndex = find(classNames == "Moderate", 1);
noDrIndex = find(classNames == "No_DR", 1);
moderateToNoDR = sum(moderate & predictedLabels == "No_DR");
moderateToMild = sum(moderate & predictedLabels == "Mild");
moderateToSevere = sum(moderate & predictedLabels == "Severe");
moderateToPDR = sum(moderate & predictedLabels == "Proliferate_DR");
moderateToNonRef = moderateToNoDR + moderateToMild;
falseNegative = trueRef & pRef < selected.threshold;
fnModerate = sum(falseNegative & moderate);
fnSevere = sum(falseNegative & labels == "Severe");
fnProliferate = sum(falseNegative & labels == "Proliferate_DR");
moderateOrNoDR = moderate | labels == "No_DR";
if any(moderate) && any(labels == "No_DR")
    [~, ~, ~, moderateNoDrAuc] = perfcurve(moderate(moderateOrNoDR), ...
        scores(moderateOrNoDR, moderateIndex) - scores(moderateOrNoDR, noDrIndex), true);
else
    moderateNoDrAuc = NaN;
end

metrics = struct('threshold', selected.threshold, 'valAccuracy', accuracy, ...
    'overallAccuracy', accuracy, 'balancedAccuracy', mean(recall, 'omitnan'), ...
    'macroF1', macroF1, 'qwk', qwk, 'rocAuc', rocAuc, 'prAuc', prAuc, ...
    'sensitivity', selected.sensitivity, 'specificity', selected.specificity, ...
    'ppv', selected.ppv, 'npv', selected.npv, ...
    'clinicalAccuracy', (selected.TP + selected.TN) / numel(labels), ...
    'TP', selected.TP, 'TN', selected.TN, 'FP', selected.FP, 'FN', selected.FN, ...
    'moderateRecall', recall(moderateIndex), 'moderateToNoDR', moderateToNoDR, ...
    'moderateToMild', moderateToMild, 'moderateToNonRef', moderateToNonRef, ...
    'moderateToSevere', moderateToSevere, 'moderateToPDR', moderateToPDR, ...
    'fnModerate', fnModerate, 'fnSevere', fnSevere, 'fnProlif', fnProliferate, ...
    'modNoRocAuc', moderateNoDrAuc, 'perClass', perClass, ...
    'perClassRecall', recall, 'perClassPrecision', precision, 'confusionMatrix', confusionMatrix, ...
    'raw05', raw05, 'calibrated', selected, 'thresholdSweep', thresholdSweep, ...
    'pRef', pRef, 'predLabels', predictedLabels, 'scores', scores);

fprintf('\n=== V3 CLINICAL EVALUATION ===\n');
fprintf('Selected threshold: %.3f | Sens %.2f%% | Spec %.2f%% | FN %d | FP %d\n', ...
    selected.threshold, 100 * selected.sensitivity, 100 * selected.specificity, selected.FN, selected.FP);
fprintf('At threshold 0.5:   Sens %.2f%% | Spec %.2f%% | FN %d | FP %d\n', ...
    100 * raw05.sensitivity, 100 * raw05.specificity, raw05.FN, raw05.FP);
fprintf('RDR AUC %.4f | QWK %.4f | Macro-F1 %.4f | 5-class accuracy %.2f%%\n', ...
    rocAuc, qwk, macroF1, 100 * accuracy);
fprintf('Moderate recall %.2f%% | ->No_DR %d | ->Mild %d | ->non-ref %d\n', ...
    100 * recall(moderateIndex), moderateToNoDR, moderateToMild, moderateToNonRef);
fprintf('Referable-side Moderate errors: ->Severe %d | ->PDR %d (not referral failures)\n', ...
    moderateToSevere, moderateToPDR);
end

function point = operatingPoint(trueRef, pRef, threshold)
predicted = pRef >= threshold;
TP = sum(trueRef & predicted); TN = sum(~trueRef & ~predicted);
FP = sum(~trueRef & predicted); FN = sum(trueRef & ~predicted);
sensitivity = safeRatio(TP, TP + FN); specificity = safeRatio(TN, TN + FP);
ppv = safeRatio(TP, TP + FP); npv = safeRatio(TN, TN + FN);
point = struct('threshold', threshold, 'TP', TP, 'TN', TN, 'FP', FP, 'FN', FN, ...
    'sensitivity', sensitivity, 'specificity', specificity, 'ppv', ppv, 'npv', npv, ...
    'f1', safeRatio(2 * ppv * sensitivity, ppv + sensitivity));
end

function value = safeRatio(numerator, denominator)
if denominator == 0
    value = NaN;
else
    value = numerator / denominator;
end
end

function qwk = quadraticWeightedKappa(confusionMatrix, classNames)
ordinalNames = ["No_DR"; "Mild"; "Moderate"; "Severe"; "Proliferate_DR"];
[found, order] = ismember(ordinalNames, classNames);
if ~all(found)
    qwk = NaN;
    return;
end
observed = double(confusionMatrix(order, order));
count = sum(observed, 'all');
expected = sum(observed, 2) * sum(observed, 1) / max(count, 1);
[row, column] = ndgrid(0:4, 0:4);
weights = ((row - column) .^ 2) / 16;
denominator = sum(weights .* expected, 'all');
qwk = 1 - safeRatio(sum(weights .* observed, 'all'), denominator);
end
