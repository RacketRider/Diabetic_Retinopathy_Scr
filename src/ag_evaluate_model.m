function metrics = ag_evaluate_model(scores, labels, classNames, fixedThreshold)
% AG_EVALUATE_MODEL Evaluate five-class and referable-DR performance.
% With no fixedThreshold, calibrates the lowest-FN operating point whose
% specificity is at least 0.85. Pass a threshold to evaluate a frozen point.

if nargin < 4
    fixedThreshold = [];
end

if isa(scores, 'dlarray')
    scores = extractdata(scores);
end
scores = gather(scores);
classNames = string(classNames(:));
labels = categorical(string(labels(:)), classNames, classNames);
requiredClasses = ["Mild"; "Moderate"; "No_DR"; "Proliferate_DR"; "Severe"];

if ~isnumeric(scores) || ~ismatrix(scores) || size(scores, 1) ~= numel(labels)
    error('ag:evaluate:InvalidScores', 'Scores must be a numeric N-by-C matrix matching the N labels.');
end
if numel(classNames) ~= size(scores, 2) || numel(unique(classNames)) ~= numel(classNames)
    error('ag:evaluate:InvalidClasses', 'classNames must contain one unique name per score column.');
end
if ~all(ismember(requiredClasses, classNames)) || any(isundefined(labels))
    error('ag:evaluate:InvalidLabels', 'Labels/classNames must contain the five expected DR classes.');
end
if any(~isfinite(scores), 'all') || any(scores < 0, 'all') || any(abs(sum(scores, 2) - 1) > 1e-3)
    error('ag:evaluate:InvalidProbabilities', 'Scores must be finite, nonnegative probabilities whose rows sum to one.');
end
if ~isempty(fixedThreshold) && (~isscalar(fixedThreshold) || ~isfinite(fixedThreshold) || fixedThreshold < 0)
    error('ag:evaluate:InvalidThreshold', 'fixedThreshold must be a finite nonnegative scalar.');
end

moderateIdx = classNames == "Moderate";
noDrIdx = classNames == "No_DR";
referableIdx = ismember(classNames, ["Moderate", "Proliferate_DR", "Severe"]);
pRef = sum(scores(:, referableIdx), 2);
trueRef = ismember(string(labels), ["Moderate", "Proliferate_DR", "Severe"]);
numPositive = sum(trueRef);
numNegative = sum(~trueRef);
if numPositive == 0 || numNegative == 0
    error('ag:evaluate:MissingClinicalClass', 'Evaluation labels need both referable and non-referable cases.');
end

[~, ~, ~, rocAuc] = perfcurve(trueRef, pRef, true);
[~, ~, ~, prAuc] = perfcurve(trueRef, pRef, true, 'xCrit', 'reca', 'yCrit', 'prec');

if isempty(fixedThreshold)
    % O(N log N): evaluate each distinct operating point using prefix counts.
    [sortedScores, order] = sort(pRef, 'ascend');
    sortedTruth = trueRef(order);
    firstAtScore = find([true; diff(sortedScores) > 0]);
    thresholds = sortedScores(firstAtScore);
    positivePrefix = [0; cumsum(sortedTruth)];
    negativePrefix = [0; cumsum(~sortedTruth)];
    FN = positivePrefix(firstAtScore);
    TN = negativePrefix(firstAtScore);
    TP = numPositive - FN;
    FP = numNegative - TN;
    sensitivity = TP / numPositive;
    specificity = TN / numNegative;

    candidates = find(specificity >= 0.85);
    if isempty(candidates)
        % The all-negative point is the only guaranteed feasible fallback.
        bestThreshold = 1 + eps(1);
        bestTP = 0; bestTN = numNegative; bestFP = 0; bestFN = numPositive;
    else
        maxSensitivity = max(sensitivity(candidates));
        tied = candidates(sensitivity(candidates) == maxSensitivity);
        [~, bestTie] = max(specificity(tied));
        best = tied(bestTie);
        bestThreshold = thresholds(best);
        bestTP = TP(best); bestTN = TN(best); bestFP = FP(best); bestFN = FN(best);
    end
else
    bestThreshold = fixedThreshold;
    predictedRef = pRef >= bestThreshold;
    bestTP = sum(predictedRef & trueRef);
    bestTN = sum(~predictedRef & ~trueRef);
    bestFP = sum(predictedRef & ~trueRef);
    bestFN = sum(~predictedRef & trueRef);
end

sensitivity = bestTP / numPositive;
specificity = bestTN / numNegative;
ppv = bestTP / (bestTP + bestFP);
npv = bestTN / (bestTN + bestFN);
clinicalAccuracy = (bestTP + bestTN) / numel(labels);

[~, argmaxIdx] = max(scores, [], 2);
predictedLabels = categorical(classNames(argmaxIdx), classNames, classNames);
accuracy = mean(predictedLabels == labels);
perClassRecall = NaN(numel(classNames), 1);
for c = 1:numel(classNames)
    mask = labels == classNames(c);
    if any(mask)
        perClassRecall(c) = mean(predictedLabels(mask) == classNames(c));
    end
end

predictedRef = pRef >= bestThreshold;
falseNegative = ~predictedRef & trueRef;
fnModerate = sum(falseNegative & labels == "Moderate");
fnProlif = sum(falseNegative & labels == "Proliferate_DR");
fnSevere = sum(falseNegative & labels == "Severe");
modNoMask = labels == "Moderate" | labels == "No_DR";
modTrue = labels(modNoMask) == "Moderate";
modScore = scores(modNoMask, moderateIdx) - scores(modNoMask, noDrIdx);
[~, ~, ~, modNoRocAuc] = perfcurve(modTrue, modScore, true);
classOrder = categorical(classNames, classNames, classNames);
confusionMatrix = confusionmat(labels, predictedLabels, 'Order', classOrder);

metrics = struct( ...
    'threshold', bestThreshold, ...
    'valAccuracy', accuracy, ... % Backward-compatible field name.
    'rocAuc', single(rocAuc), ...
    'prAuc', single(prAuc), ...
    'sensitivity', sensitivity, ...
    'specificity', specificity, ...
    'ppv', ppv, ...
    'npv', npv, ...
    'clinicalAccuracy', clinicalAccuracy, ...
    'TP', bestTP, 'TN', bestTN, 'FP', bestFP, 'FN', bestFN, ...
    'perClassRecall', perClassRecall, ...
    'fnModerate', fnModerate, 'fnProlif', fnProlif, 'fnSevere', fnSevere, ...
    'modNoRocAuc', modNoRocAuc, ...
    'confusionMatrix', confusionMatrix, ...
    'pRef', pRef, 'predLabels', predictedLabels, 'scores', scores);

fprintf('\n=== EVALUATION RESULTS ===\n');
fprintf('Threshold:         %.4f\n', bestThreshold);
fprintf('Sensitivity:       %.2f%% (TP=%d, FN=%d)\n', sensitivity*100, bestTP, bestFN);
fprintf('Specificity:       %.2f%% (TN=%d, FP=%d)\n', specificity*100, bestTN, bestFP);
fprintf('ROC-AUC:           %.4f\n', rocAuc);
fprintf('PR-AUC:            %.4f\n', prAuc);
fprintf('PPV:               %.2f%%\n', ppv*100);
fprintf('NPV:               %.2f%%\n', npv*100);
fprintf('Clinical Accuracy: %.2f%%\n', clinicalAccuracy*100);
fprintf('5-Class Accuracy:  %.2f%%\n', accuracy*100);
fprintf('Mod vs No_DR AUC:  %.4f\n', modNoRocAuc);
fprintf('FN Breakdown:      Moderate=%d, Prolif=%d, Severe=%d\n', fnModerate, fnProlif, fnSevere);
fprintf('==========================\n\n');
end
