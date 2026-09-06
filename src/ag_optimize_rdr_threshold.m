function [selected, sweep] = ag_optimize_rdr_threshold(trueRef, pRef, minSpecificity)
% AG_OPTIMIZE_RDR_THRESHOLD Dense clinical operating-point sweep.

if nargin < 3
    minSpecificity = 0.85;
end
trueRef = logical(trueRef(:));
pRef = double(gather(pRef(:)));
if numel(trueRef) ~= numel(pRef) || isempty(pRef) || any(~isfinite(pRef)) || any(pRef < 0 | pRef > 1)
    error('ag:threshold:InvalidInput', 'trueRef and finite probabilities in [0,1] must have equal nonzero length.');
end
if ~isscalar(minSpecificity) || minSpecificity < 0 || minSpecificity > 1
    error('ag:threshold:InvalidSpecificity', 'minSpecificity must be in [0,1].');
end

threshold = (0:0.001:1)';
TP = zeros(size(threshold)); TN = TP; FP = TP; FN = TP;
for i = 1:numel(threshold)
    predicted = pRef >= threshold(i);
    TP(i) = sum(trueRef & predicted);
    TN(i) = sum(~trueRef & ~predicted);
    FP(i) = sum(~trueRef & predicted);
    FN(i) = sum(trueRef & ~predicted);
end
sensitivity = TP ./ max(TP + FN, 1);
specificity = TN ./ max(TN + FP, 1);
PPV = TP ./ max(TP + FP, 1);
NPV = TN ./ max(TN + FN, 1);
F1 = 2 * PPV .* sensitivity ./ max(PPV + sensitivity, eps);
sweep = table(threshold, TP, TN, FP, FN, sensitivity, specificity, PPV, NPV, F1, ...
    'VariableNames', {'Threshold','TP','TN','FP','FN','Sensitivity','Specificity','PPV','NPV','F1'});

feasible = find(specificity >= minSpecificity);
if isempty(feasible)
    selected = struct('found', false, 'index', NaN, 'threshold', NaN, ...
        'TP', NaN, 'TN', NaN, 'FP', NaN, 'FN', NaN, ...
        'sensitivity', NaN, 'specificity', NaN, 'ppv', NaN, 'npv', NaN, 'f1', NaN);
    return;
end
bestSensitivity = max(sensitivity(feasible));
tied = feasible(sensitivity(feasible) == bestSensitivity);
[~, tieIndex] = max(specificity(tied));
best = tied(tieIndex);
selected = struct('found', true, 'index', best, 'threshold', threshold(best), ...
    'TP', TP(best), 'TN', TN(best), 'FP', FP(best), 'FN', FN(best), ...
    'sensitivity', sensitivity(best), 'specificity', specificity(best), ...
    'ppv', PPV(best), 'npv', NPV(best), 'f1', F1(best));
end
