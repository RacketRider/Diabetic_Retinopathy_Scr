function metrics = ag_evaluate_model(scores, YVal, classNames)
% AG_EVALUATE_MODEL Evaluates 5-class model on validation set according to the project protocol.
%
% Classes: Mild (1), Moderate (2), No_DR (3), Proliferate_DR (4), Severe (5)
% Referable: Moderate, Proliferate_DR, Severe (columns 2, 4, 5)

% Ensure categorical
if ~iscategorical(YVal)
    YVal = categorical(YVal, classNames);
end

% Compute Referable probability
pRef = sum(scores(:, [2, 4, 5]), 2);
trueRef = (YVal == 'Moderate') | (YVal == 'Proliferate_DR') | (YVal == 'Severe');

% Compute ROC-AUC and PR-AUC
[~, ~, ~, roc_auc] = perfcurve(trueRef, pRef, true);
[~, ~, ~, pr_auc] = perfcurve(trueRef, pRef, true, 'xCrit', 'reca', 'yCrit', 'prec');

% Find optimal threshold: maximize sensitivity subject to specificity >= 0.85
uniqueTh = sort(unique(pRef));
% Sample if too many points, but 5270 points is very fast
bestTh = 0.5;
maxSens = -1;
bestSpec = -1;
bestTP = 0; bestTN = 0; bestFP = 0; bestFN = 0;

for i = 1:length(uniqueTh)
    th = uniqueTh(i);
    predRef = (pRef >= th);
    TP = sum(predRef & trueRef);
    TN = sum(~predRef & ~trueRef);
    FP = sum(predRef & ~trueRef);
    FN = sum(~predRef & trueRef);
    
    spec = TN / (TN + FP);
    sens = TP / (TP + FN);
    
    if spec >= 0.8500
        if (sens > maxSens) || (abs(sens - maxSens) < 1e-6 && spec > bestSpec)
            maxSens = sens;
            bestSpec = spec;
            bestTh = th;
            bestTP = TP;
            bestTN = TN;
            bestFP = FP;
            bestFN = FN;
        end
    end
end

% If none met spec >= 0.85, take the one closest to 0.85
if maxSens < 0
    [~, idx] = min(abs(spec - 0.8500));
    bestTh = uniqueTh(idx);
    predRef = (pRef >= bestTh);
    bestTP = sum(predRef & trueRef);
    bestTN = sum(~predRef & ~trueRef);
    bestFP = sum(predRef & ~trueRef);
    bestFN = sum(~predRef & trueRef);
    maxSens = bestTP / (bestTP + bestFN);
    bestSpec = bestTN / (bestTN + bestFP);
end

% Operating point metrics
sens = maxSens;
spec = bestSpec;
ppv = bestTP / (bestTP + bestFP);
npv = bestTN / (bestTN + bestFN);
clinicalAcc = (bestTP + bestTN) / (bestTP + bestTN + bestFP + bestFN);

% 5-class argmax accuracy
[~, argmaxIdx] = max(scores, [], 2);
predLabels = categorical(classNames(argmaxIdx), classNames);
valAcc = mean(predLabels == YVal);

% Per-class recall
classes = categories(YVal);
perClassRecall = zeros(length(classes), 1);
for c = 1:length(classes)
    mask = (YVal == classes{c});
    perClassRecall(c) = mean(predLabels(mask) == classes{c});
end

% Clinical false negative breakdown by true class
predRefFinal = (pRef >= bestTh);
fnMask = (~predRefFinal & trueRef);
fnModerate = sum(fnMask & (YVal == 'Moderate'));
fnProlif = sum(fnMask & (YVal == 'Proliferate_DR'));
fnSevere = sum(fnMask & (YVal == 'Severe'));

% Moderate vs No_DR binary ROC-AUC
modNoMask = (YVal == 'Moderate' | YVal == 'No_DR');
modTrue = (YVal(modNoMask) == 'Moderate');
modScore = scores(modNoMask, 2) - scores(modNoMask, 3);
[~, ~, ~, modNoRocAuc] = perfcurve(modTrue, modScore, true);

% 5-class confusion matrix
cm = confusionmat(YVal, predLabels, 'Order', classNames);

% Package metrics
metrics = struct();
metrics.threshold = bestTh;
metrics.valAccuracy = valAcc;
metrics.rocAuc = single(roc_auc);
metrics.prAuc = single(pr_auc);
metrics.sensitivity = sens;
metrics.specificity = spec;
metrics.ppv = ppv;
metrics.npv = npv;
metrics.clinicalAccuracy = clinicalAcc;
metrics.TP = bestTP;
metrics.TN = bestTN;
metrics.FP = bestFP;
metrics.FN = bestFN;
metrics.perClassRecall = perClassRecall;
metrics.fnModerate = fnModerate;
metrics.fnProlif = fnProlif;
metrics.fnSevere = fnSevere;
metrics.modNoRocAuc = modNoRocAuc;
metrics.confusionMatrix = cm;
metrics.pRef = pRef;
metrics.predLabels = predLabels;
metrics.scores = scores;

fprintf('\n=== VALIDATION EVALUATION RESULTS ===\n');
fprintf('Threshold:         %.4f\n', bestTh);
fprintf('Sensitivity:       %.2f%% (TP=%d, FN=%d)\n', sens*100, bestTP, bestFN);
fprintf('Specificity:       %.2f%% (TN=%d, FP=%d)\n', spec*100, bestTN, bestFP);
fprintf('ROC-AUC:           %.4f\n', roc_auc);
fprintf('PR-AUC:            %.4f\n', pr_auc);
fprintf('PPV:               %.2f%%\n', ppv*100);
fprintf('NPV:               %.2f%%\n', npv*100);
fprintf('Clinical Accuracy: %.2f%%\n', clinicalAcc*100);
fprintf('5-Class Accuracy:  %.2f%%\n', valAcc*100);
fprintf('Mod vs No_DR AUC:  %.4f\n', modNoRocAuc);
fprintf('FN Breakdown:      Moderate=%d, Prolif=%d, Severe=%d\n', fnModerate, fnProlif, fnSevere);
fprintf('=====================================\n\n');
end
