function diag = ag_moderate_nodr_analysis(valScores, YVal, classNames, modelName)
% AG_MODERATE_NODR_ANALYSIS Performs comprehensive representation and error analysis
% for Moderate DR vs No_DR on the validation set.

if nargin < 4
    modelName = 'Model';
end

if ~iscategorical(YVal)
    YVal = categorical(YVal, classNames);
end

% Index 2 = Moderate, Index 3 = No_DR
pMod = valScores(:, 2);
pNoDR = valScores(:, 3);
margin = pMod - pNoDR;

% Moderate vs No_DR subset
mask = (YVal == 'Moderate' | YVal == 'No_DR');
ySub = YVal(mask);
pModSub = pMod(mask);
pNoDRSub = pNoDR(mask);
marginSub = margin(mask);
isModSub = (ySub == 'Moderate');

% Binary ROC-AUC
[Xroc, Yroc, ~, aucModNo] = perfcurve(isModSub, marginSub, true);

% Distribution statistics
meanPMod_Mod = mean(pMod(YVal == 'Moderate'));
medianPMod_Mod = median(pMod(YVal == 'Moderate'));
meanPNoDR_Mod = mean(pNoDR(YVal == 'Moderate'));
medianPNoDR_Mod = median(pNoDR(YVal == 'Moderate'));

meanPMod_NoDR = mean(pMod(YVal == 'No_DR'));
meanPNoDR_NoDR = mean(pNoDR(YVal == 'No_DR'));

fprintf('\n====================================================================\n');
fprintf('MODERATE DR vs No_DR REPRESENTATION ANALYSIS: %s\n', modelName);
fprintf('====================================================================\n');
fprintf('Binary Moderate vs No_DR ROC-AUC: %.4f\n\n', aucModNo);
fprintf('True Moderate Cases (N=%d):\n', sum(YVal == 'Moderate'));
fprintf('  Mean P(Moderate):        %.4f (Median: %.4f)\n', meanPMod_Mod, medianPMod_Mod);
fprintf('  Mean P(No_DR):           %.4f (Median: %.4f)\n', meanPNoDR_Mod, medianPNoDR_Mod);
fprintf('  Cases where P(No_DR) > P(Mod): %d / %d (%.2f%%)\n', ...
    sum(pNoDR(YVal == 'Moderate') > pMod(YVal == 'Moderate')), sum(YVal == 'Moderate'), ...
    100 * sum(pNoDR(YVal == 'Moderate') > pMod(YVal == 'Moderate')) / sum(YVal == 'Moderate'));
fprintf('\nTrue No_DR Cases (N=%d):\n', sum(YVal == 'No_DR'));
fprintf('  Mean P(Moderate):        %.4f\n', meanPMod_NoDR);
fprintf('  Mean P(No_DR):           %.4f\n', meanPNoDR_NoDR);
fprintf('====================================================================\n\n');

diag = struct();
diag.modelName = modelName;
diag.aucModNo = aucModNo;
diag.meanPMod_Mod = meanPMod_Mod;
diag.medianPMod_Mod = medianPMod_Mod;
diag.meanPNoDR_Mod = meanPNoDR_Mod;
diag.medianPNoDR_Mod = medianPNoDR_Mod;
diag.meanPMod_NoDR = meanPMod_NoDR;
diag.meanPNoDR_NoDR = meanPNoDR_NoDR;
diag.numModSuppressedByNoDR = sum(pNoDR(YVal == 'Moderate') > pMod(YVal == 'Moderate'));
diag.pctModSuppressedByNoDR = 100 * diag.numModSuppressedByNoDR / sum(YVal == 'Moderate');
end
