function diagnostics = ag_moderate_nodr_analysis(scores, labels, classNames, modelName)
% AG_MODERATE_NODR_ANALYSIS Analyze Moderate-vs-No_DR score separation.

if nargin < 4
    modelName = 'Model';
end
classNames = string(classNames(:));
labels = categorical(string(labels(:)), classNames, classNames);
moderateIdx = find(classNames == "Moderate", 1);
noDrIdx = find(classNames == "No_DR", 1);
if isempty(moderateIdx) || isempty(noDrIdx) || size(scores, 2) ~= numel(classNames)
    error('ag:analysis:InvalidClasses', 'Scores/classNames must include Moderate and No_DR columns.');
end

pModerate = scores(:, moderateIdx);
pNoDr = scores(:, noDrIdx);
moderateMask = labels == "Moderate";
noDrMask = labels == "No_DR";
subsetMask = moderateMask | noDrMask;
if ~any(moderateMask) || ~any(noDrMask)
    error('ag:analysis:MissingClass', 'Analysis needs both Moderate and No_DR cases.');
end

isModerate = moderateMask(subsetMask);
margin = pModerate(subsetMask) - pNoDr(subsetMask);
[~, ~, ~, aucModNo] = perfcurve(isModerate, margin, true);
meanPModMod = mean(pModerate(moderateMask));
medianPModMod = median(pModerate(moderateMask));
meanPNoDrMod = mean(pNoDr(moderateMask));
medianPNoDrMod = median(pNoDr(moderateMask));
meanPModNoDr = mean(pModerate(noDrMask));
meanPNoDrNoDr = mean(pNoDr(noDrMask));
suppressed = sum(pNoDr(moderateMask) > pModerate(moderateMask));
numModerate = sum(moderateMask);

fprintf('\n====================================================================\n');
fprintf('MODERATE DR vs No_DR REPRESENTATION ANALYSIS: %s\n', modelName);
fprintf('====================================================================\n');
fprintf('Binary Moderate vs No_DR ROC-AUC: %.4f\n\n', aucModNo);
fprintf('True Moderate Cases (N=%d):\n', numModerate);
fprintf('  Mean P(Moderate):        %.4f (Median: %.4f)\n', meanPModMod, medianPModMod);
fprintf('  Mean P(No_DR):           %.4f (Median: %.4f)\n', meanPNoDrMod, medianPNoDrMod);
fprintf('  Cases where P(No_DR) > P(Mod): %d / %d (%.2f%%)\n', ...
    suppressed, numModerate, 100 * suppressed / numModerate);
fprintf('\nTrue No_DR Cases (N=%d):\n', sum(noDrMask));
fprintf('  Mean P(Moderate):        %.4f\n', meanPModNoDr);
fprintf('  Mean P(No_DR):           %.4f\n', meanPNoDrNoDr);
fprintf('====================================================================\n\n');

diagnostics = struct( ...
    'modelName', modelName, 'aucModNo', aucModNo, ...
    'meanPMod_Mod', meanPModMod, 'medianPMod_Mod', medianPModMod, ...
    'meanPNoDR_Mod', meanPNoDrMod, 'medianPNoDR_Mod', medianPNoDrMod, ...
    'meanPMod_NoDR', meanPModNoDr, 'meanPNoDR_NoDR', meanPNoDrNoDr, ...
    'numModSuppressedByNoDR', suppressed, ...
    'pctModSuppressedByNoDR', 100 * suppressed / numModerate);
end
