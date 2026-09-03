% Deep diagnostic of Moderate-vs-No_DR representation for V4.1, V4.2, and V4.3.
clear; clc;
projectRoot = fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(projectRoot, 'src'));
paths = ag_project_paths();

baseline = load(fullfile(paths.models, 'V4_1_base.mat'), 'valScoresV41', 'YValV41', 'classNamesV4');
v42 = load(fullfile(paths.results, 'AG_V4_2_LR5e5_results.mat'), 'valScores', 'YVal', 'classNames');
v43 = load(fullfile(paths.results, 'AG_V4_3_LR2e5_results.mat'), 'valScores', 'YVal', 'classNames');
if ~isequal(string(baseline.YValV41), string(v42.YVal), string(v43.YVal))
    error('ag:diagnostic:SplitMismatch', 'Validation labels/order differ across experiment artifacts.');
end
if ~isequal(string(baseline.classNamesV4), string(v42.classNames), string(v43.classNames))
    error('ag:diagnostic:ClassMismatch', 'Score-column class order differs across experiment artifacts.');
end

labels = categorical(string(v42.YVal), string(v42.classNames), string(v42.classNames));
classNames = string(v42.classNames(:));
moderateIdx = find(classNames == "Moderate", 1);
noDrIdx = find(classNames == "No_DR", 1);
moderateMask = labels == "Moderate";
noDrMask = labels == "No_DR";
modelNames = ["V4.1 Baseline", "AG_V4_2_LR5e5", "AG_V4_3_LR2e5"];
thresholds = [0.1460, 0.1413, 0.1290];
allScores = {baseline.valScoresV41, v42.valScores, v43.valScores};
referableIdx = find(ismember(classNames, ["Moderate", "Proliferate_DR", "Severe"]));

fprintf('=== DEEP MODERATE DR ERROR DIAGNOSTICS (VALIDATION N=%d Moderate) ===\n\n', sum(moderateMask));
for model = 1:numel(allScores)
    scores = allScores{model};
    pModerate = scores(moderateMask, moderateIdx);
    pNoDr = scores(moderateMask, noDrIdx);
    pReferable = sum(scores(moderateMask, referableIdx), 2);
    falseNegative = pReferable < thresholds(model);
    numFalseNegative = sum(falseNegative);
    pNoDrFalseNegative = pNoDr(falseNegative);

    fprintf('--- %s ---\n', modelNames(model));
    fprintf('Clinical FN on Moderate: %d / %d (%.2f%%)\n', numFalseNegative, sum(moderateMask), 100*mean(falseNegative));
    fprintf('Moderate P(Mod):  mean=%.4f, median=%.4f, std=%.4f\n', mean(pModerate), median(pModerate), std(pModerate));
    fprintf('Moderate P(NoDR): mean=%.4f, median=%.4f, std=%.4f\n', mean(pNoDr), median(pNoDr), std(pNoDr));
    fprintf('FN high-confidence No_DR (>=0.80): %d / %d\n', sum(pNoDrFalseNegative >= 0.80), numFalseNegative);
    fprintf('FN moderate-confidence No_DR ([0.60,0.80)): %d / %d\n', ...
        sum(pNoDrFalseNegative >= 0.60 & pNoDrFalseNegative < 0.80), numFalseNegative);
    fprintf('FN ambiguous/low-margin (<0.60): %d / %d\n\n', sum(pNoDrFalseNegative < 0.60), numFalseNegative);
end

if ~isfolder(paths.figures), mkdir(paths.figures); end
fig = figure('Visible', 'off', 'Position', [100 100 900 600]);
subplot(2, 2, 1);
histogram(baseline.valScoresV41(moderateMask, moderateIdx), 30, 'FaceColor', 'b', 'FaceAlpha', 0.5); hold on;
histogram(v43.valScores(moderateMask, moderateIdx), 30, 'FaceColor', 'r', 'FaceAlpha', 0.5);
title('P(Moderate) on True Moderate'); xlabel('Probability'); ylabel('Count'); legend('V4.1', 'V4.3');
subplot(2, 2, 2);
histogram(baseline.valScoresV41(moderateMask, noDrIdx), 30, 'FaceColor', 'b', 'FaceAlpha', 0.5); hold on;
histogram(v43.valScores(moderateMask, noDrIdx), 30, 'FaceColor', 'r', 'FaceAlpha', 0.5);
title('P(No DR) on True Moderate'); xlabel('Probability'); ylabel('Count'); legend('V4.1', 'V4.3');
subplot(2, 2, 3);
histogram(baseline.valScoresV41(noDrMask, moderateIdx), 30, 'FaceColor', 'b', 'FaceAlpha', 0.5); hold on;
histogram(v43.valScores(noDrMask, moderateIdx), 30, 'FaceColor', 'r', 'FaceAlpha', 0.5);
title('P(Moderate) on True No DR'); xlabel('Probability'); ylabel('Count'); legend('V4.1', 'V4.3');
subplot(2, 2, 4);
histogram(baseline.valScoresV41(moderateMask, moderateIdx) - baseline.valScoresV41(moderateMask, noDrIdx), 30, 'FaceColor', 'b', 'FaceAlpha', 0.5); hold on;
histogram(v43.valScores(moderateMask, moderateIdx) - v43.valScores(moderateMask, noDrIdx), 30, 'FaceColor', 'r', 'FaceAlpha', 0.5);
title('P(Mod) - P(No DR), True Moderate'); xlabel('Margin'); ylabel('Count'); legend('V4.1', 'V4.3');
saveas(fig, fullfile(paths.figures, 'moderate_nodr_distribution.png'));
close(fig);
