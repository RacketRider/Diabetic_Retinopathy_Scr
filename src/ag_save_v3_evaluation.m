function metrics = ag_save_v3_evaluation(experimentName, scores, labels, classNames, config, pReferable, trainingInfo)
% AG_SAVE_V3_EVALUATION Evaluate and persist the complete V3 artifact set.

if nargin < 6
    pReferable = [];
end
if nargin < 7
    trainingInfo = struct();
end
paths = ag_project_paths();
resultDir = fullfile(paths.v3Results, char(experimentName));
plotDir = fullfile(paths.v3Figures, char(experimentName));
if ~isfolder(resultDir), mkdir(resultDir); end
if ~isfolder(plotDir), mkdir(plotDir); end

metrics = ag_evaluate_model_v3(scores, labels, classNames, pReferable);
classNames = string(classNames(:));
YVal = labels; %#ok<NASGU>
valScores = scores; %#ok<NASGU>
save(fullfile(resultDir, 'results.mat'), 'metrics', 'valScores', 'YVal', ...
    'classNames', 'config', 'trainingInfo', '-v7.3');
writetable(metrics.thresholdSweep, fullfile(resultDir, 'clinical_threshold_results.csv'));
writetable(metrics.perClass, fullfile(resultDir, 'per_class_metrics.csv'));
writeText(fullfile(resultDir, 'selected_threshold.txt'), sprintf( ...
    'threshold=%.3f\nsensitivity=%.6f\nspecificity=%.6f\nTP=%d\nTN=%d\nFP=%d\nFN=%d\n', ...
    metrics.threshold, metrics.sensitivity, metrics.specificity, ...
    metrics.TP, metrics.TN, metrics.FP, metrics.FN));
writeText(fullfile(resultDir, 'configuration.json'), jsonencode(config, 'PrettyPrint', true));

summary = table(string(experimentName), config.inputResolution, logical(config.fovCrop), ...
    string(config.inputChannels), logical(config.rdrHead), logical(config.ordinalHead), ...
    logical(config.hardMining), metrics.valAccuracy, metrics.macroF1, metrics.qwk, ...
    metrics.moderateRecall, metrics.moderateToNoDR, metrics.moderateToMild, ...
    metrics.moderateToNonRef, metrics.moderateToSevere, metrics.moderateToPDR, ...
    metrics.rocAuc, metrics.raw05.sensitivity, ...
    metrics.raw05.specificity, metrics.threshold, metrics.sensitivity, ...
    metrics.specificity, metrics.FN, metrics.FP, ...
    'VariableNames', {'Model','InputResolution','FOVCrop','InputChannels','RDRHead', ...
    'OrdinalHead','HardMining','Accuracy5Class','MacroF1','QWK','ModerateRecall', ...
    'ModerateToNoDR','ModerateToMild','ModerateToNonReferable', ...
    'ModerateToSevere','ModerateToPDR','RDRAUC', ...
    'SensitivityAt05','SpecificityAt05','BestThreshold','SelectedSensitivity', ...
    'SelectedSpecificity','SelectedFN','SelectedFP'});
writetable(summary, fullfile(resultDir, 'summary.csv'));

try
    fig = figure('Visible', 'off', 'Position', [100 100 850 600]);
    cm = confusionchart(labels, metrics.predLabels, 'RowSummary', 'row-normalized', ...
        'ColumnSummary', 'column-normalized');
    cm.Title = sprintf('%s validation confusion matrix', experimentName);
    cm.Interpreter = 'none';
    exportgraphics(fig, fullfile(plotDir, 'confusion_matrix.png'));
    close(fig);
catch ME
    warning('Confusion matrix export failed: %s', ME.message);
end

trueRef = ismember(string(labels), ["Moderate", "Severe", "Proliferate_DR"]);
[xRoc, yRoc] = perfcurve(trueRef, metrics.pRef, true);
fig = figure('Visible', 'off');
plot(xRoc, yRoc, 'LineWidth', 1.5); hold on; plot([0 1], [0 1], 'k--');
xlabel('False positive rate'); ylabel('Sensitivity');
title(sprintf('%s referable ROC (AUC %.4f)', experimentName, metrics.rocAuc), 'Interpreter', 'none');
grid on;
exportgraphics(fig, fullfile(plotDir, 'referable_roc.png'));
close(fig);

fig = figure('Visible', 'off');
plot(metrics.thresholdSweep.Threshold, metrics.thresholdSweep.Sensitivity, 'LineWidth', 1.5); hold on;
plot(metrics.thresholdSweep.Threshold, metrics.thresholdSweep.Specificity, 'LineWidth', 1.5);
yline(0.85, '--'); xline(metrics.threshold, ':');
xlabel('Referable threshold'); ylabel('Metric'); ylim([0 1]); grid on;
legend('Sensitivity', 'Specificity', 'Specificity constraint', 'Selected threshold', 'Location', 'best');
title(sprintf('%s clinical threshold sweep', experimentName), 'Interpreter', 'none');
exportgraphics(fig, fullfile(plotDir, 'clinical_threshold_curve.png'));
close(fig);
end

function writeText(path, text)
fid = fopen(path, 'w');
if fid < 0
    error('ag:v3artifacts:WriteFailed', 'Cannot write %s.', path);
end
cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>
fprintf(fid, '%s', text);
end
