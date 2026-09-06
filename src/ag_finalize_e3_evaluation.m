function metrics = ag_finalize_e3_evaluation(metrics, validationPrediction, testPrediction, ...
        validationData, testData, model, cfg, trainingInfo, visualReport, paths)
% AG_FINALIZE_E3_EVALUATION Add E3 diagnostics, ablations, test metrics, and figures.

resultDir = fullfile(paths.v3Results, char(cfg.name));
plotDir = fullfile(paths.v3Figures, char(cfg.name));
if ~isfolder(resultDir), mkdir(resultDir); end
if ~isfolder(plotDir), mkdir(plotDir); end
metrics.e3 = struct('numWalkerPatches', cfg.numPatches, 'patchSize', cfg.patchSize, ...
    'vesselMethod', string(cfg.actualVesselMethod), 'aggregationMethod', "mean", ...
    'initialCheckpoint', string(cfg.initialCheckpoint), ...
    'globalFeatureDim', model.globalFeatureDim, 'localFeatureDim', model.localFeatureDim, ...
    'fusionFeatureDim', model.fusionFeatureDim, ...
    'meanInferenceTime', validationPrediction.meanInferenceTime, ...
    'walkerVisualPreflight', visualReport);
if isfield(validationPrediction.scores, 'globalOnly')
    metrics.ablations.globalOnly = compactMetrics(ag_evaluate_model_v3( ...
        validationPrediction.scores.globalOnly, validationData.labels, validationData.classNames));
end
if isfield(validationPrediction.scores, 'random')
    metrics.ablations.randomRetinal = compactMetrics(ag_evaluate_model_v3( ...
        validationPrediction.scores.random, validationData.labels, validationData.classNames));
end
metrics.ablations.fullVesselWalker = compactMetrics(metrics);
metrics.ablations.note = "Inference-only branch masking/substitution; no separate retraining.";
metrics.test = ag_evaluate_model_v3(testPrediction.scores.full, testData.labels, ...
    testData.classNames, [], metrics.threshold);

YVal = validationData.labels; valScores = validationPrediction.scores.full; %#ok<NASGU>
classNames = validationData.classNames; config = cfg; %#ok<NASGU>
save(fullfile(resultDir, 'results.mat'), 'metrics', 'valScores', 'YVal', ...
    'classNames', 'config', 'trainingInfo', '-v7.3');
testMetrics = metrics.test; %#ok<NASGU>
save(fullfile(resultDir, 'test_results.mat'), 'testPrediction', 'testMetrics', ...
    'classNames', 'config', '-v7.3');
writetable(ablationTable(metrics), fullfile(resultDir, 'inference_only_ablations.csv'));
writetable(metrics.test.perClass, fullfile(resultDir, 'test_per_class_metrics.csv'));

saveTrainingCurves(trainingInfo, fullfile(plotDir, 'training_curves.png'));
savePerClassRecall(metrics, fullfile(plotDir, 'per_class_recall.png'));
saveClinicalSummary(metrics, fullfile(plotDir, 'sensitivity_specificity_summary.png'));
saveFailureAnalysis(validationData, metrics, cfg, resultDir, plotDir);
end

function output = compactMetrics(input)
fields = {'overallAccuracy','balancedAccuracy','macroF1','moderateRecall', ...
    'sensitivity','specificity','rocAuc','prAuc','moderateToNoDR'};
output = struct();
for i = 1:numel(fields), output.(fields{i}) = input.(fields{i}); end
end

function output = ablationTable(metrics)
names = ["Global only"; "Global + vessel walker"];
values = [metrics.ablations.globalOnly; metrics.ablations.fullVesselWalker];
if isfield(metrics.ablations, 'randomRetinal')
    names(end + 1) = "Global + random retinal patches";
    values(end + 1) = metrics.ablations.randomRetinal;
end
output = table(names, [values.overallAccuracy]', [values.balancedAccuracy]', ...
    [values.macroF1]', [values.moderateRecall]', [values.sensitivity]', ...
    [values.specificity]', [values.rocAuc]', [values.moderateToNoDR]', ...
    'VariableNames', {'Mode','OverallAccuracy','BalancedAccuracy','MacroF1', ...
    'ModerateRecall','ReferableSensitivity','ReferableSpecificity','ReferableROCAUC', ...
    'ModerateToNoDR'});
end

function saveTrainingCurves(info, outputPath)
fig = figure('Visible', 'off', 'Position', [50 50 900 450]);
yyaxis left; plot(info.Epoch, info.TrainingLoss, '-o', 'LineWidth', 1.5); ylabel('Training loss');
yyaxis right; plot(info.Epoch, info.ValidationBalancedAccuracy, '-s', 'LineWidth', 1.5); ylabel('Validation balanced accuracy');
xlabel('Epoch'); grid on; title('E3 training curves');
exportgraphics(fig, outputPath); close(fig);
end

function savePerClassRecall(metrics, outputPath)
fig = figure('Visible', 'off', 'Position', [50 50 850 500]);
bar(100 * metrics.perClass.Recall); ylim([0 100]); ylabel('Recall (%)'); grid on;
xticklabels(metrics.perClass.Class); xtickangle(25); title('E3 validation per-class recall');
exportgraphics(fig, outputPath); close(fig);
end

function saveClinicalSummary(metrics, outputPath)
fig = figure('Visible', 'off', 'Position', [50 50 700 450]);
bar(100 * [metrics.sensitivity metrics.specificity metrics.test.sensitivity metrics.test.specificity]);
ylim([0 100]); ylabel('Percent'); grid on;
xticklabels({'Val sensitivity','Val specificity','Test sensitivity','Test specificity'}); xtickangle(20);
title(sprintf('E3 referable DR (validation threshold %.3f)', metrics.threshold));
exportgraphics(fig, outputPath); close(fig);
end

function saveFailureAnalysis(data, metrics, cfg, resultDir, plotDir)
trueLabel = string(data.labels(:)); predictedLabel = string(metrics.predLabels(:));
selected = (trueLabel == "Moderate" & predictedLabel == "No_DR") | ...
    (trueLabel == "No_DR" & predictedLabel == "Moderate") | ...
    (trueLabel == "Mild" & predictedLabel == "Moderate") | ...
    (trueLabel == "Moderate" & predictedLabel == "Mild") | ...
    (trueLabel == "Severe" & predictedLabel == "Moderate");
indices = find(selected);
T = table(string(data.files(indices)), trueLabel(indices), predictedLabel(indices), ...
    'VariableNames', {'File','TrueClass','PredictedClass'});
for c = 1:numel(data.classNames)
    T.("p" + matlab.lang.makeValidName(data.classNames(c))) = metrics.scores(indices, c);
end
writetable(T, fullfile(resultDir, 'failure_cases.csv'));
failureDir = fullfile(plotDir, 'failure_analysis');
if ~isfolder(failureDir), mkdir(failureDir); end
pairs = ["Moderate" "No_DR"; "No_DR" "Moderate"; "Mild" "Moderate"; ...
    "Moderate" "Mild"; "Severe" "Moderate"];
for p = 1:size(pairs, 1)
    pairIndices = find(trueLabel == pairs(p, 1) & predictedLabel == pairs(p, 2));
    for n = 1:min(cfg.failureExamplesPerType, numel(pairIndices))
        i = pairIndices(n); image = preprocessFundus(data.files{i}, cfg.inputResolution);
        coords = double(data.coords(:, :, i)); patches = ag_e3_extract_patches(image, coords, cfg.patchSize);
        fig = figure('Visible', 'off', 'Position', [20 20 1800 750]);
        tiledlayout(2, 5, 'Padding', 'compact', 'TileSpacing', 'compact');
        nexttile; imshow(image); hold on;
        for k = 1:size(coords, 1)
            rectangle('Position', [coords(k,1)-cfg.patchSize(2)/2 coords(k,2)-cfg.patchSize(1)/2 ...
                cfg.patchSize(2) cfg.patchSize(1)], 'EdgeColor', 'y');
            text(coords(k,1), coords(k,2), string(k), 'Color', 'white', 'FontWeight', 'bold');
        end
        title(sprintf('%s -> %s', pairs(p,1), pairs(p,2)));
        for k = 1:cfg.numPatches, nexttile; imshow(patches(:,:,:,k)); title(sprintf('P%d', k)); end
        sgtitle(sprintf('Probabilities: %s', strjoin(compose('%s %.3f', data.classNames, metrics.scores(i,:)'), ' | ')), ...
            'Interpreter', 'none');
        exportgraphics(fig, fullfile(failureDir, sprintf('%s_to_%s_%02d.png', ...
            lower(pairs(p,1)), lower(pairs(p,2)), n))); close(fig);
    end
end
end


