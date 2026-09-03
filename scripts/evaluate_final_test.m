function metrics = evaluate_final_test(maxImages)
% EVALUATE_FINAL_TEST Evaluate the frozen model at its validation-locked threshold.
% Pass a small maxImages (for example, 10) for a stratified smoke test.
% Pass Inf only for the explicitly authorized full, final evaluation.

if nargin < 1 || isempty(maxImages) || ~isscalar(maxImages) || maxImages < 5
    error('ag:test:ExplicitSubsetRequired', 'Pass an explicit maxImages >= 5; use Inf only for a full authorized evaluation.');
end

projectRoot = fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(projectRoot, 'src'));
paths = ag_project_paths();
data = load(paths.datastore, 'imdsTest', 'classNames');
frozen = load(fullfile(paths.models, 'AG_V4_5_ReducedAug.mat'), 'netTrained');
imdsTest = data.imdsTest;
classNames = string(data.classNames(:));

numAvailable = numel(imdsTest.Files);
if isfinite(maxImages) && maxImages < numAvailable
    maxImages = floor(maxImages);
    labels = string(imdsTest.Labels);
    selected = zeros(0, 1);
    rng(42);
    for className = classNames'
        candidates = find(labels == className);
        if isempty(candidates)
            error('ag:test:MissingClass', 'Test datastore has no %s image.', className);
        end
        take = min(numel(candidates), max(1, floor(maxImages / numel(classNames))));
        selected = [selected; candidates(randperm(numel(candidates), take))]; %#ok<AGROW>
    end
    if numel(selected) < maxImages
        remaining = setdiff((1:numAvailable)', selected);
        selected = [selected; remaining(randperm(numel(remaining), maxImages - numel(selected)))];
    end
    imdsTest = subset(imdsTest, selected(1:maxImages));
    isFullEvaluation = false;
else
    isFullEvaluation = true;
end

fprintf('Evaluating frozen AG_V4_5 model on %d test image(s).\n', numel(imdsTest.Files));
augTest = augmentedImageDatastore([224 224 3], imdsTest);
tStart = tic;
testScores = minibatchpredict(frozen.netTrained, augTest);
fprintf('Prediction completed in %.1f seconds.\n', toc(tStart));

threshold = 0.1199;
metrics = ag_evaluate_model(testScores, imdsTest.Labels, classNames, threshold);
if ~isFullEvaluation
    fprintf('Smoke-test subset only: official result artifacts were not changed.\n');
    return;
end

if isfile(paths.baselineTestResults)
    baseline = load(paths.baselineTestResults, ...
        'sensitivityTestV41', 'specificityTestV41', 'aucTestV41', 'aucPRTestV41', ...
        'clinicalAccuracyTestV41', 'ppvTestV41', 'npvTestV41', 'testAccuracyV41', ...
        'FPTestV41', 'FNTestV41');
    fprintf('\n=== FULL-TEST COMPARISON WITH V4.1 ===\n');
    fprintf('Sensitivity: %.2f%% (FN=%d) -> %.2f%% (FN=%d)\n', ...
        baseline.sensitivityTestV41*100, baseline.FNTestV41, metrics.sensitivity*100, metrics.FN);
    fprintf('Specificity: %.2f%% (FP=%d) -> %.2f%% (FP=%d)\n', ...
        baseline.specificityTestV41*100, baseline.FPTestV41, metrics.specificity*100, metrics.FP);
    fprintf('ROC-AUC: %.4f -> %.4f | PR-AUC: %.4f -> %.4f\n', ...
        baseline.aucTestV41, metrics.rocAuc, baseline.aucPRTestV41, metrics.prAuc);
else
    warning('ag:test:MissingBaseline', 'Baseline comparison file not found: %s', paths.baselineTestResults);
end

if ~isfolder(paths.results), mkdir(paths.results); end
if ~isfolder(paths.figures), mkdir(paths.figures); end
save(fullfile(paths.results, 'AG_FINAL_TEST_RESULTS.mat'), 'testScores', 'classNames', 'threshold', 'metrics', '-v7.3');

trueRef = ismember(string(imdsTest.Labels), ["Moderate", "Proliferate_DR", "Severe"]);
fig = figure('Visible', 'off', 'Position', [100 100 1000 450]);
subplot(1, 2, 1);
[Xroc, Yroc] = perfcurve(trueRef, metrics.pRef, true);
plot(Xroc, Yroc, 'b-', 'LineWidth', 2); hold on; plot([0 1], [0 1], 'k--');
xlabel('False Positive Rate (1 - Specificity)'); ylabel('Sensitivity');
title(sprintf('Test ROC Curve (AUC = %.4f)', metrics.rocAuc)); grid on;
subplot(1, 2, 2);
[Xpr, Ypr] = perfcurve(trueRef, metrics.pRef, true, 'xCrit', 'reca', 'yCrit', 'prec');
plot(Xpr, Ypr, 'r-', 'LineWidth', 2);
xlabel('Recall (Sensitivity)'); ylabel('Precision (PPV)');
title(sprintf('Test Precision-Recall Curve (AUC = %.4f)', metrics.prAuc)); grid on;
saveas(fig, fullfile(paths.figures, 'final_test_roc_pr_curves.png'));
close(fig);
end
