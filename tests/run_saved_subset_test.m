% Compatibility smoke check using five saved test predictions; no images or model are loaded.
projectRoot = fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(projectRoot, 'src'));
resultPath = fullfile(projectRoot, 'artifacts', 'results', 'AG_FINAL_TEST_RESULTS.mat');
artifact = matfile(resultPath);
metadata = load(resultPath, 'classNames', 'thFrozen', 'YTest');
labels = metadata.YTest;
classNames = string(metadata.classNames(:));

selected = zeros(numel(classNames), 1);
for c = 1:numel(classNames)
    index = find(string(labels) == classNames(c), 1);
    assert(~isempty(index), 'Saved predictions contain no %s case.', classNames(c));
    selected(c) = index;
end
scores = zeros(numel(classNames), numel(classNames), 'single');
for c = 1:numel(classNames)
    scores(c, :) = artifact.testScores(selected(c), :);
end
metrics = ag_evaluate_model(scores, labels(selected), classNames, metadata.thFrozen);
assert(sum(metrics.confusionMatrix, 'all') == numel(classNames));
assert(isfinite(metrics.rocAuc) && isfinite(metrics.prAuc));
disp('Saved-result smoke check passed on one prediction per class (5 total).');
