function metrics = evaluate_e3_split(folderPath)
% EVALUATE_E3_SPLIT Evaluate frozen E3 on an explicit DR1 unified split/class folder.
% Example: metrics = evaluate_e3_split("data/downloads/DR1/dr_unified_v2/dr_unified_v2/val");

projectRoot = fileparts(mfilename('fullpath'));
addpath(fullfile(projectRoot, 'src'));
paths = ag_project_paths();
sourceRoot = canonicalFolder(paths.highResolutionSource);
if ~isfolder(folderPath) && isfolder(fullfile(projectRoot, folderPath))
    folderPath = fullfile(projectRoot, folderPath);
end
selectedFolder = canonicalFolder(folderPath);
if selectedFolder ~= sourceRoot && ~startsWith(selectedFolder, sourceRoot + filesep)
    error('ag:e3split:OutsideDR1', ...
        'Selected folder must be under the configured DR1 unified root: %s', sourceRoot);
end

fprintf('[E3 SPLIT] Discovering images under %s\n', selectedFolder);
imds = imageDatastore(selectedFolder, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
if isempty(imds.Files)
    error('ag:e3split:NoImages', 'No supported images found under %s.', selectedFolder);
end
[files, order] = sort(string(imds.Files(:)));
rawGrades = string(imds.Labels(order));
grades = ["0"; "1"; "2"; "3"; "4"];
classByGrade = ["No_DR"; "Mild"; "Moderate"; "Severe"; "Proliferate_DR"];
[knownGrade, gradeIndex] = ismember(rawGrades, grades);
if ~all(knownGrade)
    error('ag:e3split:InvalidClassFolder', ...
        'Every image must be directly inside a class folder named 0, 1, 2, 3, or 4.');
end
mappedLabels = classByGrade(gradeIndex);

checkpointCandidates = [string(fullfile(paths.models, 'V3_E3_RetinaWalker_best.mat')); ...
    string(fullfile(paths.models, 'V3_E3_RetinaWalker.mat'))];
checkpointPath = checkpointCandidates(find(isfile(checkpointCandidates), 1));
if isempty(checkpointPath)
    error('ag:e3split:MissingCheckpoint', ...
        'Missing trained E3 best/final checkpoint under %s.', paths.models);
end
fprintf('[E3 SPLIT] Loading frozen checkpoint: %s\n', checkpointPath);
saved = load(checkpointPath, 'model', 'config', 'classNames');
if ~all(isfield(saved, {'model','config','classNames'})) || ...
        ~all(isfield(saved.model, {'globalNet','localNet','fusionNet'}))
    error('ag:e3split:InvalidCheckpoint', 'E3 checkpoint is missing its trained model/configuration schema.');
end
model = saved.model;
cfg = saved.config;
classNames = string(saved.classNames(:));
if numel(classNames) ~= 5 || ~all(ismember(classByGrade, classNames)) || ...
        cfg.inputResolution ~= 448 || cfg.walkerResolution ~= 448
    error('ag:e3split:InvalidCheckpoint', 'E3 checkpoint class schema or 448x448 configuration is invalid.');
end
labels = categorical(mappedLabels, classNames, classNames);
presentClasses = unique(mappedLabels);
isDiagnostic = numel(presentClasses) < numel(classByGrade);
if isDiagnostic
    evaluationMode = "single-class diagnostic";
    if numel(presentClasses) > 1, evaluationMode = "partial-class diagnostic"; end
    warning('ag:e3split:DiagnosticOnly', ...
        ['%s evaluation (%s). Sensitivity/specificity, ROC/PR AUC, QWK, and balanced ' ...
         'multi-class metrics may be undefined or not meaningful.'], evaluationMode, strjoin(presentClasses, ', '));
else
    evaluationMode = "five-class evaluation";
end

relativeFolder = extractAfter(selectedFolder, strlength(sourceRoot));
if startsWith(relativeFolder, filesep), relativeFolder = extractAfter(relativeFolder, 1); end
if strlength(relativeFolder) == 0, relativeFolder = "root"; end
artifactKey = string(matlab.lang.makeValidName(char(replace(relativeFolder, filesep, '__'))));
resultDir = fullfile(paths.v3Results, 'V3_E3_RetinaWalker_explicit_splits', artifactKey);
cacheFile = fullfile(paths.artifacts, 'cache', 'E3_walker_explicit_splits', artifactKey + ".mat");

cfg.useGPU = gpuDeviceCount('available') > 0;
fprintf('[E3 SPLIT] %d images | %s | GPU: %d\n', numel(files), evaluationMode, cfg.useGPU);
fprintf('[E3 SPLIT] Building/reusing Retina Walker coordinates...\n');
cache = ag_build_e3_walker_cache(cellstr(files), cfg, cacheFile, artifactKey, labels);
if ~all(cache.completed) || ~isequal(cache.files, cellstr(files))
    error('ag:e3split:CacheMismatch', 'Walker cache does not match the selected file order.');
end
cfg.actualVesselMethod = strjoin(unique(cache.vesselMethod), ', ');
cfg.actualFovMaskMethod = strjoin(unique(cache.fovMethod), ', ');
cfg.E3.actualVesselMethod = cfg.actualVesselMethod;
cfg.E3.actualFovMaskMethod = cfg.actualFovMaskMethod;
data = struct('files', {cellstr(files)}, 'labels', labels, 'classNames', classNames, ...
    'coords', double(cache.coords), 'randomCoords', double(cache.randomCoords), ...
    'walkerScores', cache.scores, 'samplingType', cache.samplingType, ...
    'fovMethod', cache.fovMethod, 'fovFraction', cache.fovFraction);

fprintf('[E3 SPLIT] Running frozen full global + Retina Walker inference...\n');
prediction = ag_e3_predict(model, data, cfg, "full");
fixedThreshold = 0.061;
metrics = ag_evaluate_model_v3(prediction.scores.full, labels, classNames, [], ...
    fixedThreshold, isDiagnostic);
metrics.e3 = struct('numWalkerPatches', cfg.numPatches, 'patchSize', cfg.patchSize, ...
    'vesselMethod', string(cfg.actualVesselMethod), 'aggregationMethod', "mean", ...
    'initialCheckpoint', string(cfg.initialCheckpoint), ...
    'globalFeatureDim', model.globalFeatureDim, 'localFeatureDim', model.localFeatureDim, ...
    'fusionFeatureDim', model.fusionFeatureDim, ...
    'meanInferenceTime', prediction.meanInferenceTime, 'walkerVisualPreflight', []);
metrics.evaluation = struct('mode', evaluationMode, 'sourceFolder', selectedFolder, ...
    'checkpoint', checkpointPath, 'fixedValidationThreshold', fixedThreshold, ...
    'imageCount', numel(files), 'classFolders', grades, 'classNames', classByGrade);

if ~isfolder(resultDir), mkdir(resultDir); end
sourceFolder = selectedFolder;
checkpoint = checkpointPath;
config = cfg;
save(fullfile(resultDir, 'results.mat'), 'metrics', 'prediction', 'files', 'labels', ...
    'classNames', 'config', 'sourceFolder', 'checkpoint', 'fixedThreshold', '-v7.3');
writetable(metrics.perClass, fullfile(resultDir, 'per_class_metrics.csv'));
columnNames = matlab.lang.makeValidName(cellstr("Predicted_" + classNames));
confusionTable = array2table(metrics.confusionMatrix, 'VariableNames', columnNames, ...
    'RowNames', cellstr("True_" + classNames));
writetable(confusionTable, fullfile(resultDir, 'confusion_matrix.csv'), 'WriteRowNames', true);

fprintf('[E3 SPLIT] Fixed validation threshold: %.3f (not re-optimized)\n', metrics.threshold);
fprintf('[E3 SPLIT] Confusion matrix (rows=true, columns=predicted):\n');
disp(confusionTable);
fprintf('[E3 SPLIT] Saved isolated results: %s\n', resultDir);
end

function folder = canonicalFolder(path)
folder = string(java.io.File(char(path)).getCanonicalPath());
if ~isfolder(folder)
    error('ag:e3split:MissingFolder', 'Folder not found: %s', string(path));
end
end
