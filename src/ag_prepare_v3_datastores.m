function [trainHigh, validationHigh, report] = ag_prepare_v3_datastores(imdsTrain, imdsValidation, targetSize, sourceRoot)
% AG_PREPARE_V3_DATASTORES Remap the fixed split to genuine high-res files using canonical basenames.

if ~isfolder(sourceRoot)
    error('ag:v3data:MissingSource', 'High-resolution source directory not found: %s', sourceRoot);
end

projectRoot = fileparts(fileparts(mfilename('fullpath')));
cacheDir = fullfile(projectRoot, 'artifacts', 'cache');
cacheFile = fullfile(cacheDir, 'highres_manifest.mat');
manifestVersion = 2;

if isfile(cacheFile)
    cached = load(cacheFile, 'index', 'sourceRoot', 'manifestVersion');
    if isfield(cached, 'index') && isfield(cached, 'sourceRoot') && isequal(cached.sourceRoot, sourceRoot) && ...
            isfield(cached, 'manifestVersion') && cached.manifestVersion == manifestVersion
        index = cached.index;
    else
        index = buildIndex(sourceRoot, cacheDir, cacheFile, manifestVersion);
    end
else
    index = buildIndex(sourceRoot, cacheDir, cacheFile, manifestVersion);
end

[trainFiles, trainMissing] = remapFiles(imdsTrain.Files, imdsTrain.Labels, index);
[validationFiles, validationMissing] = remapFiles(imdsValidation.Files, imdsValidation.Labels, index);
if ~isempty(trainMissing) || ~isempty(validationMissing)
    error('ag:v3data:MissingMappings', '%d training and %d validation images lack high-resolution matches.', ...
        numel(trainMissing), numel(validationMissing));
end

trainBases = arrayfun(@(f) lower(char(portableStem(f{1}))), trainFiles, 'UniformOutput', false);
valBases = arrayfun(@(f) lower(char(portableStem(f{1}))), validationFiles, 'UniformOutput', false);
if ~isempty(intersect(trainBases, valBases))
    error('ag:v3data:SplitLeakage', 'Mapped train and validation sets overlap.');
end

readFcn = @(filename) preprocessFundus(filename, targetSize);
trainHigh = imageDatastore(trainFiles, 'Labels', imdsTrain.Labels, 'ReadFcn', readFcn);
validationHigh = imageDatastore(validationFiles, 'Labels', imdsValidation.Labels, 'ReadFcn', readFcn);
report = struct('targetSize', targetSize, 'sourceRoot', sourceRoot, ...
    'trainCount', numel(trainFiles), 'validationCount', numel(validationFiles), ...
    'trainFiles', {trainFiles}, 'validationFiles', {validationFiles}, 'index', index);
end

function [mapped, missing] = remapFiles(files, labels, index)
classToGrade = containers.Map( ...
    {'No_DR','Mild','Moderate','Severe','Proliferate_DR'}, {'0','1','2','3','4'});
mapped = cell(size(files));
missing = strings(0, 1);
for i = 1:numel(files)
    stem = portableStem(files{i});
    key = lower(char(stem));
    if ~isKey(index, key)
        missing(end + 1, 1) = string(stem); %#ok<AGROW>
        continue;
    end
    candidate = index(key);
    [parent, ~, ~] = fileparts(candidate);
    [~, grade] = fileparts(parent);
    label = char(string(labels(i)));
    if ~isKey(classToGrade, label) || grade ~= string(classToGrade(label))
        error('ag:v3data:LabelMismatch', 'Mapped source label mismatch for %s (%s vs folder %s).', stem, label, grade);
    end
    mapped{i} = candidate;
end
mapped = mapped(~cellfun('isempty', mapped));
end

function index = buildIndex(sourceRoot, cacheDir, cacheFile, manifestVersion)
sourceFiles = dir(fullfile(sourceRoot, '**', '*.jpg'));
if isempty(sourceFiles)
    error('ag:v3data:NoImages', 'No high-resolution JPG files found under %s.', sourceRoot);
end
index = containers.Map('KeyType', 'char', 'ValueType', 'char');
for i = 1:numel(sourceFiles)
    stem = portableStem(sourceFiles(i).name);
    key = lower(char(stem));
    if isKey(index, key)
        error('ag:v3data:DuplicateStem', 'Duplicate high-resolution basename: %s', stem);
    end
    index(key) = fullfile(sourceFiles(i).folder, sourceFiles(i).name);
end
if ~isfolder(cacheDir), mkdir(cacheDir); end
save(cacheFile, 'index', 'sourceRoot', 'manifestVersion', '-v7.3');
end
