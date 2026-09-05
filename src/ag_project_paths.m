function paths = ag_project_paths()
% AG_PROJECT_PATHS Return repository-relative paths used by all scripts.

root = fileparts(fileparts(mfilename('fullpath')));
paths.root = root;
paths.data = fullfile(root, 'data');
paths.models = fullfile(root, 'models');
paths.artifacts = fullfile(root, 'artifacts');
paths.results = fullfile(paths.artifacts, 'results');
paths.figures = fullfile(paths.artifacts, 'figures');
paths.v3Results = fullfile(paths.results, 'V3');
paths.v3Figures = fullfile(paths.figures, 'V3');
paths.datastore = fullfile(paths.data, 'DR_V4_RESNET101_SCREENING.mat');
paths.highResolutionSource = fullfile(paths.data, 'downloads', 'DR1', ...
    'dr_unified_v2', 'dr_unified_v2');
end
