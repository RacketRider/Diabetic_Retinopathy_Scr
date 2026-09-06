function metrics = run_V3(experimentName)
% RUN_V3 Public project-root entry point for controlled V3 experiments.
projectRoot = fileparts(mfilename('fullpath'));
addpath(fullfile(projectRoot, 'src'));
metrics = ag_run_v3_dispatch(experimentName);
end
