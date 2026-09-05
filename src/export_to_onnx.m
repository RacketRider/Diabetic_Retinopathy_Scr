%% export_to_onnx.m — Export trained ResNet-101 E1 model to ONNX format
%
% Requirements: MATLAB R2024a+ with Deep Learning Toolbox
%
% Usage:
%   Run from the repository root:
%     >> cd('path/to/Diabetic_Retinopathy_Scr')
%     >> run('src/export_to_onnx.m')
%
% Output: models/dr_resnet101_e1.onnx

%% Setup paths
run(fullfile('src', 'ag_project_paths.m'));

%% Load trained E1 checkpoint
checkpointFile = fullfile('models', 'V3_E1_HighResFOV448.mat');
if ~isfile(checkpointFile)
    error('Checkpoint not found: %s\nPlease ensure the trained model is in the models/ directory.', ...
          checkpointFile);
end

fprintf('Loading checkpoint: %s\n', checkpointFile);
data = load(checkpointFile);

% The checkpoint stores the trained network — extract it
if isfield(data, 'trainedNet')
    net = data.trainedNet;
elseif isfield(data, 'net')
    net = data.net;
else
    fields = fieldnames(data);
    error('Could not find network in checkpoint. Available fields: %s', ...
          strjoin(fields, ', '));
end

fprintf('Network class: %s\n', class(net));
fprintf('Input size: %s\n', mat2str(net.Layers(1).InputSize));
fprintf('Number of layers: %d\n', numel(net.Layers));

%% Export to ONNX
outputFile = fullfile('models', 'dr_resnet101_e1.onnx');

fprintf('Exporting to ONNX: %s\n', outputFile);

try
    exportONNXNetwork(net, outputFile, ...
        'InputDataFormats', 'BCSS', ...
        'OutputDataFormats', 'BC', ...
        'OpsetVersion', 17);
    fprintf('✓ Successfully exported to %s\n', outputFile);
    
    % Report file size
    info = dir(outputFile);
    fprintf('  Model size: %.1f MB\n', info.bytes / 1024 / 1024);
    
catch ME
    fprintf('\n⚠ exportONNXNetwork failed: %s\n', ME.message);
    fprintf('  This may occur if the network contains custom layers.\n');
    fprintf('  Attempting export with standard layers only...\n');
    
    % Fallback: try without custom format specifiers
    try
        exportONNXNetwork(net, outputFile, 'OpsetVersion', 17);
        fprintf('✓ Fallback export succeeded: %s\n', outputFile);
    catch ME2
        error('Export failed: %s\nConsider exporting E1 (standard ResNet-101) only.', ...
              ME2.message);
    end
end

fprintf('\nNext steps:\n');
fprintf('  1. python scripts/verify_onnx_parity.py --model %s --image <test_image>\n', outputFile);
fprintf('  2. python scripts/quantize_model.py --input %s\n', outputFile);
