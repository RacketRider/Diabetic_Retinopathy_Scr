%% create_dr_screening_pipeline.m
% Programmatically constructs the Simulink telemedicine workflow model:
%   simulink/dr_screening_pipeline.slx
%
% Problem Statement: SIH26038 (Explainable AI for Diabetic Retinopathy Screening)
%
% Pipeline Architecture:
%   PHC Fundus Capture (50 img/day, ~500 KB/img)
%     -> Rural Network Upload (2 Mbps link, ~2.0s delay)
%     -> AI Inference Server (~350ms CPU inference, ResNet-101 ONNX)
%     -> Clinical Triage (Threshold = 0.164: 82% Normal vs 18% Referable)
%     -> Ophthalmologist Review Queue (<30s review/case with Grad-CAM)
%     -> Clinical Report Generation
%
% Usage:
%   Run in MATLAB:
%     >> run('simulink/create_dr_screening_pipeline.m')
%
% Output:
%   simulink/dr_screening_pipeline.slx

clear all;
close all;
clc;

modelName = 'dr_screening_pipeline';
slxFile = fullfile(fileparts(mfilename('fullpath')), [modelName '.slx']);

fprintf('=================================================================\n');
fprintf('Creating Simulink Model: %s\n', slxFile);
fprintf('=================================================================\n');

%% 1. Initialize System
if bdIsLoaded(modelName)
    close_system(modelName, 0);
end

if isfile(slxFile)
    delete(slxFile);
end

new_system(modelName);
open_system(modelName);

% Set Model Parameters
set_param(modelName, ...
    'Solver', 'VariableStepAuto', ...
    'StartTime', '0.0', ...
    'StopTime', '28800', ... % 8-hour workday (28,800 seconds)
    'SaveTime', 'on', ...
    'TimeSaveName', 'tout', ...
    'SaveOutput', 'on', ...
    'OutputSaveName', 'yout');

fprintf('  [1/6] Initialized new model with 8-hour workday horizon (28800 s)\n');

%% 2. Add Pipeline Subsystem 1: PHC Fundus Capture
% 50 images per 8-hour day -> arrival every 576 seconds
% Image size: 500 KB mean
sub1 = [modelName '/PHC_Fundus_Capture'];
add_block('simulink/Ports & Subsystems/Subsystem', sub1, 'Position', [60, 100, 240, 230]);

% Inside Subsystem 1:
delete_line(sub1, 'In1/1', 'Out1/1');
delete_block([sub1 '/In1']);
set_param([sub1 '/Out1'], 'Name', 'Image_Trigger', 'Position', [420, 50, 450, 70]);
add_block('simulink/Ports & Subsystems/Out1', [sub1 '/Image_Size_KB'], 'Position', [420, 110, 450, 130]);
add_block('simulink/Ports & Subsystems/Out1', [sub1 '/Captured_Count'], 'Position', [420, 170, 450, 190]);

% Pulse Generator: Period = 576 s, Pulse Width = 1% of period, Amplitude = 1
add_block('simulink/Sources/Pulse Generator', [sub1 '/Arrival_Pulse'], ...
    'Position', [50, 45, 110, 75], ...
    'PulseType', 'Time based', ...
    'Period', '576', ...
    'PulseWidth', '1', ...
    'Amplitude', '1');

% Image size: Constant 500 KB
add_block('simulink/Sources/Constant', [sub1 '/Avg_Image_Size_KB'], ...
    'Position', [50, 105, 120, 135], ...
    'Value', '500');

% Discrete Arrival Step and Counter (reaches exactly 50 at t = 28,800 s)
add_block('simulink/Sources/Constant', [sub1 '/Arrival_Unit_Step'], ...
    'Position', [50, 165, 110, 195], ...
    'Value', '1', ...
    'SampleTime', '576');
add_block('simulink/Discrete/Discrete-Time Integrator', [sub1 '/Captured_Counter'], ...
    'Position', [180, 165, 230, 195], ...
    'IntegratorMethod', 'Accumulation: Forward Euler', ...
    'gainval', '1.0', ...
    'SampleTime', '576', ...
    'OutDataTypeStr', 'double');

add_line(sub1, 'Arrival_Pulse/1', 'Image_Trigger/1');
add_line(sub1, 'Avg_Image_Size_KB/1', 'Image_Size_KB/1');
add_line(sub1, 'Arrival_Unit_Step/1', 'Captured_Counter/1');
add_line(sub1, 'Captured_Counter/1', 'Captured_Count/1');

fprintf('  [2/6] Constructed PHC Fundus Capture subsystem (50 arrivals, 500 KB/img)\n');

%% 3. Add Pipeline Subsystem 2: Rural Network Upload
% Bandwidth: 2 Mbps (2000 kbps) -> Delay = (500 KB * 8) / 2000 kbps = 2.0 s
sub2 = [modelName '/Rural_Network_Upload'];
add_block('simulink/Ports & Subsystems/Subsystem', sub2, 'Position', [310, 100, 490, 230]);

delete_line(sub2, 'In1/1', 'Out1/1');
set_param([sub2 '/In1'], 'Name', 'Image_Trigger', 'Position', [50, 50, 80, 70]);
add_block('simulink/Ports & Subsystems/In1', [sub2 '/Image_Size_KB'], 'Position', [50, 110, 80, 130]);
add_block('simulink/Ports & Subsystems/In1', [sub2 '/Captured_Count'], 'Position', [50, 170, 80, 190]);

set_param([sub2 '/Out1'], 'Name', 'Uploaded_Trigger', 'Position', [420, 50, 450, 70]);
add_block('simulink/Ports & Subsystems/Out1', [sub2 '/Cumulative_MB'], 'Position', [420, 110, 450, 130]);
add_block('simulink/Ports & Subsystems/Out1', [sub2 '/Upload_Delay_s'], 'Position', [420, 170, 450, 190]);

% Uplink delay = 2.0s
add_block('simulink/Continuous/Transport Delay', [sub2 '/Uplink_Transmission_Delay'], ...
    'Position', [170, 45, 230, 75], ...
    'DelayTime', '2.0', ...
    'InitialOutput', '0');

% Upload time indicator: Constant 2.0 s
add_block('simulink/Sources/Constant', [sub2 '/Const_Upload_Delay'], ...
    'Position', [170, 165, 230, 195], ...
    'Value', '2.0');

% Cumulative Uploaded MB = Captured_Count * (500 KB / 1024 KB/MB) = 24.414 MB
add_block('simulink/Math Operations/Gain', [sub2 '/Count_to_MB'], ...
    'Position', [170, 105, 240, 135], ...
    'Gain', '500/1024');

add_line(sub2, 'Image_Trigger/1', 'Uplink_Transmission_Delay/1');
add_line(sub2, 'Uplink_Transmission_Delay/1', 'Uploaded_Trigger/1');
add_line(sub2, 'Captured_Count/1', 'Count_to_MB/1');
add_line(sub2, 'Count_to_MB/1', 'Cumulative_MB/1');
add_line(sub2, 'Const_Upload_Delay/1', 'Upload_Delay_s/1');

fprintf('  [3/6] Constructed Rural Network Upload subsystem (2 Mbps, 24.4 MB/day total)\n');

%% 4. Add Pipeline Subsystem 3: AI Inference Server
% CPU inference latency: 350 ms = 0.35 s (measured ONNX runtime)
% Triage scores: 50 clinical predictions matching 18% referable / 82% normal distribution
sub3 = [modelName '/AI_Inference_Server'];
add_block('simulink/Ports & Subsystems/Subsystem', sub3, 'Position', [560, 100, 740, 230]);

delete_line(sub3, 'In1/1', 'Out1/1');
set_param([sub3 '/In1'], 'Name', 'Uploaded_Trigger', 'Position', [50, 50, 80, 70]);
set_param([sub3 '/Out1'], 'Name', 'Inference_Complete', 'Position', [420, 50, 450, 70]);
add_block('simulink/Ports & Subsystems/Out1', [sub3 '/Referable_Score'], 'Position', [420, 110, 450, 130]);
add_block('simulink/Ports & Subsystems/Out1', [sub3 '/Inference_Latency_ms'], 'Position', [420, 170, 450, 190]);

% Inference delay = 0.35 s
add_block('simulink/Continuous/Transport Delay', [sub3 '/ONNX_Runtime_Latency'], ...
    'Position', [160, 45, 220, 75], ...
    'DelayTime', '0.35', ...
    'InitialOutput', '0');

% Clinical prediction score sequence (50 cases: exactly 9 referable >= 0.164, 41 normal < 0.164)
scores = [ ...
    0.021, 0.045, 0.012, 0.380, 0.065, 0.088, 0.034, 0.095, 0.540, 0.018, ...
    0.072, 0.041, 0.110, 0.029, 0.720, 0.055, 0.038, 0.082, 0.015, 0.067, ...
    0.610, 0.049, 0.023, 0.091, 0.033, 0.450, 0.076, 0.019, 0.084, 0.052, ...
    0.780, 0.028, 0.063, 0.037, 0.105, 0.044, 0.810, 0.022, 0.059, 0.071, ...
    0.031, 0.690, 0.017, 0.083, 0.048, 0.092, 0.026, 0.350, 0.061, 0.039 ...
];

add_block('simulink/Sources/Repeating Sequence Stair', [sub3 '/Score_Sequence'], ...
    'Position', [50, 105, 120, 135], ...
    'OutValues', mat2str(scores), ...
    'tsamp', '576', ...
    'OutDataTypeStr', 'double');

% Constant latency indicator = 350 ms
add_block('simulink/Sources/Constant', [sub3 '/Latency_Metric'], ...
    'Position', [160, 165, 220, 195], ...
    'Value', '350');

add_line(sub3, 'Uploaded_Trigger/1', 'ONNX_Runtime_Latency/1');
add_line(sub3, 'ONNX_Runtime_Latency/1', 'Inference_Complete/1');
add_line(sub3, 'Score_Sequence/1', 'Referable_Score/1');
add_line(sub3, 'Latency_Metric/1', 'Inference_Latency_ms/1');

fprintf('  [4/6] Constructed AI Inference Server subsystem (~350 ms CPU ONNX latency)\n');

%% 5. Add Pipeline Subsystem 4: Clinical Triage (18% Referral vs 82% Auto-clear)
sub4 = [modelName '/Clinical_Triage'];
add_block('simulink/Ports & Subsystems/Subsystem', sub4, 'Position', [810, 100, 990, 240]);

delete_line(sub4, 'In1/1', 'Out1/1');
set_param([sub4 '/In1'], 'Name', 'Inference_Trigger', 'Position', [50, 50, 80, 70]);
add_block('simulink/Ports & Subsystems/In1', [sub4 '/Referable_Score'], 'Position', [50, 110, 80, 130]);
set_param([sub4 '/Out1'], 'Name', 'AutoCleared_Count', 'Position', [480, 50, 510, 70]);
add_block('simulink/Ports & Subsystems/Out1', [sub4 '/Referred_Count'], 'Position', [480, 110, 510, 130]);
add_block('simulink/Ports & Subsystems/Out1', [sub4 '/Referral_Flag'], 'Position', [480, 170, 510, 190]);

% Decision threshold comparator: Score >= 0.164 (E1 Calibrated Threshold)
add_block('simulink/Logic and Bit Operations/Compare To Constant', [sub4 '/Threshold_Check'], ...
    'Position', [130, 105, 200, 135], ...
    'relop', '>=', ...
    'const', '0.164');

% Auto-clear condition: NOT (Score >= 0.164)
add_block('simulink/Logic and Bit Operations/Logical Operator', [sub4 '/Not_Referable'], ...
    'Position', [240, 45, 270, 75], ...
    'Operator', 'NOT');

% Scale / convert boolean to double
add_block('simulink/Math Operations/Gain', [sub4 '/Normal_Gain'], ...
    'Position', [310, 45, 340, 75], ...
    'Gain', '1.0');
add_block('simulink/Math Operations/Gain', [sub4 '/Referral_Gain'], ...
    'Position', [310, 105, 340, 135], ...
    'Gain', '1.0');

% Discrete accumulators for Auto-cleared (41) and Referred (9)
add_block('simulink/Discrete/Discrete-Time Integrator', [sub4 '/AutoClear_Integrator'], ...
    'Position', [380, 45, 430, 75], ...
    'IntegratorMethod', 'Accumulation: Forward Euler', ...
    'gainval', '1.0', ...
    'SampleTime', '576', ...
    'OutDataTypeStr', 'double');

add_block('simulink/Discrete/Discrete-Time Integrator', [sub4 '/Referral_Integrator'], ...
    'Position', [380, 105, 430, 135], ...
    'IntegratorMethod', 'Accumulation: Forward Euler', ...
    'gainval', '1.0', ...
    'SampleTime', '576', ...
    'OutDataTypeStr', 'double');

add_line(sub4, 'Referable_Score/1', 'Threshold_Check/1');
add_line(sub4, 'Threshold_Check/1', 'Not_Referable/1');
add_line(sub4, 'Threshold_Check/1', 'Referral_Gain/1');
add_line(sub4, 'Threshold_Check/1', 'Referral_Flag/1');
add_line(sub4, 'Not_Referable/1', 'Normal_Gain/1');
add_line(sub4, 'Normal_Gain/1', 'AutoClear_Integrator/1');
add_line(sub4, 'AutoClear_Integrator/1', 'AutoCleared_Count/1');
add_line(sub4, 'Referral_Gain/1', 'Referral_Integrator/1');
add_line(sub4, 'Referral_Integrator/1', 'Referred_Count/1');

fprintf('  [5/6] Constructed Clinical Triage subsystem (Threshold = 0.164, 82%% / 18%% split)\n');

%% 6. Add Pipeline Subsystem 5: Ophthalmologist Review & Report Generation
sub5 = [modelName '/Ophthalmologist_Review_Queue'];
add_block('simulink/Ports & Subsystems/Subsystem', sub5, 'Position', [1060, 100, 1260, 240]);

delete_line(sub5, 'In1/1', 'Out1/1');
set_param([sub5 '/In1'], 'Name', 'Referred_Count', 'Position', [50, 50, 80, 70]);
add_block('simulink/Ports & Subsystems/In1', [sub5 '/Referral_Flag'], 'Position', [50, 110, 80, 130]);
set_param([sub5 '/Out1'], 'Name', 'Specialist_Time_Minutes', 'Position', [420, 50, 450, 70]);
add_block('simulink/Ports & Subsystems/Out1', [sub5 '/Queue_Backlog'], 'Position', [420, 110, 450, 130]);
add_block('simulink/Ports & Subsystems/Out1', [sub5 '/Reports_Generated'], 'Position', [420, 170, 450, 190]);

% Specialist review time: 30s per case -> 0.5 minutes / case
add_block('simulink/Math Operations/Gain', [sub5 '/Case_to_Minutes'], ...
    'Position', [180, 45, 260, 75], ...
    'Gain', '30/60');

% Specialist review latency: cases cleared within 30s of referral
add_block('simulink/Continuous/Transport Delay', [sub5 '/Specialist_Review_Latency'], ...
    'Position', [180, 105, 240, 135], ...
    'DelayTime', '30.0', ...
    'InitialOutput', '0');
add_block('simulink/Math Operations/Subtract', [sub5 '/Backlog_Calc'], ...
    'Position', [290, 105, 320, 135]);

% Reports generated = Referred count
add_block('simulink/Math Operations/Gain', [sub5 '/Report_Identity'], ...
    'Position', [180, 165, 230, 195], ...
    'Gain', '1');

add_line(sub5, 'Referred_Count/1', 'Case_to_Minutes/1');
add_line(sub5, 'Case_to_Minutes/1', 'Specialist_Time_Minutes/1');
add_line(sub5, 'Referred_Count/1', 'Backlog_Calc/1');
add_line(sub5, 'Referred_Count/1', 'Specialist_Review_Latency/1');
add_line(sub5, 'Specialist_Review_Latency/1', 'Backlog_Calc/2');
add_line(sub5, 'Backlog_Calc/1', 'Queue_Backlog/1');
add_line(sub5, 'Referred_Count/1', 'Report_Identity/1');
add_line(sub5, 'Report_Identity/1', 'Reports_Generated/1');

fprintf('  [6/6] Constructed Ophthalmologist Review Queue (<30s review, 4.5 min workload)\n');

%% 7. Interconnect Top-Level Subsystems
add_line(modelName, 'PHC_Fundus_Capture/1', 'Rural_Network_Upload/1');
add_line(modelName, 'PHC_Fundus_Capture/2', 'Rural_Network_Upload/2');
add_line(modelName, 'PHC_Fundus_Capture/3', 'Rural_Network_Upload/3');
add_line(modelName, 'Rural_Network_Upload/1', 'AI_Inference_Server/1');
add_line(modelName, 'AI_Inference_Server/1', 'Clinical_Triage/1');
add_line(modelName, 'AI_Inference_Server/2', 'Clinical_Triage/2');
add_line(modelName, 'Clinical_Triage/2', 'Ophthalmologist_Review_Queue/1');
add_line(modelName, 'Clinical_Triage/3', 'Ophthalmologist_Review_Queue/2');

%% 8. Add Telemetry Instrumentation, Displays, and Scopes
% Displays for live numerical readouts at workday end
add_block('simulink/Sinks/Display', [modelName '/Disp_Total_Captures'], 'Position', [100, 310, 200, 350]);
add_block('simulink/Sinks/Display', [modelName '/Disp_Uploaded_MB'],    'Position', [340, 310, 440, 350]);
add_block('simulink/Sinks/Display', [modelName '/Disp_AutoCleared'],    'Position', [810, 310, 910, 350]);
add_block('simulink/Sinks/Display', [modelName '/Disp_Referred'],       'Position', [930, 310, 1030, 350]);
add_block('simulink/Sinks/Display', [modelName '/Disp_Specialist_Min'], 'Position', [1100, 310, 1200, 350]);

add_line(modelName, 'PHC_Fundus_Capture/3', 'Disp_Total_Captures/1');
add_line(modelName, 'Rural_Network_Upload/2', 'Disp_Uploaded_MB/1');
add_line(modelName, 'Clinical_Triage/1', 'Disp_AutoCleared/1');
add_line(modelName, 'Clinical_Triage/2', 'Disp_Referred/1');
add_line(modelName, 'Ophthalmologist_Review_Queue/1', 'Disp_Specialist_Min/1');

% Multi-signal Scopes
add_block('simulink/Sinks/Scope', [modelName '/Scope_Telemedicine_Pipeline'], 'Position', [1320, 130, 1370, 210]);
set_param([modelName '/Scope_Telemedicine_Pipeline'], 'NumInputPorts', '4');

add_line(modelName, 'PHC_Fundus_Capture/3', 'Scope_Telemedicine_Pipeline/1');
add_line(modelName, 'Rural_Network_Upload/2', 'Scope_Telemedicine_Pipeline/2');
add_line(modelName, 'Clinical_Triage/1', 'Scope_Telemedicine_Pipeline/3');
add_line(modelName, 'Clinical_Triage/2', 'Scope_Telemedicine_Pipeline/4');

% Export To Workspace for automated CI/verification
add_block('simulink/Sinks/To Workspace', [modelName '/ToWorkspace_Summary'], ...
    'Position', [1320, 250, 1420, 280], ...
    'VariableName', 'telemedicine_metrics', ...
    'SaveFormat', 'Array');

% Mux summary outputs for logging
add_block('simulink/Signal Routing/Mux', [modelName '/Summary_Mux'], ...
    'Position', [1280, 235, 1285, 295], ...
    'Inputs', '5');

add_line(modelName, 'PHC_Fundus_Capture/3', 'Summary_Mux/1');
add_line(modelName, 'Rural_Network_Upload/2', 'Summary_Mux/2');
add_line(modelName, 'Clinical_Triage/1', 'Summary_Mux/3');
add_line(modelName, 'Clinical_Triage/2', 'Summary_Mux/4');
add_line(modelName, 'Ophthalmologist_Review_Queue/1', 'Summary_Mux/5');
add_line(modelName, 'Summary_Mux/1', 'ToWorkspace_Summary/1');

%% 9. Add Visual Annotations
try
    Simulink.Annotation([modelName '/Title_Annotation'], ...
        'Text', sprintf(['DIABETIC RETINOPATHY TELEMEDICINE SCREENING WORKFLOW (SIH26038)\n' ...
                 'Primary Health Centre (PHC) -> 2 Mbps Rural Link -> AI Inference -> 82%% Auto-Clear / 18%% Specialist Review']), ...
        'Position', [650, 40]);
catch
end

%% 10. Save and Compile Check
save_system(modelName, slxFile);
fprintf('Successfully constructed and saved model to:\n   %s\n', slxFile);

%% 11. Run Verification Simulation
fprintf('\nRunning 8-hour workday simulation verification...\n');
simOut = sim(modelName);

% Extract metrics at workday conclusion (t = 28,800 s)
rawMetrics = simOut.get('telemedicine_metrics');
if isstruct(rawMetrics) && isfield(rawMetrics, 'signals')
    metValues = rawMetrics.signals.values;
elseif isa(rawMetrics, 'timeseries')
    metValues = rawMetrics.Data;
else
    metValues = double(rawMetrics);
end
captured = metValues(end, 1);
uploadedMB = metValues(end, 2);
autoCleared = metValues(end, 3);
referred = metValues(end, 4);
specMinutes = metValues(end, 5);

fprintf('\n=================================================================\n');
fprintf('SIMULATION RESULTS SUMMARY (Workday = 8 hours / 28,800 s)\n');
fprintf('=================================================================\n');
fprintf('  Daily Images Captured   : %.1f  (Expected: 50.0)\n', captured);
fprintf('  Cumulative Data Uploaded: %.1f MB (Expected: ~24.4 MB)\n', uploadedMB);
fprintf('  Auto-Cleared Cases (82%%): %.1f  (Expected: 41.0)\n', autoCleared);
fprintf('  Referred DR Cases (18%%) : %.1f  (Expected: 9.0)\n', referred);
fprintf('  Specialist Workload     : %.1f min (Expected: 4.5 min)\n', specMinutes);
fprintf('=================================================================\n');

% Strict assertions
assert(abs(captured - 50) < 1e-3, 'Captured count deviates from 50.0');
assert(abs(uploadedMB - 24.414) < 0.1, 'Uploaded data deviates from ~24.4 MB');
assert(abs(autoCleared - 41) < 1e-3, 'Auto-cleared count deviates from 41.0 (82%%)');
assert(abs(referred - 9) < 1e-3, 'Referred count deviates from 9.0 (18%%)');
assert(abs(specMinutes - 4.5) < 1e-3, 'Specialist workload deviates from 4.5 min');

fprintf('ALL SIMULINK TELEMEDICINE WORKFLOW ASSERTIONS PASSED!\n\n');

% Close system neatly
close_system(modelName, 1);
