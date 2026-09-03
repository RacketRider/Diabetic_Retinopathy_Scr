function ag_log_experiment(expStruct)
% AG_LOG_EXPERIMENT Append one experiment to the MAT and text logs.

paths = ag_project_paths();
if ~isfolder(paths.logs)
    mkdir(paths.logs);
end
logMatPath = fullfile(paths.logs, 'AG_experiment_log.mat');
logTxtPath = fullfile(paths.logs, 'AG_experiment_log.txt');

if isfile(logMatPath)
    previous = load(logMatPath, 'experimentLog');
    experimentLog = previous.experimentLog;
    experimentLog(end + 1) = expStruct;
else
    experimentLog = expStruct;
end

[fid, message] = fopen(logTxtPath, 'a');
if fid == -1
    error('ag:log:OpenFailed', 'Cannot open %s: %s', logTxtPath, message);
end
closeFile = onCleanup(@() fclose(fid));
save(logMatPath, 'experimentLog');

fprintf(fid, '--------------------------------------------------------------------------------\n');
fprintf(fid, 'Experiment ID:      %s\n', expStruct.ID);
fprintf(fid, 'Timestamp:          %s\n', expStruct.dateTime);
fprintf(fid, 'Architecture:       %s (%s params)\n', expStruct.architecture, num2str(expStruct.parameterCount));
fprintf(fid, 'Initial Checkpoint: %s\n', expStruct.initialCheckpoint);
fprintf(fid, 'Hyperparameters:    LR=%g, Opt=%s, Batch=%d, Epochs=%d, Loss=%s\n', ...
    expStruct.learningRate, expStruct.optimizer, expStruct.batchSize, expStruct.epochs, expStruct.loss);
fprintf(fid, 'Class Weights:      %s\n', expStruct.classWeights);
fprintf(fid, 'Augmentation:       %s\n', expStruct.augmentation);
fprintf(fid, 'Input Size:         %s\n', mat2str(expStruct.inputSize));
fprintf(fid, 'Validation Res:     Sens=%.2f%%, Spec=%.2f%%, ROC-AUC=%.4f, PR-AUC=%.4f, Thresh=%.4f\n', ...
    expStruct.valSensitivity*100, expStruct.valSpecificity*100, expStruct.valRocAuc, expStruct.valPrAuc, expStruct.threshold);
fprintf(fid, 'Clinical Res:       Acc=%.2f%%, PPV=%.2f%%, NPV=%.2f%%, 5ClassAcc=%.2f%%\n', ...
    expStruct.valClinicalAccuracy*100, expStruct.valPPV*100, expStruct.valNPV*100, expStruct.valAccuracy*100);
fprintf(fid, 'Decision:           %s\n', expStruct.decision);
fprintf(fid, 'Notes:              %s\n', expStruct.notes);
fprintf(fid, '--------------------------------------------------------------------------------\n\n');

fprintf('Successfully logged experiment %s\n', expStruct.ID);
end
