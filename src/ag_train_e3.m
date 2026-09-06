function [model, trainingInfo] = ag_train_e3(model, trainData, validationData, cfg, checkpointPaths, mapping, classNames)
% AG_TRAIN_E3 Two-stage resumable custom training loop for global/local fusion.

velocity = struct('global', [], 'local', [], 'fusion', []);
trainingInfo = struct('Epoch', [], 'TrainingLoss', [], ...
    'ValidationBalancedAccuracy', [], 'Stage', strings(0, 1));
bestValidationMetric = -Inf;
iteration = 0; startEpoch = 1;
if isfile(checkpointPaths.latest)
    saved = load(checkpointPaths.latest);
    if isfield(saved, 'config') && isfield(saved.config, 'initialCheckpoint') && ...
            string(saved.config.initialCheckpoint) ~= string(cfg.initialCheckpoint)
        error('ag:e3train:InitializationMismatch', ...
            ['E3 checkpoint initialization mismatch:\nsaved  = %s\ncurrent = %s\n' ...
            'Move or rename the stale E3 model checkpoint; walker caches remain reusable.'], ...
            string(saved.config.initialCheckpoint), string(cfg.initialCheckpoint));
    end
    required = {'model','trainingInfo','config','mapping','classNames', ...
        'velocity','bestValidationMetric','epoch','iteration'};
    if ~all(isfield(saved, required)) || ...
            string(saved.config.resumeSignature) ~= string(cfg.resumeSignature) || ...
            ~sameMapping(saved.mapping, mapping) || ...
            ~isequal(string(saved.classNames(:)), string(classNames(:)))
        error('ag:e3train:InvalidResumeCheckpoint', ...
            'E3 latest checkpoint exists but does not match this configuration/split.');
    end
    model = saved.model; trainingInfo = saved.trainingInfo; velocity = saved.velocity;
    bestValidationMetric = saved.bestValidationMetric;
    iteration = saved.iteration; startEpoch = saved.epoch + 1;
    fprintf('E3 checkpoint detected\nepoch: %d\nvalidation metric: %.6f\n', ...
        saved.epoch, bestValidationMetric);
end
if startEpoch > cfg.epochs
    if ~isfile(checkpointPaths.best)
        error('ag:e3train:MissingBestCheckpoint', 'Completed latest checkpoint has no best checkpoint.');
    end
    best = load(checkpointPaths.best, 'model'); model = best.model;
    fprintf('[E3 TRAIN] Requested %d epochs already complete; using best checkpoint.\n', cfg.epochs);
    return;
end

for epoch = startEpoch:cfg.epochs
    rng(cfg.seed + epoch, 'twister');
    order = randperm(numel(trainData.files));
    if epoch <= cfg.freezeEpochs
        stage = "A"; learnRate = cfg.learningRate;
    else
        stage = "B"; learnRate = cfg.fineTuneLearningRate;
    end
    epochLoss = zeros(ceil(numel(order) / cfg.batchSize), 1);
    batchNumber = 0;
    fprintf('[E3 TRAIN] Epoch %d/%d | stage %s | LR %.3g\n', epoch, cfg.epochs, stage, learnRate);
    for first = 1:cfg.batchSize:numel(order)
        batchNumber = batchNumber + 1; iteration = iteration + 1;
        indices = order(first:min(first + cfg.batchSize - 1, numel(order)));
        [globalBatch, patchBatch, targets] = ag_e3_minibatch(trainData, indices, cfg, true, false);
        if cfg.useGPU
            globalBatch = gpuArray(globalBatch); patchBatch = gpuArray(patchBatch); targets = gpuArray(targets);
        end
        dlGlobal = dlarray(globalBatch, 'SSCB');
        dlPatches = dlarray(patchBatch, 'SSCB');
        dlTargets = dlarray(targets, 'CB');
        [loss, gradients, states] = dlfeval(@ag_e3_model_gradients, ...
            model, dlGlobal, dlPatches, dlTargets, cfg, stage);
        assertFinite(loss, gradients, stage);
        model.localNet.State = states.local;
        if stage == "B"
            model.globalNet.State = states.global;
            gradients.global = prepareGradients(gradients.global, model.globalNet.Learnables, ...
                ["res5" "bn5"], cfg.l2Regularization);
            gradients.local = prepareGradients(gradients.local, model.localNet.Learnables, ...
                ["res3" "bn3" "e3_"], cfg.l2Regularization);
            [model.globalNet, velocity.global] = sgdmupdate(model.globalNet, gradients.global, ...
                velocity.global, learnRate, cfg.momentum);
            [model.localNet, velocity.local] = sgdmupdate(model.localNet, gradients.local, ...
                velocity.local, learnRate, cfg.momentum);
        else
            gradients.local = prepareGradients(gradients.local, model.localNet.Learnables, ...
                "e3_", cfg.l2Regularization);
            [model.localNet, velocity.local] = sgdmupdate(model.localNet, gradients.local, ...
                velocity.local, learnRate, cfg.momentum);
        end
        gradients.fusion = prepareGradients(gradients.fusion, model.fusionNet.Learnables, ...
            "", cfg.l2Regularization);
        [model.fusionNet, velocity.fusion] = sgdmupdate(model.fusionNet, gradients.fusion, ...
            velocity.fusion, learnRate, cfg.momentum);
        epochLoss(batchNumber) = double(gather(extractdata(loss)));
        if mod(batchNumber, cfg.verboseFrequency) == 0 || first + cfg.batchSize > numel(order)
            fprintf('[E3 TRAIN] epoch %d batch %d/%d loss %.5f\n', epoch, batchNumber, ...
                numel(epochLoss), mean(epochLoss(1:batchNumber)));
        end
    end

    validationPrediction = ag_e3_predict(model, validationData, cfg, "full");
    validationMetric = balancedAccuracy(validationPrediction.scores.full, ...
        validationData.labels, classNames);
    trainingInfo.Epoch(end + 1, 1) = epoch;
    trainingInfo.TrainingLoss(end + 1, 1) = mean(epochLoss);
    trainingInfo.ValidationBalancedAccuracy(end + 1, 1) = validationMetric;
    trainingInfo.Stage(end + 1, 1) = stage;
    improved = validationMetric > bestValidationMetric;
    if improved
        bestValidationMetric = validationMetric;
    end
    saveCheckpoint(checkpointPaths.latest, model, trainingInfo, cfg, mapping, classNames, ...
        velocity, bestValidationMetric, epoch, iteration);
    if improved
        saveCheckpoint(checkpointPaths.best, model, trainingInfo, cfg, mapping, classNames, ...
            velocity, bestValidationMetric, epoch, iteration);
        fprintf('[E3 TRAIN] New best validation balanced accuracy: %.4f\n', validationMetric);
    else
        fprintf('[E3 TRAIN] Validation balanced accuracy %.4f; best remains %.4f\n', ...
            validationMetric, bestValidationMetric);
    end
end
best = load(checkpointPaths.best, 'model'); model = best.model;
copyfile(checkpointPaths.best, checkpointPaths.final, 'f');
end

function gradients = prepareGradients(gradients, learnables, allowedPrefixes, l2)
layers = string(gradients.Layer);
if ~(isscalar(allowedPrefixes) && allowedPrefixes == "")
    allowed = false(height(gradients), 1);
    for prefix = allowedPrefixes
        allowed = allowed | startsWith(layers, prefix);
    end
else
    allowed = true(height(gradients), 1);
end
for i = 1:height(gradients)
    if isempty(gradients.Value{i}), continue; end
    if ~allowed(i)
        gradients.Value{i} = gradients.Value{i} * 0;
    elseif string(gradients.Parameter(i)) == "Weights"
        gradients.Value{i} = gradients.Value{i} + l2 * learnables.Value{i};
    end
end
end

function assertFinite(loss, gradients, stage)
if ~isfinite(double(gather(extractdata(loss))))
    error('ag:e3train:NonFiniteLoss', 'E3 training produced a non-finite loss.');
end
groups = {'local','fusion'};
if stage == "B", groups = {'global','local','fusion'}; end
for g = 1:numel(groups)
    tableValue = gradients.(groups{g});
    for i = 1:height(tableValue)
        value = tableValue.Value{i};
        if ~isempty(value) && ~all(gather(isfinite(extractdata(value))), 'all')
            error('ag:e3train:NonFiniteGradient', 'Non-finite %s gradient.', groups{g});
        end
    end
end
end

function value = balancedAccuracy(scores, labels, classNames)
[~, index] = max(scores, [], 2); predictions = classNames(index);
recall = NaN(numel(classNames), 1);
for i = 1:numel(classNames)
    truth = string(labels) == classNames(i);
    recall(i) = sum(truth & predictions == classNames(i)) / sum(truth);
end
value = mean(recall, 'omitnan');
end

function saveCheckpoint(path, model, trainingInfo, config, mapping, classNames, ...
        velocity, bestValidationMetric, epoch, iteration)
directory = fileparts(path); if ~isfolder(directory), mkdir(directory); end
save(path, 'model', 'trainingInfo', 'config', 'mapping', 'classNames', ...
    'velocity', 'bestValidationMetric', 'epoch', 'iteration', '-v7.3');
end

function tf = sameMapping(left, right)
fields = {'trainFiles','validationFiles','testFiles'};
tf = all(isfield(left, fields)) && all(isfield(right, fields));
for i = 1:numel(fields)
    tf = tf && isequal(string(left.(fields{i})(:)), string(right.(fields{i})(:)));
end
end
