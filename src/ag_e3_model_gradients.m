function [loss, gradients, states] = ag_e3_model_gradients(model, dlGlobal, dlPatches, targets, cfg, stage)
% AG_E3_MODEL_GRADIENTS Forward/backward pass for frozen-head or joint fine-tuning.

batchSize = size(targets, 2);
if stage == "A"
    globalFeatures = predict(model.globalNet, dlGlobal);
    [localFeatures, localState] = forward(model.localNet, dlPatches);
    logits = fuse(model, globalFeatures, localFeatures, cfg.numPatches, batchSize, true);
    loss = crossentropy(softmax(logits), targets);
    [localGradients, fusionGradients] = dlgradient(loss, ...
        model.localNet.Learnables, model.fusionNet.Learnables);
    gradients = struct('global', [], 'local', localGradients, 'fusion', fusionGradients);
    states = struct('global', model.globalNet.State, 'local', localState);
elseif stage == "B"
    [globalFeatures, globalState] = forward(model.globalNet, dlGlobal);
    [localFeatures, localState] = forward(model.localNet, dlPatches);
    logits = fuse(model, globalFeatures, localFeatures, cfg.numPatches, batchSize, true);
    loss = crossentropy(softmax(logits), targets);
    [globalGradients, localGradients, fusionGradients] = dlgradient(loss, ...
        model.globalNet.Learnables, model.localNet.Learnables, model.fusionNet.Learnables);
    gradients = struct('global', globalGradients, 'local', localGradients, ...
        'fusion', fusionGradients);
    states = struct('global', globalState, 'local', localState);
else
    error('ag:e3train:InvalidStage', 'Training stage must be A or B.');
end
end

function logits = fuse(model, globalFeatures, localFeatures, patchCount, batchSize, training)
globalFeatures = dlarray(reshape(stripdims(globalFeatures), [], batchSize), 'CB');
localFeatures = reshape(stripdims(localFeatures), [], patchCount, batchSize);
localFeatures = dlarray(mean(localFeatures, 2), 'CB');
fused = cat(1, globalFeatures, localFeatures);
if training
    logits = forward(model.fusionNet, fused);
else
    logits = predict(model.fusionNet, fused);
end
end
