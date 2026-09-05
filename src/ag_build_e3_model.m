function model = ag_build_e3_model(initialNet, cfg, classNames)
% AG_BUILD_E3_MODEL Build genuine global/local feature fusion from a V3 checkpoint.

if ~isa(initialNet, 'dlnetwork')
    error('ag:e3model:InvalidCheckpoint', 'E3 initialization must be a dlnetwork.');
end
if ~any(string({initialNet.Layers.Name}) == "pool5") || ...
        ~any(string({initialNet.Layers.Name}) == string(cfg.localFeatureLayer))
    error('ag:e3model:IncompatibleBackbone', ...
        'The V3 checkpoint lacks required ResNet feature layers pool5/%s.', cfg.localFeatureLayer);
end

globalNet = trimAfter(initialNet, "pool5");
globalNet = initialize(globalNet, ...
    dlarray(zeros(cfg.inputResolution, cfg.inputResolution, 3, 1, 'single'), 'SSCB'));
localBase = ag_build_v3_network(initialNet, cfg.patchSize(1), classNames, false);
localNet = trimAfter(localBase, string(cfg.localFeatureLayer));
localHead = [globalAveragePooling2dLayer(Name='e3_local_pool'); ...
    fullyConnectedLayer(cfg.localFeatureDim, Name='e3_local_projection'); ...
    reluLayer(Name='e3_local_relu')];
localNet = addLayers(localNet, localHead);
localNet = connectLayers(localNet, char(cfg.localFeatureLayer), 'e3_local_pool');
localNet = initialize(localNet, ...
    dlarray(zeros(cfg.patchSize(1), cfg.patchSize(2), 3, 1, 'single'), 'SSCB'));

probeGlobal = predict(globalNet, ...
    dlarray(zeros(cfg.inputResolution, cfg.inputResolution, 3, 1, 'single'), 'SSCB'));
probeLocal = predict(localNet, ...
    dlarray(zeros(cfg.patchSize(1), cfg.patchSize(2), 3, 1, 'single'), 'SSCB'));
globalFeatureDim = numel(probeGlobal);
localFeatureDim = numel(probeLocal);
if globalFeatureDim ~= 2048 || localFeatureDim ~= cfg.localFeatureDim
    error('ag:e3model:FeatureDimensionMismatch', ...
        'Expected 2048 global and %d local features; got %d and %d.', ...
        cfg.localFeatureDim, globalFeatureDim, localFeatureDim);
end
fusionInputDim = globalFeatureDim + localFeatureDim;
fusionLayers = [featureInputLayer(fusionInputDim, Normalization='none', Name='e3_fused'); ...
    fullyConnectedLayer(cfg.fusionDim, Name='e3_fusion_fc'); ...
    reluLayer(Name='e3_fusion_relu'); ...
    dropoutLayer(cfg.dropout, Name='e3_fusion_dropout'); ...
    fullyConnectedLayer(numel(classNames), Name='e3_logits')];
fusionNet = dlnetwork(fusionLayers);
assertBackboneWeightInherited(initialNet, globalNet, "global");
assertBackboneWeightInherited(initialNet, localNet, "local");
assertClassifierRemoved(initialNet, globalNet, localNet);
fprintf('[PASS] pool5 feature extraction\n');
fprintf('[PASS] %s local encoder extraction\n', cfg.localFeatureLayer);
fprintf('[PASS] global feature dim = %d\n', globalFeatureDim);
fprintf('[PASS] local feature dim = %d\n', localFeatureDim);
fprintf('[PASS] fusion dim = %d\n', cfg.fusionDim);
model = struct('globalNet', globalNet, 'localNet', localNet, 'fusionNet', fusionNet, ...
    'globalFeatureDim', globalFeatureDim, 'localFeatureDim', localFeatureDim, ...
    'fusionFeatureDim', cfg.fusionDim, 'aggregationMethod', "mean");
end

function net = trimAfter(net, terminalLayer)
connections = net.Connections;
frontier = string(terminalLayer);
toRemove = strings(0, 1);
while ~isempty(frontier)
    sourceNames = stripPort(string(connections.Source));
    destinations = string(connections.Destination(ismember(sourceNames, frontier)));
    destinations = unique(stripPort(destinations));
    destinations = setdiff(destinations, [string(terminalLayer); toRemove]);
    if isempty(destinations), break; end
    toRemove = unique([toRemove; destinations]);
    frontier = destinations;
end
if ~isempty(toRemove)
    net = removeLayers(net, cellstr(toRemove));
end
end

function names = stripPort(names)
names = extractBefore(names + "/", "/");
end

function assertBackboneWeightInherited(sourceNet, targetNet, streamName)
source = sourceNet.Learnables;
target = targetNet.Learnables;
sourceRow = source.Layer == "conv1" & source.Parameter == "Weights";
targetRow = target.Layer == "conv1" & target.Parameter == "Weights";
if nnz(sourceRow) ~= 1 || nnz(targetRow) ~= 1
    error('ag:e3model:InheritanceDiagnostic', ...
        'Cannot locate representative conv1 weights in the %s stream.', streamName);
end
sourceWeight = gather(extractdata(source.Value{sourceRow}));
targetWeight = gather(extractdata(target.Value{targetRow}));
if ~isequaln(sourceWeight, targetWeight)
    error('ag:e3model:InheritanceMismatch', ...
        'The E3 %s stream did not inherit checkpoint conv1 weights.', streamName);
end
fprintf('[PASS] E3 %s stream inherited E1 weights\n', streamName);
end

function assertClassifierRemoved(sourceNet, globalNet, localNet)
sourceLayers = sourceNet.Layers;
isClassifier = arrayfun(@(layer) isa(layer, 'nnet.cnn.layer.FullyConnectedLayer'), sourceLayers);
classifierNames = string({sourceLayers(isClassifier).Name});
globalNames = string({globalNet.Layers.Name});
localNames = string({localNet.Layers.Name});
if any(ismember(classifierNames, globalNames)) || any(ismember(classifierNames, localNames))
    error('ag:e3model:ClassifierLeakage', ...
        'The source checkpoint classifier must not be retained in E3 feature streams.');
end
fprintf('[PASS] E1 classifier excluded from E3 feature streams\n');
end
