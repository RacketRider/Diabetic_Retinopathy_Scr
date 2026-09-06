function net = ag_build_v3_network(net, targetSize, classNames, addReferableHead)
% AG_BUILD_V3_NETWORK Resize the input and optionally create 5+1 raw logits.

if nargin < 4
    addReferableHead = false;
end
classNames = string(classNames(:));
if numel(classNames) ~= 5
    error('ag:v3net:InvalidClasses', 'Exactly five class names are required.');
end
layers = net.Layers;
inputIndex = find(arrayfun(@(layer) isa(layer, 'nnet.cnn.layer.ImageInputLayer'), layers), 1);
if isempty(inputIndex)
    error('ag:v3net:NoImageInput', 'Network has no image input layer.');
end
oldInput = layers(inputIndex);
inputArguments = {'Name', oldInput.Name, 'Normalization', oldInput.Normalization};
for property = ["Mean", "StandardDeviation", "Min", "Max"]
    if isprop(oldInput, property) && ~isempty(oldInput.(property))
        val = oldInput.(property);
        if (size(val, 1) ~= targetSize || size(val, 2) ~= targetSize) && (size(val, 1) > 1 || size(val, 2) > 1)
            if exist('imresize', 'file')
                val = imresize(val, [targetSize targetSize]);
            else
                val = mean(val, [1 2]);
            end
        end
        inputArguments = [inputArguments, {char(property), val}]; %#ok<AGROW>
    end
end
newInput = imageInputLayer([targetSize targetSize 3], inputArguments{:});
net = replaceLayer(net, oldInput.Name, newInput);
if ~addReferableHead
    if ~net.Initialized
        dummyInput = dlarray(zeros([targetSize targetSize 3 1], 'single'), 'SSCB');
        net = initialize(net, dummyInput);
    end
    return;
end

layers = net.Layers;
fcIndices = find(arrayfun(@(layer) isa(layer, 'nnet.cnn.layer.FullyConnectedLayer') && layer.OutputSize == 5, layers));
if isempty(fcIndices)
    error('ag:v3net:NoSeverityHead', 'Could not find the five-class fully connected layer.');
end
oldHead = layers(fcIndices(end));
referable = ismember(classNames, ["Moderate", "Severe", "Proliferate_DR"]);
nonReferable = ~referable;
newHead = fullyConnectedLayer(6, 'Name', 'v3_multitask_logits');
newHead.Weights = [oldHead.Weights; ...
    mean(oldHead.Weights(referable, :), 1) - mean(oldHead.Weights(nonReferable, :), 1)];
newHead.Bias = [oldHead.Bias; mean(oldHead.Bias(referable)) - mean(oldHead.Bias(nonReferable))];

% Remove softmax/classification descendants: the custom loss consumes logits.
connections = net.Connections;
toRemove = strings(0, 1);
frontier = string(oldHead.Name);
while ~isempty(frontier)
    destinations = string(connections.Destination(ismember(string(connections.Source), frontier)));
    destinations = unique(stripPort(destinations));
    destinations = setdiff(destinations, [string(oldHead.Name); toRemove]);
    if isempty(destinations)
        break;
    end
    toRemove = unique([toRemove; destinations]);
    frontier = destinations;
end
net = replaceLayer(net, oldHead.Name, newHead);
if ~isempty(toRemove)
    net = removeLayers(net, cellstr(toRemove));
end
if ~net.Initialized
    dummyInput = dlarray(zeros([targetSize targetSize 3 1], 'single'), 'SSCB');
    net = initialize(net, dummyInput);
end
end

function names = stripPort(names)
names = extractBefore(names + "/", "/");
end
