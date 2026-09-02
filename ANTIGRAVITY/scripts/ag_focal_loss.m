function loss = ag_focal_loss(Y, T, gamma, alpha)
% AG_FOCAL_LOSS Multi-class Focal Loss for trainnet in MATLAB
% Y: predictions (probabilities from softmax, formatted CB or BC)
% T: one-hot target ground truth
if nargin < 3 || isempty(gamma)
    gamma = 2.0;
end
if nargin < 4
    alpha = [];
end

% Clip Y for stability
epsVal = 1e-7;
Y = max(Y, epsVal);
Y = min(Y, 1 - epsVal);

% Focal term: (1 - Y)^gamma * log(Y)
focalWeight = (1 - Y) .^ gamma;
lossPerElem = - T .* focalWeight .* log(Y);

if ~isempty(alpha)
    lossPerElem = alpha .* lossPerElem;
end

% Normalize by batch size (sum over classes, mean over batch)
loss = sum(lossPerElem, "all") / size(Y, ndims(Y));
end
