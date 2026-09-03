function loss = ag_focal_loss(Y, T, gamma, alpha)
% AG_FOCAL_LOSS Multi-class focal loss for one-hot classification targets.
% alpha may be scalar or already shaped to broadcast across the class axis.

if nargin < 3 || isempty(gamma)
    gamma = 2.0;
end
if nargin < 4
    alpha = [];
end
if ~isscalar(gamma) || gamma < 0
    error('ag:focal:InvalidGamma', 'gamma must be a nonnegative scalar.');
end

Y = min(max(Y, 1e-7), 1 - 1e-7);
lossPerElement = -T .* ((1 - Y) .^ gamma) .* log(Y);
if ~isempty(alpha)
    lossPerElement = alpha .* lossPerElement;
end

% sum(T,"all") equals batch size for one-hot targets in either CB or BC format.
loss = sum(lossPerElement, "all") / sum(T, "all");
end
