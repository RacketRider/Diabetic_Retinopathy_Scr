function result = retinai_infer(netTrained, imgProcessed, classNames)
% RETINAI_INFER Run deep learning inference on a preprocessed fundus image.
%
%   result = RETINAI_INFER(netTrained, imgProcessed)
%   result = RETINAI_INFER(netTrained, imgProcessed, classNames)
%
%   Runs inference on a single preprocessed fundus image using a trained
%   ResNet-101 dlnetwork. Returns class probabilities, predicted class,
%   referable DR probability, and screening referral determination.
%
%   Inputs:
%       netTrained   - Trained dlnetwork object (ResNet-101 architecture)
%       imgProcessed - 224x224x3 uint8 preprocessed RGB fundus image
%                      (single/double arrays in [0, 1] or [0, 255] also supported)
%       classNames   - (Optional) 1x5 cell array of class names. Default:
%                      {'Mild', 'Moderate', 'No_DR', 'Proliferate_DR', 'Severe'}
%
%   Output:
%       result - Struct containing inference results with fields:
%           .scores            - 1x5 double array of predicted class probabilities
%           .predIdx           - Index of predicted class (1 to 5)
%           .predClass         - string, predicted class name
%           .confidence        - double, confidence score as percentage (maxScore * 100)
%           .pRef              - double, referable DR probability (Moderate + Proliferate_DR + Severe)
%           .isReferable       - logical, screening decision based on argmax
%           .screeningCategory - string, 'REFERABLE' or 'NON-REFERABLE'
%           .classNames        - cell array of class names
%
%   Clinical Screening Reference:
%       - Referable DR: Moderate (col 2), Proliferate_DR (col 4), Severe (col 5)
%       - Non-Referable: Mild (col 1), No_DR (col 3)
%       - Validation-locked frozen threshold: pRef >= 0.1199 (for dataset evaluation)
%       - For single-image UI inference, isReferable uses argmax classification
%
%   Example:
%       load('checkpoints/AG_V4_5_ReducedAug.mat', 'netTrained');
%       res = retinai_infer(netTrained, imgPreprocessed);
%       fprintf('Diagnosis: %s (%.2f%% confidence)\n', res.predClass, res.confidence);
%       fprintf('Screening Decision: %s (pRef: %.4f)\n', res.screeningCategory, res.pRef);

% Validate input arguments
narginchk(2, 3);

% Default class names if not provided or empty
if nargin < 3 || isempty(classNames)
    classNames = {'Mild', 'Moderate', 'No_DR', 'Proliferate_DR', 'Severe'};
end

if isstring(classNames)
    classNames = cellstr(classNames);
end

try
    % Input validation
    if isempty(netTrained)
        error('retinai_infer:EmptyNetwork', 'The trained network object (netTrained) cannot be empty.');
    end
    
    if isempty(imgProcessed)
        error('retinai_infer:EmptyImage', 'The input image (imgProcessed) cannot be empty.');
    end

    % Ensure 3 color channels (RGB)
    if ndims(imgProcessed) == 2 || size(imgProcessed, 3) == 1
        imgProcessed = repmat(imgProcessed(:, :, 1), [1, 1, 3]);
    elseif size(imgProcessed, 3) ~= 3
        error('retinai_infer:InvalidChannels', ...
            'Input image must have 3 channels (RGB). Found %d channels.', size(imgProcessed, 3));
    end

    % Ensure spatial dimensions are 224x224
    if size(imgProcessed, 1) ~= 224 || size(imgProcessed, 2) ~= 224
        imgProcessed = imresize(imgProcessed, [224, 224]);
    end

    % 1. Convert image to single precision and scale to [0, 1]
    if isa(imgProcessed, 'uint8')
        imgSingle = single(imgProcessed) / 255.0;
    elseif isfloat(imgProcessed)
        if max(imgProcessed(:)) > 1.0
            imgSingle = single(imgProcessed) / 255.0;
        else
            imgSingle = single(imgProcessed);
        end
    else
        imgSingle = single(imgProcessed) / 255.0;
    end

    % 2. Create a dlarray with format 'SSCB' (spatial-spatial-channel-batch)
    dlX = dlarray(imgSingle, 'SSCB');

    % If GPU is available, the dlarray will automatically use GPU
    hasGPU = false;
    try
        if (exist('canUseGPU', 'builtin') == 5 || exist('canUseGPU', 'file') == 2) && canUseGPU()
            hasGPU = true;
        elseif (exist('gpuDeviceCount', 'builtin') == 5 || exist('gpuDeviceCount', 'file') == 2) && gpuDeviceCount() > 0
            hasGPU = true;
        end
    catch
        hasGPU = false;
    end

    if hasGPU
        try
            dlX = gpuArray(dlX);
        catch
            % If GPU memory allocation fails, proceed gracefully on CPU
        end
    end

    % 3. Run prediction using dlnetwork predict
    rawScores = predict(netTrained, dlX);

    % 4. Extract scores as a regular numeric array (gathering from GPU if needed)
    scores = gather(extractdata(rawScores));

    % 5. Apply softmax if needed (check if scores sum to ~1; if not, apply softmax)
    scores = double(scores(:)'); % Ensure 1xD double row vector
    scoresSum = sum(scores);
    if abs(scoresSum - 1.0) > 0.05 || any(scores < 0)
        % Apply numerically stable softmax across classes
        shiftScores = scores - max(scores);
        expScores = exp(shiftScores);
        scores = expScores / sum(expScores);
    end

    % 6. Ensure scores is a row vector
    scores = scores(:)';

    % 7. Find predicted class: [maxScore, predIdx] = max(scores)
    [maxScore, predIdx] = max(scores);
    predIdx = double(predIdx);

    % 8. Get predicted class name: predClass = classNames{predIdx}
    predClass = classNames{predIdx};

    % 9. Compute referable probability: Moderate (2), Proliferate_DR (4), Severe (5)
    refClasses = {'Moderate', 'Proliferate_DR', 'Severe'};
    refIndices = find(ismember(classNames, refClasses));
    if isempty(refIndices) || numel(refIndices) ~= 3
        refIndices = [2, 4, 5]; % Standard class index fallback
    end
    pRef = double(sum(scores(refIndices)));

    % 10. Validation-locked frozen threshold for screening decision:
    % thresholdLocked = 0.1199;
    % isReferableThresh = (pRef >= thresholdLocked);

    % 11. Argmax-based referability:
    % Single-image UI inference uses argmax-based referability:
    % Referable if predicted class is 'Moderate', 'Proliferate_DR', or 'Severe'
    isReferable = ismember(predClass, refClasses) || ismember(predIdx, refIndices);
    isReferable = logical(isReferable);

    % Determine screening category
    if isReferable
        screeningCategory = string("REFERABLE");
    else
        screeningCategory = string("NON-REFERABLE");
    end

    % Construct output struct
    result = struct();
    result.scores            = double(scores);                % 1x5 double array of probabilities
    result.predIdx           = predIdx;                       % index of predicted class (1-5)
    result.predClass         = string(predClass);             % string, predicted class name
    result.confidence        = double(maxScore * 100.0);      % double, max score percentage
    result.pRef              = double(pRef);                  % double, referable probability
    result.isReferable       = isReferable;                   % logical, screening decision based on argmax
    result.screeningCategory = screeningCategory;             % string, 'REFERABLE' or 'NON-REFERABLE'
    result.classNames        = classNames;                    % cell array of class names

catch ME
    error('retinai_infer:ExecutionError', ...
        'Failed to execute diabetic retinopathy inference: %s', ME.message);
end

end
