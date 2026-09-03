function [imgProcessed, imgInfo] = retinai_preprocess(imgPath)
% RETINAI_PREPROCESS Preprocess a single retinal fundus image for ResNet-101.
%
%   [imgProcessed, imgInfo] = RETINAI_PREPROCESS(imgPath) loads a retinal
%   fundus image from imgPath, validates its format and integrity, extracts
%   its original metadata into imgInfo, resizes the image to [224 224] using
%   bilinear interpolation, converts grayscale images to 3-channel RGB,
%   ensures uint8 precision, and returns the preprocessed image ready for
%   evaluation by the ResNet-101 screening model.
%
%   Inputs:
%       imgPath      - Path to the image file (character vector or string scalar).
%                      Supported formats: .png, .jpg, .jpeg, .bmp, .tif, .tiff.
%
%   Outputs:
%       imgProcessed - Preprocessed 224x224x3 uint8 RGB image matrix.
%       imgInfo      - Struct containing original image metadata:
%                      * filename : File name of the image (string scalar)
%                      * width    : Original width in pixels (double)
%                      * height   : Original height in pixels (double)
%                      * channels : Original number of color channels (double)
%                      * format   : Image format / file extension (char vector, e.g. 'png')
%                      * filesize : Size of the image file on disk in bytes (double)
%
%   Example:
%       [img, info] = retinai_preprocess('archive (1)/colored_images/colored_images/Mild/10030_left.png');
%       imshow(img);
%       fprintf('Original: %dx%d (%d channels), Format: %s\n', ...
%           info.width, info.height, info.channels, info.format);
%
%   See also IMREAD, IMRESIZE, REPMAT.

    % ---------------------------------------------------------------------
    % 1. Input Validation
    % ---------------------------------------------------------------------
    if nargin < 1 || isempty(imgPath)
        error('retinai_preprocess:InvalidInput', ...
            'Image path must be provided as a non-empty string or character vector.');
    end

    % Normalize input path to character vector
    if isstring(imgPath)
        if ~isscalar(imgPath)
            error('retinai_preprocess:InvalidInput', ...
                'Image path must be a scalar string, not a string array.');
        end
        imgPathChar = char(imgPath);
    elseif ischar(imgPath)
        imgPathChar = imgPath;
    else
        error('retinai_preprocess:InvalidInput', ...
            'Image path must be a string scalar or character vector.');
    end

    imgPathChar = strtrim(imgPathChar);
    if isempty(imgPathChar)
        error('retinai_preprocess:InvalidInput', ...
            'Image path cannot be empty or whitespace only.');
    end

    % ---------------------------------------------------------------------
    % 2. Validate File Existence
    % ---------------------------------------------------------------------
    if exist(imgPathChar, 'file') ~= 2 || exist(imgPathChar, 'dir') == 7
        error('retinai_preprocess:FileNotFound', ...
            'Image file does not exist or is a directory: %s', imgPathChar);
    end

    % ---------------------------------------------------------------------
    % 3. Validate File Format & Extension
    % ---------------------------------------------------------------------
    [~, baseName, fileExt] = fileparts(imgPathChar);
    if isempty(fileExt)
        error('retinai_preprocess:UnsupportedFormat', ...
            'Image file has no extension: %s', imgPathChar);
    end

    % Extract clean lowercase extension without leading dot
    extClean = lower(fileExt);
    if startsWith(extClean, '.')
        extClean = extClean(2:end);
    end

    supportedFormats = {'png', 'jpg', 'jpeg', 'bmp', 'tif', 'tiff'};
    if ~ismember(extClean, supportedFormats)
        error('retinai_preprocess:UnsupportedFormat', ...
            'Unsupported image format ''.%s''. Supported formats: %s', ...
            extClean, strjoin(supportedFormats, ', '));
    end

    % Retrieve file size on disk
    fileMetadata = dir(imgPathChar);
    if isempty(fileMetadata) || fileMetadata(1).isdir
        error('retinai_preprocess:FileNotFound', ...
            'Unable to read file metadata for: %s', imgPathChar);
    end
    fileSizeBytes = fileMetadata(1).bytes;

    % ---------------------------------------------------------------------
    % 4. Read Image with Error Handling
    % ---------------------------------------------------------------------
    try
        imgRaw = imread(imgPathChar);
    catch ME
        error('retinai_preprocess:ReadError', ...
            'Failed to read image file ''%s''. The file may be corrupted or invalid: %s', ...
            imgPathChar, ME.message);
    end

    if isempty(imgRaw)
        error('retinai_preprocess:CorruptImage', ...
            'Image file ''%s'' was read as empty.', imgPathChar);
    end

    if ndims(imgRaw) < 2 || ndims(imgRaw) > 3
        error('retinai_preprocess:InvalidDimensions', ...
            'Image ''%s'' has %d dimensions. Expected 2D or 3D image.', ...
            imgPathChar, ndims(imgRaw));
    end

    % ---------------------------------------------------------------------
    % 5. Store Original Dimensions & Metadata in imgInfo
    % ---------------------------------------------------------------------
    origHeight   = size(imgRaw, 1);
    origWidth    = size(imgRaw, 2);
    origChannels = size(imgRaw, 3);

    if origHeight == 0 || origWidth == 0
        error('retinai_preprocess:InvalidDimensions', ...
            'Image ''%s'' has zero width or height: [%d x %d].', ...
            imgPathChar, origHeight, origWidth);
    end

    imgInfo = struct();
    imgInfo.filename = string([baseName, fileExt]);
    imgInfo.width    = origWidth;
    imgInfo.height   = origHeight;
    imgInfo.channels = origChannels;
    imgInfo.format   = extClean;
    imgInfo.filesize = fileSizeBytes;

    % ---------------------------------------------------------------------
    % 6. Resize Image to [224 224] using Bilinear Interpolation
    % ---------------------------------------------------------------------
    try
        imgResized = imresize(imgRaw, [224 224], 'bilinear');
    catch ME
        error('retinai_preprocess:ResizeError', ...
            'Failed to resize image ''%s'' to [224 224]: %s', ...
            imgPathChar, ME.message);
    end

    % ---------------------------------------------------------------------
    % 7. Convert Grayscale (2D) to RGB (3 Channels)
    % ---------------------------------------------------------------------
    if size(imgResized, 3) == 1
        imgRGB = repmat(imgResized, [1 1 3]);
    elseif size(imgResized, 3) == 3
        imgRGB = imgResized;
    elseif size(imgResized, 3) == 4
        % Discard alpha channel for 4-channel RGBA images
        imgRGB = imgResized(:, :, 1:3);
    else
        error('retinai_preprocess:InvalidChannels', ...
            'Unsupported number of image channels (%d). Expected 1 (grayscale) or 3 (RGB).', ...
            size(imgResized, 3));
    end

    % ---------------------------------------------------------------------
    % 8. Ensure uint8 Precision
    % ---------------------------------------------------------------------
    if isa(imgRGB, 'uint8')
        imgProcessed = imgRGB;
    else
        try
            if islogical(imgRGB)
                imgProcessed = uint8(imgRGB) * uint8(255);
            elseif isa(imgRGB, 'uint16')
                % Scale 16-bit dynamic range [0, 65535] down to 8-bit [0, 255]
                imgProcessed = uint8(round(double(imgRGB) / 65535 * 255));
            elseif isfloat(imgRGB)
                % Check if floating point values are in [0, 1] or [0, 255]
                if max(imgRGB(:)) <= 1.0
                    imgProcessed = uint8(round(imgRGB * 255));
                else
                    imgProcessed = uint8(max(0, min(255, round(imgRGB))));
                end
            else
                imgProcessed = uint8(imgRGB);
            end
        catch ME
            error('retinai_preprocess:ConversionError', ...
                'Failed to convert image to uint8: %s', ME.message);
        end
    end

    % ---------------------------------------------------------------------
    % 9. Final Assertion of Preprocessed Output
    % ---------------------------------------------------------------------
    if ~isequal(size(imgProcessed), [224 224 3]) || ~isa(imgProcessed, 'uint8')
        error('retinai_preprocess:OutputError', ...
            'Preprocessed image output does not conform to [224 224 3] uint8.');
    end
end
