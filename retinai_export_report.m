function reportPath = retinai_export_report(result, imgInfo, imgPath, savePath)
% RETINAI_EXPORT_REPORT Export Retina-AI analysis report as a formatted text file.
%
% Syntax:
%   reportPath = retinai_export_report(result, imgInfo, imgPath)
%   reportPath = retinai_export_report(result, imgInfo, imgPath, savePath)
%
% Description:
%   retinai_export_report generates a comprehensive, professional clinical
%   screening report for diabetic retinopathy (DR) analysis performed by
%   Retina-AI. The report contains image metadata, AI predictions, class
%   probability distributions, screening referral decisions, model
%   architecture specifications, and academic/clinical disclaimers.
%
% Inputs:
%   result   - Struct containing inference outputs (from retinai_infer):
%                .scores            - Numeric array of 5 class probabilities/scores
%                .predClass         - Predicted class name (char or string)
%                .confidence        - Prediction confidence score (numeric)
%                .pRef              - Probability of referable DR (numeric)
%                .isReferable       - Logical or numeric referable decision flag
%                .screeningCategory - Screening category string ('REFERABLE' or 'NON-REFERABLE')
%                .classNames        - Cell array of 5 class names (optional)
%   imgInfo  - Struct containing image metadata (from retinai_preprocess):
%                .filename          - Image file name (char or string)
%                .width             - Image width in pixels (numeric)
%                .height            - Image height in pixels (numeric)
%                .channels          - Number of color channels (numeric)
%                .format            - Image format (char or string, e.g., 'JPEG')
%                .filesize          - File size in bytes (numeric)
%   imgPath  - Char array or string representing original image path
%   savePath - (Optional) Destination text file path. If empty or not provided,
%              the user is prompted via uiputfile to select a destination.
%
% Outputs:
%   reportPath - Full path where the report was saved. Returns empty string
%                ('') if the user cancelled the dialog or if an error occurred.
%
% Example:
%   res.scores = [0.03, 0.82, 0.01, 0.10, 0.04];
%   res.predClass = 'Moderate';
%   res.confidence = 82.0;
%   res.pRef = 0.96;
%   res.isReferable = true;
%   res.screeningCategory = 'REFERABLE';
%   res.classNames = {'Mild', 'Moderate', 'No_DR', 'Proliferate_DR', 'Severe'};
%   info.filename = 'patient_01_left.jpeg';
%   info.width = 2240; info.height = 1488; info.channels = 3;
%   info.format = 'JPEG'; info.filesize = 152048;
%   reportPath = retinai_export_report(res, info, 'patient_01_left.jpeg');

% Default return value
reportPath = '';
fid = -1;

try
    % -------------------------------------------------------------------------
    % 1. Input Validation & Save Path Resolution
    % -------------------------------------------------------------------------
    if nargin < 3
        error('retinai_export_report:InvalidInput', ...
            'retinai_export_report requires at least 3 arguments: (result, imgInfo, imgPath).');
    end

    % Normalize imgPath
    if isstring(imgPath)
        imgPath = char(imgPath);
    end

    % Prompt user if savePath is omitted or empty
    if nargin < 4 || isempty(savePath)
        defaultFilename = sprintf('RetinaAI_Report_%s.txt', datestr(now, 'yyyymmdd_HHMMSS'));
        [selectedFile, selectedDir] = uiputfile({'*.txt', 'Text Report'}, ...
            'Save Analysis Report', defaultFilename);

        % User cancelled the file selection dialog
        if isequal(selectedFile, 0) || isequal(selectedDir, 0)
            reportPath = '';
            return;
        end

        savePath = fullfile(selectedDir, selectedFile);
    else
        if isstring(savePath)
            savePath = char(savePath);
        end

        % If savePath is an existing directory, generate default filename inside it
        if exist(savePath, 'dir') == 7
            defaultFilename = sprintf('RetinaAI_Report_%s.txt', datestr(now, 'yyyymmdd_HHMMSS'));
            savePath = fullfile(savePath, defaultFilename);
        end
    end

    % Ensure destination directory exists
    destinationDir = fileparts(savePath);
    if ~isempty(destinationDir) && ~exist(destinationDir, 'dir')
        mkdir(destinationDir);
    end

    % -------------------------------------------------------------------------
    % 2. Extract & Format Metadata
    % -------------------------------------------------------------------------
    % Timestamp and random 8-character hex Report ID
    currentTimestamp = datestr(now, 'yyyy-mm-dd HH:MM:SS');
    reportId = sprintf('%04X%04X', randi([0, 65535]), randi([0, 65535]));

    % Image Information (imgInfo)
    if isfield(imgInfo, 'filename') && ~isempty(imgInfo.filename)
        filenameStr = char(imgInfo.filename);
    else
        [~, fname, fext] = fileparts(imgPath);
        filenameStr = [fname, fext];
    end

    imgWidth = 0;
    imgHeight = 0;
    imgChannels = 0;
    if isfield(imgInfo, 'width') && ~isempty(imgInfo.width)
        imgWidth = round(double(imgInfo.width));
    end
    if isfield(imgInfo, 'height') && ~isempty(imgInfo.height)
        imgHeight = round(double(imgInfo.height));
    end
    if isfield(imgInfo, 'channels') && ~isempty(imgInfo.channels)
        imgChannels = round(double(imgInfo.channels));
    end

    % If dimensions are zero, attempt reading from image file if accessible
    if (imgWidth == 0 || imgHeight == 0) && exist(imgPath, 'file') == 2
        try
            fInfo = imfinfo(imgPath);
            imgWidth = fInfo.Width;
            imgHeight = fInfo.Height;
            if isfield(fInfo, 'NumberOfSamples')
                imgChannels = fInfo.NumberOfSamples;
            elseif isfield(fInfo, 'ColorType') && strcmpi(fInfo.ColorType, 'truecolor')
                imgChannels = 3;
            else
                imgChannels = 1;
            end
        catch
            % Silently keep defaults if imfinfo fails
        end
    end

    % Image Format
    if isfield(imgInfo, 'format') && ~isempty(imgInfo.format)
        formatStr = upper(char(imgInfo.format));
    else
        [~, ~, fext] = fileparts(imgPath);
        formatStr = upper(regexprep(fext, '^\.', ''));
    end
    if isempty(formatStr)
        formatStr = 'UNKNOWN';
    end

    % File Size
    imgFilesize = 0;
    if isfield(imgInfo, 'filesize') && ~isempty(imgInfo.filesize)
        imgFilesize = round(double(imgInfo.filesize));
    elseif exist(imgPath, 'file') == 2
        dirEntry = dir(imgPath);
        if ~isempty(dirEntry)
            imgFilesize = dirEntry.bytes;
        end
    end

    % -------------------------------------------------------------------------
    % 3. Extract & Format AI Predictions & Probability Distribution
    % -------------------------------------------------------------------------
    % Predicted Stage
    if isfield(result, 'predClass') && ~isempty(result.predClass)
        predClassStr = char(result.predClass);
    else
        predClassStr = 'Unknown';
    end

    % Class Names & Probability Scores
    targetClasses = {'Mild', 'Moderate', 'No_DR', 'Proliferate_DR', 'Severe'};
    scoreMap = containers.Map();
    for k = 1:numel(targetClasses)
        scoreMap(targetClasses{k}) = 0.0;
    end

    scoresRaw = [];
    if isfield(result, 'scores') && ~isempty(result.scores)
        scoresRaw = double(result.scores(:)');
    end

    % Determine scale: if probabilities sum to <= 1.5, scale to percentage [0, 100]
    isProbScale = false;
    if ~isempty(scoresRaw)
        if sum(scoresRaw) <= 1.5 && max(scoresRaw) <= 1.0001
            isProbScale = true;
            scaledScores = scoresRaw * 100;
        else
            scaledScores = scoresRaw;
        end

        % Map scores to specific target classes
        if isfield(result, 'classNames') && ~isempty(result.classNames)
            cNames = cellstr(result.classNames);
            for k = 1:min(numel(cNames), numel(scaledScores))
                currName = cNames{k};
                matched = false;
                for tc = targetClasses
                    if strcmpi(currName, tc{1})
                        scoreMap(tc{1}) = scaledScores(k);
                        matched = true;
                        break;
                    end
                end
                if ~matched
                    scoreMap(currName) = scaledScores(k);
                end
            end
        else
            for k = 1:min(numel(targetClasses), numel(scaledScores))
                scoreMap(targetClasses{k}) = scaledScores(k);
            end
        end
    end

    % Confidence
    confVal = 0.0;
    if isfield(result, 'confidence') && ~isempty(result.confidence)
        confVal = double(result.confidence);
        if isProbScale && confVal <= 1.0001
            confVal = confVal * 100;
        elseif confVal <= 1.0001 && isempty(scoresRaw)
            confVal = confVal * 100;
        end
    elseif ~isempty(scoresRaw)
        confVal = max(scaledScores);
    end

    % -------------------------------------------------------------------------
    % 4. Extract Clinical Screening Decision
    % -------------------------------------------------------------------------
    % Screening Category
    if isfield(result, 'screeningCategory') && ~isempty(result.screeningCategory)
        screeningCategoryStr = upper(char(result.screeningCategory));
    elseif isfield(result, 'isReferable') && ~isempty(result.isReferable)
        if result.isReferable
            screeningCategoryStr = 'REFERABLE';
        else
            screeningCategoryStr = 'NON-REFERABLE';
        end
    else
        screeningCategoryStr = 'NON-REFERABLE';
    end

    % Referable Probability
    pRefVal = 0.0;
    if isfield(result, 'pRef') && ~isempty(result.pRef)
        pRefVal = double(result.pRef);
        if isProbScale && pRefVal <= 1.0001
            pRefVal = pRefVal * 100;
        elseif pRefVal <= 1.0001 && isempty(scoresRaw)
            pRefVal = pRefVal * 100;
        end
    elseif scoreMap.isKey('Moderate') && scoreMap.isKey('Proliferate_DR') && scoreMap.isKey('Severe')
        % Fallback sum: Moderate + Proliferate_DR + Severe
        pRefVal = scoreMap('Moderate') + scoreMap('Proliferate_DR') + scoreMap('Severe');
    end

    % -------------------------------------------------------------------------
    % 5. Write Report to Destination File
    % -------------------------------------------------------------------------
    % Open file with UTF-8 encoding
    fid = fopen(savePath, 'w', 'n', 'UTF-8');
    if fid == -1
        % Fallback open mode if native UTF-8 encoding parameter is unavailable
        fid = fopen(savePath, 'w');
        if fid == -1
            error('retinai_export_report:FileOpenError', ...
                'Failed to open file for writing: %s', savePath);
        end
    end

    % Formatting constants
    EQUAL_DIVIDER = repmat('=', 1, 80);
    LIGHT_DIVIDER = repmat(char(9472), 1, 80); % Unicode U+2500 '─'
    MULT_SIGN     = char(215);                 % Unicode U+00D7 '×'

    % Header
    fprintf(fid, '%s\n', EQUAL_DIVIDER);
    fprintf(fid, '                    RETINA-AI ANALYSIS REPORT\n');
    fprintf(fid, '          AI-Powered Diabetic Retinopathy Screening System\n');
    fprintf(fid, '%s\n\n', EQUAL_DIVIDER);

    fprintf(fid, 'Generated: %s\n', currentTimestamp);
    fprintf(fid, 'Report ID: %s\n\n', reportId);

    % Section 1: Image Information
    fprintf(fid, '%s\n', LIGHT_DIVIDER);
    fprintf(fid, '1. IMAGE INFORMATION\n');
    fprintf(fid, '%s\n', LIGHT_DIVIDER);
    fprintf(fid, '%-17s%s\n', 'Filename:', filenameStr);
    fprintf(fid, '%-17s%d %s %d %s %d\n', 'Original Size:', imgWidth, MULT_SIGN, imgHeight, MULT_SIGN, imgChannels);
    fprintf(fid, '%-17s%s\n', 'Format:', formatStr);
    fprintf(fid, '%-17s%d bytes\n', 'File Size:', imgFilesize);
    fprintf(fid, '%-17s%s\n\n', 'Source Path:', imgPath);

    % Section 2: AI Prediction
    fprintf(fid, '%s\n', LIGHT_DIVIDER);
    fprintf(fid, '2. AI PREDICTION\n');
    fprintf(fid, '%s\n', LIGHT_DIVIDER);
    fprintf(fid, '%-21s%s\n', 'Predicted Stage:', predClassStr);
    fprintf(fid, '%-21s%.2f%%\n\n', 'Confidence:', confVal);

    % Section 3: Probability Distribution
    fprintf(fid, '%s\n', LIGHT_DIVIDER);
    fprintf(fid, '3. PROBABILITY DISTRIBUTION\n');
    fprintf(fid, '%s\n', LIGHT_DIVIDER);
    fprintf(fid, '%-21s%.2f%%\n', 'Mild:', scoreMap('Mild'));
    fprintf(fid, '%-21s%.2f%%\n', 'Moderate:', scoreMap('Moderate'));
    fprintf(fid, '%-21s%.2f%%\n', 'No_DR:', scoreMap('No_DR'));
    fprintf(fid, '%-21s%.2f%%\n', 'Proliferate_DR:', scoreMap('Proliferate_DR'));
    fprintf(fid, '%-21s%.2f%%\n\n', 'Severe:', scoreMap('Severe'));

    % Section 4: Clinical Screening Decision
    fprintf(fid, '%s\n', LIGHT_DIVIDER);
    fprintf(fid, '4. CLINICAL SCREENING DECISION\n');
    fprintf(fid, '%s\n', LIGHT_DIVIDER);
    fprintf(fid, '%-21s%s\n', 'Screening Category:', screeningCategoryStr);
    fprintf(fid, '%-21s%.2f%%\n\n', 'Referable Prob:', pRefVal);
    fprintf(fid, 'Referral Grouping:\n');
    fprintf(fid, '  NON-REFERABLE: Mild, No_DR\n');
    fprintf(fid, '  REFERABLE:     Moderate, Proliferate_DR, Severe\n\n');

    % Section 5: Model Information
    fprintf(fid, '%s\n', LIGHT_DIVIDER);
    fprintf(fid, '5. MODEL INFORMATION\n');
    fprintf(fid, '%s\n', LIGHT_DIVIDER);
    fprintf(fid, '%-17sAG_V4_5_ReducedAug (ResNet-101)\n', 'Model:');
    fprintf(fid, '%-17sMATLAB Deep Learning Toolbox\n', 'Framework:');
    fprintf(fid, '%-17s224 %s 224 %s 3 RGB\n', 'Input Size:', MULT_SIGN, MULT_SIGN);
    fprintf(fid, '%-17s5 (Mild, Moderate, No_DR, Proliferate_DR, Severe)\n', 'Classes:');
    fprintf(fid, '%-17s~42.56M\n\n', 'Parameters:');

    % Section 6: Disclaimer
    fprintf(fid, '%s\n', LIGHT_DIVIDER);
    fprintf(fid, '6. DISCLAIMER\n');
    fprintf(fid, '%s\n', LIGHT_DIVIDER);
    fprintf(fid, 'This is a research/experimental diabetic-retinopathy screening model developed\n');
    fprintf(fid, 'for academic evaluation. It is NOT approved as a medical device, is NOT\n');
    fprintf(fid, 'clinically validated, and is NOT intended for diagnostic use. Always consult\n');
    fprintf(fid, 'a qualified ophthalmologist for clinical decisions.\n');
    fprintf(fid, '%s\n', EQUAL_DIVIDER);

    % Close file
    fclose(fid);
    fid = -1;

    % Return successfully saved report path
    reportPath = savePath;

catch ME
    % Clean up file handle if open
    if fid ~= -1
        try
            fclose(fid);
        catch
        end
    end
    warning('retinai_export_report:ExportFailed', ...
        'Failed to export analysis report: %s', ME.message);
    reportPath = '';
end
end
