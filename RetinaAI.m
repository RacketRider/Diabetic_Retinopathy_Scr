classdef RetinaAI < matlab.apps.AppBase
    % RETINA-AI: AI-Powered Diabetic Retinopathy Screening & Analysis System
    %
    % Professional MATLAB App Designer application wrapping the
    % AG_V4_5_ReducedAug trained ResNet-101 model for 5-class diabetic
    % retinopathy classification and referable/non-referable screening.
    %
    % Usage:
    %   app = RetinaAI;
    %
    % Requirements:
    %   - MATLAB R2026a
    %   - Deep Learning Toolbox
    %   - Image Processing Toolbox
    %   - Checkpoint: ANTIGRAVITY/checkpoints/AG_V4_5_ReducedAug.mat

    % =====================================================================
    % PUBLIC PROPERTIES — UI Components
    % =====================================================================
    properties (Access = public)
        UIFigure               matlab.ui.Figure

        % --- Main Layout ---
        MainGrid               matlab.ui.container.GridLayout
        SidebarPanel           matlab.ui.container.Panel
        ContentPanel           matlab.ui.container.Panel

        % --- Sidebar ---
        SidebarGrid            matlab.ui.container.GridLayout
        LogoLabel              matlab.ui.control.Label
        SubtitleLabel          matlab.ui.control.Label
        NavDashboard           matlab.ui.control.Button
        NavAnalyze             matlab.ui.control.Button
        NavInsights            matlab.ui.control.Button
        NavAbout               matlab.ui.control.Button
        ModelStatusLabel       matlab.ui.control.Label
        ModelStatusDot         matlab.ui.control.Label
        VersionLabel           matlab.ui.control.Label

        % --- Dashboard Page ---
        DashboardGrid          matlab.ui.container.GridLayout
        DashHeaderLabel        matlab.ui.control.Label
        DashSubLabel           matlab.ui.control.Label
        DashTagsLabel          matlab.ui.control.Label
        CardModelStatus        matlab.ui.container.Panel
        CardModelName          matlab.ui.container.Panel
        CardInputSpec          matlab.ui.container.Panel
        CardClasses            matlab.ui.container.Panel
        CardScreening          matlab.ui.container.Panel
        CardModelStatusTitle   matlab.ui.control.Label
        CardModelStatusValue   matlab.ui.control.Label
        CardModelNameTitle     matlab.ui.control.Label
        CardModelNameValue     matlab.ui.control.Label
        CardInputSpecTitle     matlab.ui.control.Label
        CardInputSpecValue     matlab.ui.control.Label
        CardClassesTitle       matlab.ui.control.Label
        CardClassesValue       matlab.ui.control.Label
        CardScreeningTitle     matlab.ui.control.Label
        CardScreeningValue     matlab.ui.control.Label
        DashCTAButton          matlab.ui.control.Button
        DashPipelineLabel      matlab.ui.control.Label
        DashPipelineText       matlab.ui.control.Label

        % --- Analyze Page ---
        AnalyzeGrid            matlab.ui.container.GridLayout

        % Left: Input Image
        InputPanel             matlab.ui.container.Panel
        InputGrid              matlab.ui.container.GridLayout
        InputTitleLabel        matlab.ui.control.Label
        InputAxes              matlab.ui.control.UIAxes
        InputEmptyLabel        matlab.ui.control.Label
        ChooseImageBtn         matlab.ui.control.Button
        InputInfoLabel         matlab.ui.control.Label

        % Center: Model View
        ModelViewPanel         matlab.ui.container.Panel
        ModelViewGrid          matlab.ui.container.GridLayout
        ModelViewTitle         matlab.ui.control.Label
        ModelInputAxes         matlab.ui.control.UIAxes
        PipelineStep1          matlab.ui.control.Label
        PipelineStep2          matlab.ui.control.Label
        PipelineStep3          matlab.ui.control.Label
        PipelineStep4          matlab.ui.control.Label
        AnalyzeBtn             matlab.ui.control.Button
        ResetBtn               matlab.ui.control.Button

        % Right: Results
        ResultPanel            matlab.ui.container.Panel
        ResultGrid             matlab.ui.container.GridLayout
        ResultTitleLabel       matlab.ui.control.Label
        PredStageTitle         matlab.ui.control.Label
        PredStageValue         matlab.ui.control.Label
        ConfidenceTitle        matlab.ui.control.Label
        ConfidenceValue        matlab.ui.control.Label
        ScreeningTitle         matlab.ui.control.Label
        ScreeningValue         matlab.ui.control.Label
        ScreeningExplain       matlab.ui.control.Label
        ProbTitle              matlab.ui.control.Label
        ProbAxes               matlab.ui.control.UIAxes
        ExportBtn              matlab.ui.control.Button

        % --- History Panel (within Analyze) ---
        HistoryPanel           matlab.ui.container.Panel
        HistoryGrid            matlab.ui.container.GridLayout
        HistoryTitle           matlab.ui.control.Label
        HistoryListBox         matlab.ui.control.ListBox

        % --- Model Insights Page ---
        InsightsGrid           matlab.ui.container.GridLayout
        InsightsTitle          matlab.ui.control.Label
        InsightsSubTitle       matlab.ui.control.Label

        % Metric Cards
        MetricROCAUC           matlab.ui.container.Panel
        MetricROCAUCTitle      matlab.ui.control.Label
        MetricROCAUCValue      matlab.ui.control.Label
        MetricPRAUC            matlab.ui.container.Panel
        MetricPRAUCTitle       matlab.ui.control.Label
        MetricPRAUCValue       matlab.ui.control.Label
        MetricSens             matlab.ui.container.Panel
        MetricSensTitle        matlab.ui.control.Label
        MetricSensValue        matlab.ui.control.Label
        MetricSpec             matlab.ui.container.Panel
        MetricSpecTitle        matlab.ui.control.Label
        MetricSpecValue        matlab.ui.control.Label
        MetricAcc5             matlab.ui.container.Panel
        MetricAcc5Title        matlab.ui.control.Label
        MetricAcc5Value        matlab.ui.control.Label
        MetricClinAcc          matlab.ui.container.Panel
        MetricClinAccTitle     matlab.ui.control.Label
        MetricClinAccValue     matlab.ui.control.Label
        MetricPPV              matlab.ui.container.Panel
        MetricPPVTitle         matlab.ui.control.Label
        MetricPPVValue         matlab.ui.control.Label
        MetricNPV              matlab.ui.container.Panel
        MetricNPVTitle         matlab.ui.control.Label
        MetricNPVValue         matlab.ui.control.Label

        % Confusion matrix
        ClinCMPanel            matlab.ui.container.Panel
        ClinCMTitle            matlab.ui.control.Label
        ClinCMText             matlab.ui.control.TextArea

        % Model Info
        ModelInfoPanel         matlab.ui.container.Panel
        ModelInfoTitle         matlab.ui.control.Label
        ModelInfoText          matlab.ui.control.Label

        % Figures
        ROCPRPanel             matlab.ui.container.Panel
        ROCPRTitle             matlab.ui.control.Label
        ROCPRImage             matlab.ui.control.Image
        ModNoDRPanel           matlab.ui.container.Panel
        ModNoDRTitle           matlab.ui.control.Label
        ModNoDRImage           matlab.ui.control.Image

        % --- About Page ---
        AboutGrid              matlab.ui.container.GridLayout
        AboutTitle             matlab.ui.control.Label
        AboutText              matlab.ui.control.TextArea
    end

    % =====================================================================
    % PRIVATE PROPERTIES — Application State
    % =====================================================================
    properties (Access = private)
        NetTrained                       % Trained dlnetwork
        ClassNames       cell            % Cell array of class names
        ModelLoaded      logical = false % Whether model is loaded
        CurrentImgPath   string = ""     % Path to current image
        CurrentImgOrig                   % Original loaded image
        CurrentImgProc                   % Preprocessed image
        CurrentImgInfo   struct          % Image metadata
        CurrentResult    struct          % Inference result
        HasResult        logical = false % Whether result exists
        AppRootPath      string          % Application root directory
        CheckpointPath   string          % Path to model checkpoint
        TestResultsPath  string          % Path to test results
        FiguresPath      string          % Path to figures directory

        % Analysis history (session only)
        HistoryEntries   struct          % Array of history entries
        HistoryCount     double = 0      % Number of history entries

        % Active page tracking
        ActivePage       string = "dashboard"

        % Colors
        ColorPrimary     = [0.15 0.45 0.75]    % Deep blue
        ColorAccent      = [0.20 0.60 0.86]    % Lighter blue
        ColorSuccess     = [0.18 0.69 0.49]    % Green
        ColorDanger      = [0.85 0.25 0.25]    % Red
        ColorWarning     = [0.93 0.69 0.13]    % Amber
        ColorBgDark      = [0.11 0.13 0.17]    % Dark sidebar
        ColorBgLight     = [0.95 0.96 0.97]    % Light content bg
        ColorCardBg      = [1.00 1.00 1.00]    % White cards
        ColorTextPrimary = [0.12 0.14 0.18]    % Near-black text
        ColorTextSecondary = [0.45 0.50 0.55]  % Gray text
        ColorBorder      = [0.88 0.90 0.92]    % Subtle border
    end

    % =====================================================================
    % PRIVATE METHODS
    % =====================================================================
    methods (Access = private)

        % -----------------------------------------------------------------
        % MODEL LOADING
        % -----------------------------------------------------------------
        function loadModel(app)
            % Load the trained model checkpoint
            app.updateModelStatus('Loading...', [0.93 0.69 0.13]);

            % Determine checkpoint path
            possiblePaths = {
                fullfile(app.AppRootPath, 'ANTIGRAVITY', 'checkpoints', 'AG_V4_5_ReducedAug.mat')
                fullfile(app.AppRootPath, '..', 'ANTIGRAVITY', 'checkpoints', 'AG_V4_5_ReducedAug.mat')
                fullfile(pwd, 'ANTIGRAVITY', 'checkpoints', 'AG_V4_5_ReducedAug.mat')
                fullfile(pwd, 'checkpoints', 'AG_V4_5_ReducedAug.mat')
            };

            checkpointFound = false;
            for i = 1:length(possiblePaths)
                if isfile(possiblePaths{i})
                    app.CheckpointPath = possiblePaths{i};
                    checkpointFound = true;
                    break;
                end
            end

            if ~checkpointFound
                % Ask user to locate the checkpoint
                [file, path] = uigetfile('*.mat', ...
                    'Locate AG_V4_5_ReducedAug.mat Checkpoint', ...
                    app.AppRootPath);
                if isequal(file, 0)
                    app.updateModelStatus('Not Loaded', app.ColorDanger);
                    app.ModelLoaded = false;
                    return;
                end
                app.CheckpointPath = fullfile(path, file);
            end

            try
                data = load(app.CheckpointPath, 'netTrained');
                app.NetTrained = data.netTrained;
                app.ClassNames = {'Mild', 'Moderate', 'No_DR', 'Proliferate_DR', 'Severe'};
                app.ModelLoaded = true;
                app.updateModelStatus('Loaded', app.ColorSuccess);
                app.updateDashboardCards();
            catch ME
                app.ModelLoaded = false;
                app.updateModelStatus('Error', app.ColorDanger);
                uialert(app.UIFigure, ...
                    sprintf('Failed to load model:\n%s', ME.message), ...
                    'Model Load Error', 'Icon', 'error');
            end
        end

        function updateModelStatus(app, statusText, color)
            app.ModelStatusDot.Text = char(9679); % ●
            app.ModelStatusDot.FontColor = color;
            app.ModelStatusLabel.Text = sprintf('Model: %s', statusText);
            app.ModelStatusLabel.FontColor = color;

            % Update dashboard card
            if isvalid(app.CardModelStatusValue)
                app.CardModelStatusValue.Text = statusText;
                app.CardModelStatusValue.FontColor = color;
            end
        end

        % -----------------------------------------------------------------
        % IMAGE SELECTION
        % -----------------------------------------------------------------
        function selectImage(app)
            % Open file dialog for image selection
            startPath = app.AppRootPath;
            archivePath = fullfile(app.AppRootPath, 'archive (1)', 'colored_images', 'colored_images');
            if isfolder(archivePath)
                startPath = archivePath;
            end

            [file, path] = uigetfile( ...
                {'*.png;*.jpg;*.jpeg;*.bmp;*.tif;*.tiff', 'Image Files'}, ...
                'Select Retinal Fundus Image', startPath);

            if isequal(file, 0)
                return; % User cancelled
            end

            imgPath = fullfile(path, file);
            app.loadAndDisplayImage(imgPath);
        end

        function loadAndDisplayImage(app, imgPath)
            try
                [imgProc, imgInfo] = retinai_preprocess(imgPath);

                app.CurrentImgPath = imgPath;
                app.CurrentImgOrig = imread(imgPath);
                app.CurrentImgProc = imgProc;
                app.CurrentImgInfo = imgInfo;
                app.HasResult = false;

                % Display original image
                imshow(app.CurrentImgOrig, 'Parent', app.InputAxes);
                app.InputAxes.Visible = 'on';
                app.InputEmptyLabel.Visible = 'off';

                % Display preprocessed image
                imshow(app.CurrentImgProc, 'Parent', app.ModelInputAxes);
                app.ModelInputAxes.Visible = 'on';

                % Update image info
                app.InputInfoLabel.Text = sprintf( ...
                    '%s  |  %d × %d  |  %s  |  %.1f KB', ...
                    imgInfo.filename, imgInfo.width, imgInfo.height, ...
                    upper(imgInfo.format), imgInfo.filesize / 1024);
                app.InputInfoLabel.Visible = 'on';

                % Update pipeline
                app.updatePipeline('imageLoaded');

                % Enable analyze button
                if app.ModelLoaded
                    app.AnalyzeBtn.Enable = 'on';
                    app.AnalyzeBtn.BackgroundColor = app.ColorPrimary;
                end

                % Reset result panel
                app.resetResultPanel();

            catch ME
                uialert(app.UIFigure, ...
                    sprintf('Unable to load image:\n%s', ME.message), ...
                    'Image Error', 'Icon', 'error');
            end
        end

        % -----------------------------------------------------------------
        % INFERENCE
        % -----------------------------------------------------------------
        function runInference(app)
            if ~app.ModelLoaded
                uialert(app.UIFigure, ...
                    'Model is not loaded. Please load the model first.', ...
                    'Model Required', 'Icon', 'warning');
                return;
            end

            if isempty(app.CurrentImgProc)
                uialert(app.UIFigure, ...
                    'Please select an image first.', ...
                    'No Image', 'Icon', 'warning');
                return;
            end

            % Disable controls during inference
            app.AnalyzeBtn.Enable = 'off';
            app.AnalyzeBtn.Text = 'Analyzing...';
            app.ChooseImageBtn.Enable = 'off';
            app.ResetBtn.Enable = 'off';
            app.updatePipeline('inferring');
            drawnow;

            try
                result = retinai_infer(app.NetTrained, app.CurrentImgProc, app.ClassNames);
                app.CurrentResult = result;
                app.HasResult = true;

                % Update UI with results
                app.updatePredictionUI(result);
                app.updateProbabilityChart(result);
                app.updateScreeningDecision(result);
                app.updatePipeline('complete');

                % Add to history
                app.addHistoryEntry(result);

                % Enable export
                app.ExportBtn.Enable = 'on';
                app.ExportBtn.BackgroundColor = app.ColorPrimary;

            catch ME
                app.updatePipeline('error');
                uialert(app.UIFigure, ...
                    sprintf('Inference failed:\n%s', ME.message), ...
                    'Analysis Error', 'Icon', 'error');
            end

            % Re-enable controls
            app.AnalyzeBtn.Enable = 'on';
            app.AnalyzeBtn.Text = sprintf('%s Analyze', char(9654));
            app.ChooseImageBtn.Enable = 'on';
            app.ResetBtn.Enable = 'on';
        end

        % -----------------------------------------------------------------
        % UI UPDATE METHODS
        % -----------------------------------------------------------------
        function updatePredictionUI(app, result)
            app.ResultTitleLabel.Text = 'AI ANALYSIS RESULT';
            app.ResultTitleLabel.FontColor = app.ColorTextPrimary;

            app.PredStageTitle.Text = 'PREDICTED STAGE';
            app.PredStageTitle.Visible = 'on';
            app.PredStageValue.Text = strrep(result.predClass, '_', ' ');
            app.PredStageValue.Visible = 'on';

            % Color code the prediction
            if result.isReferable
                app.PredStageValue.FontColor = app.ColorDanger;
            else
                app.PredStageValue.FontColor = app.ColorSuccess;
            end

            app.ConfidenceTitle.Text = 'CONFIDENCE';
            app.ConfidenceTitle.Visible = 'on';
            app.ConfidenceValue.Text = sprintf('%.1f%%', result.confidence);
            app.ConfidenceValue.Visible = 'on';

            if result.confidence >= 80
                app.ConfidenceValue.FontColor = app.ColorSuccess;
            elseif result.confidence >= 50
                app.ConfidenceValue.FontColor = app.ColorWarning;
            else
                app.ConfidenceValue.FontColor = app.ColorDanger;
            end
        end

        function updateProbabilityChart(app, result)
            app.ProbTitle.Visible = 'on';
            app.ProbAxes.Visible = 'on';

            cla(app.ProbAxes);

            scores = result.scores * 100;
            names = result.classNames;

            % Display names without underscores
            displayNames = strrep(names, '_', ' ');

            % Sort by score descending for display
            [sortedScores, sortIdx] = sort(scores, 'descend');
            sortedNames = displayNames(sortIdx);

            % Create horizontal bar chart
            barh(app.ProbAxes, categorical(sortedNames, sortedNames), sortedScores, ...
                'FaceColor', 'flat', 'EdgeColor', 'none', 'BarWidth', 0.6);

            % Color the bars
            bars = findobj(app.ProbAxes, 'Type', 'Bar');
            if ~isempty(bars)
                cdata = repmat(app.ColorAccent, length(sortedScores), 1);
                % Highlight predicted class
                predDisplayName = strrep(result.predClass, '_', ' ');
                for k = 1:length(sortedNames)
                    if strcmp(sortedNames{k}, predDisplayName)
                        cdata(k, :) = app.ColorPrimary;
                    end
                end
                bars.CData = cdata;
            end

            app.ProbAxes.XLim = [0 100];
            app.ProbAxes.FontSize = 10;
            app.ProbAxes.FontName = 'Segoe UI';
            app.ProbAxes.XLabel.String = 'Probability (%)';
            app.ProbAxes.Color = app.ColorCardBg;
            app.ProbAxes.XColor = app.ColorTextSecondary;
            app.ProbAxes.YColor = app.ColorTextPrimary;
            app.ProbAxes.Box = 'off';
            app.ProbAxes.XGrid = 'on';
            app.ProbAxes.GridAlpha = 0.15;

            % Add percentage labels on bars
            hold(app.ProbAxes, 'on');
            for k = 1:length(sortedScores)
                text(app.ProbAxes, sortedScores(k) + 1.5, k, ...
                    sprintf('%.1f%%', sortedScores(k)), ...
                    'FontSize', 9, 'FontName', 'Segoe UI', ...
                    'Color', app.ColorTextSecondary, ...
                    'VerticalAlignment', 'middle');
            end
            hold(app.ProbAxes, 'off');
        end

        function updateScreeningDecision(app, result)
            app.ScreeningTitle.Visible = 'on';
            app.ScreeningValue.Visible = 'on';
            app.ScreeningExplain.Visible = 'on';

            if result.isReferable
                app.ScreeningValue.Text = sprintf('%s REFERABLE DR', char(9888));
                app.ScreeningValue.FontColor = app.ColorDanger;
                app.ScreeningExplain.Text = sprintf( ...
                    '%s belongs to the REFERABLE screening group.\nRecommend ophthalmological referral.', ...
                    strrep(result.predClass, '_', ' '));
            else
                app.ScreeningValue.Text = sprintf('%s NON-REFERABLE', char(10003));
                app.ScreeningValue.FontColor = app.ColorSuccess;
                app.ScreeningExplain.Text = sprintf( ...
                    '%s belongs to the NON-REFERABLE screening group.\nRoutine follow-up recommended.', ...
                    strrep(result.predClass, '_', ' '));
            end
        end

        function updatePipeline(app, stage)
            check = char(10003); % ✓
            dot   = char(9679);  % ●
            circle = char(9675); % ○

            switch stage
                case 'initial'
                    app.PipelineStep1.Text = sprintf('%s  Image loaded', circle);
                    app.PipelineStep2.Text = sprintf('%s  Preprocessing', circle);
                    app.PipelineStep3.Text = sprintf('%s  Model inference', circle);
                    app.PipelineStep4.Text = sprintf('%s  Screening decision', circle);
                    app.PipelineStep1.FontColor = app.ColorTextSecondary;
                    app.PipelineStep2.FontColor = app.ColorTextSecondary;
                    app.PipelineStep3.FontColor = app.ColorTextSecondary;
                    app.PipelineStep4.FontColor = app.ColorTextSecondary;

                case 'imageLoaded'
                    app.PipelineStep1.Text = sprintf('%s  Image loaded', check);
                    app.PipelineStep2.Text = sprintf('%s  Preprocessed', check);
                    app.PipelineStep3.Text = sprintf('%s  Model inference', circle);
                    app.PipelineStep4.Text = sprintf('%s  Screening decision', circle);
                    app.PipelineStep1.FontColor = app.ColorSuccess;
                    app.PipelineStep2.FontColor = app.ColorSuccess;
                    app.PipelineStep3.FontColor = app.ColorTextSecondary;
                    app.PipelineStep4.FontColor = app.ColorTextSecondary;

                case 'inferring'
                    app.PipelineStep1.Text = sprintf('%s  Image loaded', check);
                    app.PipelineStep2.Text = sprintf('%s  Preprocessed', check);
                    app.PipelineStep3.Text = sprintf('%s  Running model...', dot);
                    app.PipelineStep4.Text = sprintf('%s  Screening decision', circle);
                    app.PipelineStep1.FontColor = app.ColorSuccess;
                    app.PipelineStep2.FontColor = app.ColorSuccess;
                    app.PipelineStep3.FontColor = app.ColorWarning;
                    app.PipelineStep4.FontColor = app.ColorTextSecondary;

                case 'complete'
                    app.PipelineStep1.Text = sprintf('%s  Image loaded', check);
                    app.PipelineStep2.Text = sprintf('%s  Preprocessed', check);
                    app.PipelineStep3.Text = sprintf('%s  Model inference', check);
                    app.PipelineStep4.Text = sprintf('%s  Screening complete', check);
                    app.PipelineStep1.FontColor = app.ColorSuccess;
                    app.PipelineStep2.FontColor = app.ColorSuccess;
                    app.PipelineStep3.FontColor = app.ColorSuccess;
                    app.PipelineStep4.FontColor = app.ColorSuccess;

                case 'error'
                    app.PipelineStep3.Text = sprintf('%s  Inference error', dot);
                    app.PipelineStep3.FontColor = app.ColorDanger;
            end
        end

        function resetResultPanel(app)
            app.PredStageTitle.Visible = 'off';
            app.PredStageValue.Text = '';
            app.PredStageValue.Visible = 'off';
            app.ConfidenceTitle.Visible = 'off';
            app.ConfidenceValue.Text = '';
            app.ConfidenceValue.Visible = 'off';
            app.ScreeningTitle.Visible = 'off';
            app.ScreeningValue.Text = '';
            app.ScreeningValue.Visible = 'off';
            app.ScreeningExplain.Visible = 'off';
            app.ProbTitle.Visible = 'off';
            app.ProbAxes.Visible = 'off';
            cla(app.ProbAxes);
            app.ExportBtn.Enable = 'off';
            app.ExportBtn.BackgroundColor = [0.75 0.78 0.80];
            app.ResultTitleLabel.Text = 'AWAITING ANALYSIS';
            app.ResultTitleLabel.FontColor = app.ColorTextSecondary;
        end

        % -----------------------------------------------------------------
        % RESET
        % -----------------------------------------------------------------
        function resetAnalysis(app)
            % Clear image
            cla(app.InputAxes);
            app.InputAxes.Visible = 'off';
            app.InputEmptyLabel.Visible = 'on';
            app.InputInfoLabel.Visible = 'off';

            % Clear model input
            cla(app.ModelInputAxes);
            app.ModelInputAxes.Visible = 'off';

            % Clear results
            app.resetResultPanel();

            % Reset pipeline
            app.updatePipeline('initial');

            % Reset state
            app.CurrentImgPath = "";
            app.CurrentImgOrig = [];
            app.CurrentImgProc = [];
            app.CurrentImgInfo = struct();
            app.CurrentResult = struct();
            app.HasResult = false;

            % Disable analyze
            app.AnalyzeBtn.Enable = 'off';
            app.AnalyzeBtn.BackgroundColor = [0.75 0.78 0.80];
        end

        % -----------------------------------------------------------------
        % HISTORY
        % -----------------------------------------------------------------
        function addHistoryEntry(app, result)
            entry.filename = app.CurrentImgInfo.filename;
            entry.predClass = result.predClass;
            entry.confidence = result.confidence;
            entry.screening = result.screeningCategory;
            entry.timestamp = datestr(now, 'HH:MM:SS');
            entry.imgPath = app.CurrentImgPath;
            entry.result = result;
            entry.imgInfo = app.CurrentImgInfo;

            app.HistoryCount = app.HistoryCount + 1;
            app.HistoryEntries(app.HistoryCount) = entry;

            % Update listbox
            items = {};
            itemData = {};
            for k = app.HistoryCount:-1:1
                e = app.HistoryEntries(k);
                label = sprintf('[%s]  %s  →  %s (%.0f%%)', ...
                    e.timestamp, e.filename, strrep(e.predClass, '_', ' '), e.confidence);
                items{end+1} = label; %#ok<AGROW>
                itemData{end+1} = num2str(k); %#ok<AGROW>
            end
            app.HistoryListBox.Items = items;
            app.HistoryListBox.ItemsData = itemData;
        end

        function restoreHistoryEntry(app, idx)
            if idx < 1 || idx > app.HistoryCount
                return;
            end

            entry = app.HistoryEntries(idx);
            try
                app.loadAndDisplayImage(entry.imgPath);
                % Restore result directly
                app.CurrentResult = entry.result;
                app.HasResult = true;
                app.updatePredictionUI(entry.result);
                app.updateProbabilityChart(entry.result);
                app.updateScreeningDecision(entry.result);
                app.updatePipeline('complete');
                app.ExportBtn.Enable = 'on';
                app.ExportBtn.BackgroundColor = app.ColorPrimary;
            catch
                % If image no longer available, just show the result
            end
        end

        % -----------------------------------------------------------------
        % EXPORT
        % -----------------------------------------------------------------
        function exportReport(app)
            if ~app.HasResult
                uialert(app.UIFigure, ...
                    'No analysis result to export. Please analyze an image first.', ...
                    'Nothing to Export', 'Icon', 'info');
                return;
            end

            try
                reportPath = retinai_export_report( ...
                    app.CurrentResult, app.CurrentImgInfo, ...
                    app.CurrentImgPath, '');

                if ~isempty(reportPath) && strlength(reportPath) > 0
                    uialert(app.UIFigure, ...
                        sprintf('Report saved to:\n%s', reportPath), ...
                        'Export Successful', 'Icon', 'success');
                end
            catch ME
                uialert(app.UIFigure, ...
                    sprintf('Export failed:\n%s', ME.message), ...
                    'Export Error', 'Icon', 'error');
            end
        end

        % -----------------------------------------------------------------
        % METRICS LOADING
        % -----------------------------------------------------------------
        function loadModelMetrics(app)
            % Try to load test results from the repository
            possiblePaths = {
                fullfile(app.AppRootPath, 'ANTIGRAVITY', 'results', 'AG_FINAL_TEST_RESULTS.mat')
                fullfile(pwd, 'ANTIGRAVITY', 'results', 'AG_FINAL_TEST_RESULTS.mat')
            };

            metricsLoaded = false;
            for i = 1:length(possiblePaths)
                if isfile(possiblePaths{i})
                    try
                        data = load(possiblePaths{i});
                        app.populateMetrics(data);
                        metricsLoaded = true;
                        break;
                    catch
                        continue;
                    end
                end
            end

            if ~metricsLoaded
                % Use the verified benchmark values from README
                app.populateMetricsFromReadme();
            end
        end

        function populateMetrics(app, data)
            % Populate from loaded .mat file
            if isfield(data, 'roc_auc_test')
                app.MetricROCAUCValue.Text = sprintf('%.4f', data.roc_auc_test);
            end
            if isfield(data, 'pr_auc_test')
                app.MetricPRAUCValue.Text = sprintf('%.4f', data.pr_auc_test);
            end
            if isfield(data, 'sensTest')
                app.MetricSensValue.Text = sprintf('%.2f%%', data.sensTest * 100);
            end
            if isfield(data, 'specTest')
                app.MetricSpecValue.Text = sprintf('%.2f%%', data.specTest * 100);
            end
            if isfield(data, 'testAcc5Class')
                app.MetricAcc5Value.Text = sprintf('%.2f%%', data.testAcc5Class * 100);
            end
            if isfield(data, 'clinAccTest')
                app.MetricClinAccValue.Text = sprintf('%.2f%%', data.clinAccTest * 100);
            end
            if isfield(data, 'ppvTest')
                app.MetricPPVValue.Text = sprintf('%.2f%%', data.ppvTest * 100);
            end
            if isfield(data, 'npvTest')
                app.MetricNPVValue.Text = sprintf('%.2f%%', data.npvTest * 100);
            end

            % Clinical confusion matrix
            if isfield(data, 'TP') && isfield(data, 'FN') && isfield(data, 'FP') && isfield(data, 'TN')
                cmText = sprintf([ ...
                    'Clinical Confusion Matrix (Test N=5,268)\n\n' ...
                    '                          Pred Referable    Pred Non-Ref\n' ...
                    'True Referable:         %6d            %6d\n' ...
                    'True Non-Referable:     %6d            %6d'], ...
                    data.TP, data.FN, data.FP, data.TN);
                app.ClinCMText.Value = cmText;
            end
        end

        function populateMetricsFromReadme(app)
            % Fallback: use verified values from the project README
            app.MetricROCAUCValue.Text = '0.8564';
            app.MetricPRAUCValue.Text = '0.6914';
            app.MetricSensValue.Text = '69.84%';
            app.MetricSpecValue.Text = '85.18%';
            app.MetricAcc5Value.Text = '77.16%';
            app.MetricClinAccValue.Text = '82.18%';
            app.MetricPPVValue.Text = '53.41%';
            app.MetricNPVValue.Text = '92.07%';

            cmText = sprintf([ ...
                'Clinical Confusion Matrix (Test N=5,268)\n\n' ...
                '                          Pred Referable    Pred Non-Ref\n' ...
                'True Referable:         %6d            %6d\n' ...
                'True Non-Referable:     %6d            %6d'], ...
                720, 311, 628, 3609);
            app.ClinCMText.Value = cmText;
        end

        % -----------------------------------------------------------------
        % NAVIGATION
        % -----------------------------------------------------------------
        function showPage(app, pageName)
            app.ActivePage = pageName;

            % Hide all pages
            app.DashboardGrid.Visible = 'off';
            app.AnalyzeGrid.Visible = 'off';
            app.InsightsGrid.Visible = 'off';
            app.AboutGrid.Visible = 'off';

            % Reset nav button styles
            navButtons = [app.NavDashboard, app.NavAnalyze, app.NavInsights, app.NavAbout];
            for btn = navButtons
                btn.BackgroundColor = app.ColorBgDark;
                btn.FontColor = [0.65 0.70 0.75];
            end

            % Show selected page and highlight button
            switch pageName
                case 'dashboard'
                    app.DashboardGrid.Visible = 'on';
                    app.NavDashboard.BackgroundColor = [0.18 0.22 0.28];
                    app.NavDashboard.FontColor = [1 1 1];
                case 'analyze'
                    app.AnalyzeGrid.Visible = 'on';
                    app.NavAnalyze.BackgroundColor = [0.18 0.22 0.28];
                    app.NavAnalyze.FontColor = [1 1 1];
                case 'insights'
                    app.InsightsGrid.Visible = 'on';
                    app.NavInsights.BackgroundColor = [0.18 0.22 0.28];
                    app.NavInsights.FontColor = [1 1 1];
                case 'about'
                    app.AboutGrid.Visible = 'on';
                    app.NavAbout.BackgroundColor = [0.18 0.22 0.28];
                    app.NavAbout.FontColor = [1 1 1];
            end
        end

        % -----------------------------------------------------------------
        % DASHBOARD CARDS
        % -----------------------------------------------------------------
        function updateDashboardCards(app)
            if app.ModelLoaded
                app.CardModelStatusValue.Text = 'Loaded';
                app.CardModelStatusValue.FontColor = app.ColorSuccess;
                app.CardModelNameValue.Text = 'AG_V4_5_ReducedAug';
            else
                app.CardModelStatusValue.Text = 'Not Loaded';
                app.CardModelStatusValue.FontColor = app.ColorDanger;
                app.CardModelNameValue.Text = '—';
            end
        end

        % -----------------------------------------------------------------
        % HELPER: Create metric card
        % -----------------------------------------------------------------
        function [panel, titleLbl, valueLbl] = createMetricCard(app, parent, row, col, titleText, valueText)
            panel = uipanel(parent);
            panel.Layout.Row = row;
            panel.Layout.Column = col;
            panel.BackgroundColor = app.ColorCardBg;
            panel.BorderType = 'line';
            panel.BorderColor = app.ColorBorder;
            panel.BorderWidth = 1;

            g = uigridlayout(panel, [2 1]);
            g.RowHeight = {'fit', 'fit'};
            g.Padding = [12 10 12 10];
            g.RowSpacing = 4;

            titleLbl = uilabel(g);
            titleLbl.Layout.Row = 1;
            titleLbl.Layout.Column = 1;
            titleLbl.Text = titleText;
            titleLbl.FontSize = 10;
            titleLbl.FontColor = app.ColorTextSecondary;
            titleLbl.FontName = 'Segoe UI';

            valueLbl = uilabel(g);
            valueLbl.Layout.Row = 2;
            valueLbl.Layout.Column = 1;
            valueLbl.Text = valueText;
            valueLbl.FontSize = 20;
            valueLbl.FontWeight = 'bold';
            valueLbl.FontColor = app.ColorPrimary;
            valueLbl.FontName = 'Segoe UI';
        end

        % -----------------------------------------------------------------
        % HELPER: Create dashboard status card
        % -----------------------------------------------------------------
        function [panel, titleLbl, valueLbl] = createDashCard(app, parent, row, col, titleText, valueText)
            panel = uipanel(parent);
            panel.Layout.Row = row;
            panel.Layout.Column = col;
            panel.BackgroundColor = app.ColorCardBg;
            panel.BorderType = 'line';
            panel.BorderColor = app.ColorBorder;
            panel.BorderWidth = 1;

            g = uigridlayout(panel, [2 1]);
            g.RowHeight = {'fit', 'fit'};
            g.Padding = [14 10 14 10];
            g.RowSpacing = 6;

            titleLbl = uilabel(g);
            titleLbl.Layout.Row = 1;
            titleLbl.Layout.Column = 1;
            titleLbl.Text = titleText;
            titleLbl.FontSize = 10;
            titleLbl.FontWeight = 'bold';
            titleLbl.FontColor = app.ColorTextSecondary;
            titleLbl.FontName = 'Segoe UI';
            titleLbl.HorizontalAlignment = 'center';

            valueLbl = uilabel(g);
            valueLbl.Layout.Row = 2;
            valueLbl.Layout.Column = 1;
            valueLbl.Text = valueText;
            valueLbl.FontSize = 14;
            valueLbl.FontWeight = 'bold';
            valueLbl.FontColor = app.ColorTextPrimary;
            valueLbl.FontName = 'Segoe UI';
            valueLbl.HorizontalAlignment = 'center';
        end

    end % private methods

    % =====================================================================
    % UI CREATION
    % =====================================================================
    methods (Access = private)

        function createComponents(app)
            % ---------------------------------------------------------
            % MAIN FIGURE
            % ---------------------------------------------------------
            app.UIFigure = uifigure('Visible', 'off');
            app.UIFigure.Name = 'RETINA-AI | Diabetic Retinopathy Screening';
            app.UIFigure.Position = [50 50 1440 820];
            app.UIFigure.Color = app.ColorBgLight;
            app.UIFigure.Resize = 'on';

            % ---------------------------------------------------------
            % MAIN GRID: Sidebar + Content
            % ---------------------------------------------------------
            app.MainGrid = uigridlayout(app.UIFigure, [1 2]);
            app.MainGrid.ColumnWidth = {220, '1x'};
            app.MainGrid.RowHeight = {'1x'};
            app.MainGrid.Padding = [0 0 0 0];
            app.MainGrid.ColumnSpacing = 0;

            % ---------------------------------------------------------
            % SIDEBAR
            % ---------------------------------------------------------
            app.SidebarPanel = uipanel(app.MainGrid);
            app.SidebarPanel.Layout.Row = 1;
            app.SidebarPanel.Layout.Column = 1;
            app.SidebarPanel.BackgroundColor = app.ColorBgDark;
            app.SidebarPanel.BorderType = 'none';

            app.SidebarGrid = uigridlayout(app.SidebarPanel, [10 1]);
            app.SidebarGrid.RowHeight = {'fit', 'fit', 20, 40, 40, 40, 40, '1x', 'fit', 'fit'};
            app.SidebarGrid.Padding = [12 16 12 12];
            app.SidebarGrid.RowSpacing = 4;

            % Logo
            app.LogoLabel = uilabel(app.SidebarGrid);
            app.LogoLabel.Layout.Row = 1;
            app.LogoLabel.Layout.Column = 1;
            app.LogoLabel.Text = sprintf('%s  RETINA-AI', char(128065));
            app.LogoLabel.FontSize = 18;
            app.LogoLabel.FontWeight = 'bold';
            app.LogoLabel.FontColor = [1 1 1];
            app.LogoLabel.FontName = 'Segoe UI';
            app.LogoLabel.HorizontalAlignment = 'center';

            app.SubtitleLabel = uilabel(app.SidebarGrid);
            app.SubtitleLabel.Layout.Row = 2;
            app.SubtitleLabel.Layout.Column = 1;
            app.SubtitleLabel.Text = 'DR Screening System';
            app.SubtitleLabel.FontSize = 10;
            app.SubtitleLabel.FontColor = [0.55 0.60 0.65];
            app.SubtitleLabel.FontName = 'Segoe UI';
            app.SubtitleLabel.HorizontalAlignment = 'center';

            % Navigation Buttons
            navStyle = struct('FontSize', 13, 'FontName', 'Segoe UI', ...
                'FontColor', [0.65 0.70 0.75], 'BackgroundColor', app.ColorBgDark, ...
                'FontWeight', 'normal', 'HorizontalAlignment', 'left');

            app.NavDashboard = uibutton(app.SidebarGrid, 'push');
            app.NavDashboard.Layout.Row = 4;
            app.NavDashboard.Layout.Column = 1;
            app.NavDashboard.Text = sprintf('  %s  Dashboard', char(9632));
            app.NavDashboard.FontSize = navStyle.FontSize;
            app.NavDashboard.FontName = navStyle.FontName;
            app.NavDashboard.FontColor = [1 1 1];
            app.NavDashboard.BackgroundColor = [0.18 0.22 0.28];
            app.NavDashboard.HorizontalAlignment = 'left';
            app.NavDashboard.ButtonPushedFcn = @(~,~) app.showPage('dashboard');

            app.NavAnalyze = uibutton(app.SidebarGrid, 'push');
            app.NavAnalyze.Layout.Row = 5;
            app.NavAnalyze.Layout.Column = 1;
            app.NavAnalyze.Text = sprintf('  %s  Analyze Retina', char(128269));
            app.NavAnalyze.FontSize = navStyle.FontSize;
            app.NavAnalyze.FontName = navStyle.FontName;
            app.NavAnalyze.FontColor = navStyle.FontColor;
            app.NavAnalyze.BackgroundColor = navStyle.BackgroundColor;
            app.NavAnalyze.HorizontalAlignment = 'left';
            app.NavAnalyze.ButtonPushedFcn = @(~,~) app.showPage('analyze');

            app.NavInsights = uibutton(app.SidebarGrid, 'push');
            app.NavInsights.Layout.Row = 6;
            app.NavInsights.Layout.Column = 1;
            app.NavInsights.Text = sprintf('  %s  Model Insights', char(128202));
            app.NavInsights.FontSize = navStyle.FontSize;
            app.NavInsights.FontName = navStyle.FontName;
            app.NavInsights.FontColor = navStyle.FontColor;
            app.NavInsights.BackgroundColor = navStyle.BackgroundColor;
            app.NavInsights.HorizontalAlignment = 'left';
            app.NavInsights.ButtonPushedFcn = @(~,~) app.showPage('insights');

            app.NavAbout = uibutton(app.SidebarGrid, 'push');
            app.NavAbout.Layout.Row = 7;
            app.NavAbout.Layout.Column = 1;
            app.NavAbout.Text = sprintf('  %s  About', char(8505));
            app.NavAbout.FontSize = navStyle.FontSize;
            app.NavAbout.FontName = navStyle.FontName;
            app.NavAbout.FontColor = navStyle.FontColor;
            app.NavAbout.BackgroundColor = navStyle.BackgroundColor;
            app.NavAbout.HorizontalAlignment = 'left';
            app.NavAbout.ButtonPushedFcn = @(~,~) app.showPage('about');

            % Model status in sidebar
            app.ModelStatusDot = uilabel(app.SidebarGrid);
            app.ModelStatusDot.Layout.Row = 9;
            app.ModelStatusDot.Layout.Column = 1;
            app.ModelStatusDot.Text = [char(9679) ' '];
            app.ModelStatusDot.FontSize = 10;
            app.ModelStatusDot.FontColor = app.ColorWarning;
            app.ModelStatusDot.HorizontalAlignment = 'center';

            app.ModelStatusLabel = uilabel(app.SidebarGrid);
            app.ModelStatusLabel.Layout.Row = 10;
            app.ModelStatusLabel.Layout.Column = 1;
            app.ModelStatusLabel.Text = 'Model: Initializing...';
            app.ModelStatusLabel.FontSize = 10;
            app.ModelStatusLabel.FontColor = app.ColorWarning;
            app.ModelStatusLabel.FontName = 'Segoe UI';
            app.ModelStatusLabel.HorizontalAlignment = 'center';

            % ---------------------------------------------------------
            % CONTENT PANEL (hosts all pages)
            % ---------------------------------------------------------
            app.ContentPanel = uipanel(app.MainGrid);
            app.ContentPanel.Layout.Row = 1;
            app.ContentPanel.Layout.Column = 2;
            app.ContentPanel.BackgroundColor = app.ColorBgLight;
            app.ContentPanel.BorderType = 'none';

            % Create all pages
            app.createDashboardPage();
            app.createAnalyzePage();
            app.createInsightsPage();
            app.createAboutPage();
        end

        % =================================================================
        % DASHBOARD PAGE
        % =================================================================
        function createDashboardPage(app)
            app.DashboardGrid = uigridlayout(app.ContentPanel, [8 5]);
            app.DashboardGrid.RowHeight = {50, 'fit', 'fit', 20, 'fit', 20, 50, '1x'};
            app.DashboardGrid.ColumnWidth = {'1x', '1x', '1x', '1x', '1x'};
            app.DashboardGrid.Padding = [40 30 40 20];
            app.DashboardGrid.RowSpacing = 10;
            app.DashboardGrid.ColumnSpacing = 14;
            app.DashboardGrid.BackgroundColor = app.ColorBgLight;

            % Header
            app.DashHeaderLabel = uilabel(app.DashboardGrid);
            app.DashHeaderLabel.Layout.Row = 1;
            app.DashHeaderLabel.Layout.Column = [1 5];
            app.DashHeaderLabel.Text = sprintf('%s  RETINA-AI', char(128065));
            app.DashHeaderLabel.FontSize = 28;
            app.DashHeaderLabel.FontWeight = 'bold';
            app.DashHeaderLabel.FontColor = app.ColorTextPrimary;
            app.DashHeaderLabel.FontName = 'Segoe UI';
            app.DashHeaderLabel.VerticalAlignment = 'bottom';

            app.DashSubLabel = uilabel(app.DashboardGrid);
            app.DashSubLabel.Layout.Row = 2;
            app.DashSubLabel.Layout.Column = [1 5];
            app.DashSubLabel.Text = 'AI-Assisted Diabetic Retinopathy Screening & Analysis';
            app.DashSubLabel.FontSize = 15;
            app.DashSubLabel.FontColor = app.ColorTextSecondary;
            app.DashSubLabel.FontName = 'Segoe UI';

            app.DashTagsLabel = uilabel(app.DashboardGrid);
            app.DashTagsLabel.Layout.Row = 3;
            app.DashTagsLabel.Layout.Column = [1 5];
            app.DashTagsLabel.Text = 'MATLAB  •  Deep Learning  •  ResNet-101  •  Retinal Image Analysis';
            app.DashTagsLabel.FontSize = 11;
            app.DashTagsLabel.FontColor = app.ColorAccent;
            app.DashTagsLabel.FontName = 'Segoe UI';

            % Status Cards (Row 5)
            [app.CardModelStatus, app.CardModelStatusTitle, app.CardModelStatusValue] = ...
                app.createDashCard(app.DashboardGrid, 5, 1, 'MODEL STATUS', 'Initializing...');

            [app.CardModelName, app.CardModelNameTitle, app.CardModelNameValue] = ...
                app.createDashCard(app.DashboardGrid, 5, 2, 'CHECKPOINT', 'AG_V4_5_ReducedAug');

            [app.CardInputSpec, app.CardInputSpecTitle, app.CardInputSpecValue] = ...
                app.createDashCard(app.DashboardGrid, 5, 3, 'INPUT', '224 × 224 RGB');

            [app.CardClasses, app.CardClassesTitle, app.CardClassesValue] = ...
                app.createDashCard(app.DashboardGrid, 5, 4, 'CLASSES', '5 DR Stages');

            [app.CardScreening, app.CardScreeningTitle, app.CardScreeningValue] = ...
                app.createDashCard(app.DashboardGrid, 5, 5, 'SCREENING', 'Ref / Non-Ref');

            % CTA Button
            app.DashCTAButton = uibutton(app.DashboardGrid, 'push');
            app.DashCTAButton.Layout.Row = 7;
            app.DashCTAButton.Layout.Column = [2 4];
            app.DashCTAButton.Text = sprintf('%s  Analyze Retinal Image', char(9654));
            app.DashCTAButton.FontSize = 16;
            app.DashCTAButton.FontWeight = 'bold';
            app.DashCTAButton.FontColor = [1 1 1];
            app.DashCTAButton.BackgroundColor = app.ColorPrimary;
            app.DashCTAButton.FontName = 'Segoe UI';
            app.DashCTAButton.ButtonPushedFcn = @(~,~) app.showPage('analyze');

            % Pipeline visualization
            app.DashPipelineLabel = uilabel(app.DashboardGrid);
            app.DashPipelineLabel.Layout.Row = 8;
            app.DashPipelineLabel.Layout.Column = [1 2];
            app.DashPipelineLabel.Text = 'ANALYSIS PIPELINE';
            app.DashPipelineLabel.FontSize = 11;
            app.DashPipelineLabel.FontWeight = 'bold';
            app.DashPipelineLabel.FontColor = app.ColorTextSecondary;
            app.DashPipelineLabel.FontName = 'Segoe UI';
            app.DashPipelineLabel.VerticalAlignment = 'top';

            pipeText = sprintf([ ...
                '  Fundus Image\n' ...
                '       %s\n' ...
                '  Preprocess (224×224)\n' ...
                '       %s\n' ...
                '  ResNet-101 Model\n' ...
                '       %s\n' ...
                '  5-Class Prediction\n' ...
                '       %s\n' ...
                '  Screening Decision'], ...
                char(8595), char(8595), char(8595), char(8595));

            app.DashPipelineText = uilabel(app.DashboardGrid);
            app.DashPipelineText.Layout.Row = 8;
            app.DashPipelineText.Layout.Column = [3 5];
            app.DashPipelineText.Text = pipeText;
            app.DashPipelineText.FontSize = 12;
            app.DashPipelineText.FontColor = app.ColorTextPrimary;
            app.DashPipelineText.FontName = 'Consolas';
            app.DashPipelineText.VerticalAlignment = 'top';
        end

        % =================================================================
        % ANALYZE PAGE
        % =================================================================
        function createAnalyzePage(app)
            app.AnalyzeGrid = uigridlayout(app.ContentPanel, [2 3]);
            app.AnalyzeGrid.RowHeight = {'3x', '1x'};
            app.AnalyzeGrid.ColumnWidth = {'1x', '1x', '1x'};
            app.AnalyzeGrid.Padding = [16 12 16 12];
            app.AnalyzeGrid.RowSpacing = 10;
            app.AnalyzeGrid.ColumnSpacing = 10;
            app.AnalyzeGrid.Visible = 'off';
            app.AnalyzeGrid.BackgroundColor = app.ColorBgLight;

            % =============================================================
            % LEFT COLUMN: Input Image
            % =============================================================
            app.InputPanel = uipanel(app.AnalyzeGrid);
            app.InputPanel.Layout.Row = 1;
            app.InputPanel.Layout.Column = 1;
            app.InputPanel.Title = '';
            app.InputPanel.BackgroundColor = app.ColorCardBg;
            app.InputPanel.BorderType = 'line';
            app.InputPanel.BorderColor = app.ColorBorder;
            app.InputPanel.BorderWidth = 1;

            app.InputGrid = uigridlayout(app.InputPanel, [5 1]);
            app.InputGrid.RowHeight = {'fit', '1x', 'fit', 'fit', 'fit'};
            app.InputGrid.Padding = [14 12 14 12];
            app.InputGrid.RowSpacing = 8;

            app.InputTitleLabel = uilabel(app.InputGrid);
            app.InputTitleLabel.Layout.Row = 1;
            app.InputTitleLabel.Layout.Column = 1;
            app.InputTitleLabel.Text = 'INPUT IMAGE';
            app.InputTitleLabel.FontSize = 12;
            app.InputTitleLabel.FontWeight = 'bold';
            app.InputTitleLabel.FontColor = app.ColorTextPrimary;
            app.InputTitleLabel.FontName = 'Segoe UI';

            app.InputAxes = uiaxes(app.InputGrid);
            app.InputAxes.Layout.Row = 2;
            app.InputAxes.Layout.Column = 1;
            app.InputAxes.Visible = 'off';
            app.InputAxes.XTick = [];
            app.InputAxes.YTick = [];
            app.InputAxes.Box = 'on';
            app.InputAxes.Color = [0.97 0.97 0.97];
            app.InputAxes.XColor = app.ColorBorder;
            app.InputAxes.YColor = app.ColorBorder;

            app.InputEmptyLabel = uilabel(app.InputGrid);
            app.InputEmptyLabel.Layout.Row = 2;
            app.InputEmptyLabel.Layout.Column = 1;
            app.InputEmptyLabel.Text = sprintf('%s\n\nUpload a retinal fundus image\nto begin analysis', char(128065));
            app.InputEmptyLabel.FontSize = 13;
            app.InputEmptyLabel.FontColor = app.ColorTextSecondary;
            app.InputEmptyLabel.FontName = 'Segoe UI';
            app.InputEmptyLabel.HorizontalAlignment = 'center';
            app.InputEmptyLabel.VerticalAlignment = 'center';

            app.InputInfoLabel = uilabel(app.InputGrid);
            app.InputInfoLabel.Layout.Row = 3;
            app.InputInfoLabel.Layout.Column = 1;
            app.InputInfoLabel.Text = '';
            app.InputInfoLabel.FontSize = 10;
            app.InputInfoLabel.FontColor = app.ColorTextSecondary;
            app.InputInfoLabel.FontName = 'Segoe UI';
            app.InputInfoLabel.Visible = 'off';

            app.ChooseImageBtn = uibutton(app.InputGrid, 'push');
            app.ChooseImageBtn.Layout.Row = 4;
            app.ChooseImageBtn.Layout.Column = 1;
            app.ChooseImageBtn.Text = sprintf('%s  Choose Image', char(128194));
            app.ChooseImageBtn.FontSize = 13;
            app.ChooseImageBtn.FontColor = [1 1 1];
            app.ChooseImageBtn.BackgroundColor = app.ColorPrimary;
            app.ChooseImageBtn.FontName = 'Segoe UI';
            app.ChooseImageBtn.FontWeight = 'bold';
            app.ChooseImageBtn.ButtonPushedFcn = @(~,~) app.selectImage();

            % =============================================================
            % CENTER COLUMN: Model View + Pipeline
            % =============================================================
            app.ModelViewPanel = uipanel(app.AnalyzeGrid);
            app.ModelViewPanel.Layout.Row = 1;
            app.ModelViewPanel.Layout.Column = 2;
            app.ModelViewPanel.Title = '';
            app.ModelViewPanel.BackgroundColor = app.ColorCardBg;
            app.ModelViewPanel.BorderType = 'line';
            app.ModelViewPanel.BorderColor = app.ColorBorder;
            app.ModelViewPanel.BorderWidth = 1;

            app.ModelViewGrid = uigridlayout(app.ModelViewPanel, [9 2]);
            app.ModelViewGrid.RowHeight = {'fit', '1x', 10, 'fit', 'fit', 'fit', 'fit', 14, 'fit'};
            app.ModelViewGrid.ColumnWidth = {'1x', '1x'};
            app.ModelViewGrid.Padding = [14 12 14 12];
            app.ModelViewGrid.RowSpacing = 6;

            app.ModelViewTitle = uilabel(app.ModelViewGrid);
            app.ModelViewTitle.Layout.Row = 1;
            app.ModelViewTitle.Layout.Column = [1 2];
            app.ModelViewTitle.Text = 'MODEL INPUT (224 × 224)';
            app.ModelViewTitle.FontSize = 12;
            app.ModelViewTitle.FontWeight = 'bold';
            app.ModelViewTitle.FontColor = app.ColorTextPrimary;
            app.ModelViewTitle.FontName = 'Segoe UI';

            app.ModelInputAxes = uiaxes(app.ModelViewGrid);
            app.ModelInputAxes.Layout.Row = 2;
            app.ModelInputAxes.Layout.Column = [1 2];
            app.ModelInputAxes.Visible = 'off';
            app.ModelInputAxes.XTick = [];
            app.ModelInputAxes.YTick = [];
            app.ModelInputAxes.Box = 'on';
            app.ModelInputAxes.Color = [0.97 0.97 0.97];
            app.ModelInputAxes.XColor = app.ColorBorder;
            app.ModelInputAxes.YColor = app.ColorBorder;

            % Pipeline status labels
            pipelineTitleLbl = uilabel(app.ModelViewGrid);
            pipelineTitleLbl.Layout.Row = 3;
            pipelineTitleLbl.Layout.Column = [1 2];
            pipelineTitleLbl.Text = '';

            app.PipelineStep1 = uilabel(app.ModelViewGrid);
            app.PipelineStep1.Layout.Row = 4;
            app.PipelineStep1.Layout.Column = [1 2];
            app.PipelineStep1.Text = sprintf('%s  Image loaded', char(9675));
            app.PipelineStep1.FontSize = 11;
            app.PipelineStep1.FontColor = app.ColorTextSecondary;
            app.PipelineStep1.FontName = 'Segoe UI';

            app.PipelineStep2 = uilabel(app.ModelViewGrid);
            app.PipelineStep2.Layout.Row = 5;
            app.PipelineStep2.Layout.Column = [1 2];
            app.PipelineStep2.Text = sprintf('%s  Preprocessing', char(9675));
            app.PipelineStep2.FontSize = 11;
            app.PipelineStep2.FontColor = app.ColorTextSecondary;
            app.PipelineStep2.FontName = 'Segoe UI';

            app.PipelineStep3 = uilabel(app.ModelViewGrid);
            app.PipelineStep3.Layout.Row = 6;
            app.PipelineStep3.Layout.Column = [1 2];
            app.PipelineStep3.Text = sprintf('%s  Model inference', char(9675));
            app.PipelineStep3.FontSize = 11;
            app.PipelineStep3.FontColor = app.ColorTextSecondary;
            app.PipelineStep3.FontName = 'Segoe UI';

            app.PipelineStep4 = uilabel(app.ModelViewGrid);
            app.PipelineStep4.Layout.Row = 7;
            app.PipelineStep4.Layout.Column = [1 2];
            app.PipelineStep4.Text = sprintf('%s  Screening decision', char(9675));
            app.PipelineStep4.FontSize = 11;
            app.PipelineStep4.FontColor = app.ColorTextSecondary;
            app.PipelineStep4.FontName = 'Segoe UI';

            % Analyze & Reset buttons
            app.AnalyzeBtn = uibutton(app.ModelViewGrid, 'push');
            app.AnalyzeBtn.Layout.Row = 9;
            app.AnalyzeBtn.Layout.Column = 1;
            app.AnalyzeBtn.Text = [char(9654) ' Analyze'];
            app.AnalyzeBtn.FontSize = 14;
            app.AnalyzeBtn.FontWeight = 'bold';
            app.AnalyzeBtn.FontColor = [1 1 1];
            app.AnalyzeBtn.BackgroundColor = [0.75 0.78 0.80];
            app.AnalyzeBtn.FontName = 'Segoe UI';
            app.AnalyzeBtn.Enable = 'off';
            app.AnalyzeBtn.ButtonPushedFcn = @(~,~) app.runInference();

            app.ResetBtn = uibutton(app.ModelViewGrid, 'push');
            app.ResetBtn.Layout.Row = 9;
            app.ResetBtn.Layout.Column = 2;
            app.ResetBtn.Text = sprintf('%s Reset', char(8635));
            app.ResetBtn.FontSize = 13;
            app.ResetBtn.FontColor = app.ColorTextSecondary;
            app.ResetBtn.BackgroundColor = app.ColorCardBg;
            app.ResetBtn.FontName = 'Segoe UI';
            app.ResetBtn.ButtonPushedFcn = @(~,~) app.resetAnalysis();

            % =============================================================
            % RIGHT COLUMN: Results
            % =============================================================
            app.ResultPanel = uipanel(app.AnalyzeGrid);
            app.ResultPanel.Layout.Row = 1;
            app.ResultPanel.Layout.Column = 3;
            app.ResultPanel.Title = '';
            app.ResultPanel.BackgroundColor = app.ColorCardBg;
            app.ResultPanel.BorderType = 'line';
            app.ResultPanel.BorderColor = app.ColorBorder;
            app.ResultPanel.BorderWidth = 1;

            app.ResultGrid = uigridlayout(app.ResultPanel, [12 1]);
            app.ResultGrid.RowHeight = {'fit', 8, 'fit', 'fit', 8, 'fit', 'fit', 8, 'fit', 'fit', '1x', 'fit'};
            app.ResultGrid.Padding = [14 12 14 12];
            app.ResultGrid.RowSpacing = 4;

            app.ResultTitleLabel = uilabel(app.ResultGrid);
            app.ResultTitleLabel.Layout.Row = 1;
            app.ResultTitleLabel.Layout.Column = 1;
            app.ResultTitleLabel.Text = 'AWAITING ANALYSIS';
            app.ResultTitleLabel.FontSize = 12;
            app.ResultTitleLabel.FontWeight = 'bold';
            app.ResultTitleLabel.FontColor = app.ColorTextSecondary;
            app.ResultTitleLabel.FontName = 'Segoe UI';

            app.PredStageTitle = uilabel(app.ResultGrid);
            app.PredStageTitle.Layout.Row = 3;
            app.PredStageTitle.Layout.Column = 1;
            app.PredStageTitle.Text = 'PREDICTED STAGE';
            app.PredStageTitle.FontSize = 10;
            app.PredStageTitle.FontColor = app.ColorTextSecondary;
            app.PredStageTitle.FontName = 'Segoe UI';
            app.PredStageTitle.Visible = 'off';

            app.PredStageValue = uilabel(app.ResultGrid);
            app.PredStageValue.Layout.Row = 4;
            app.PredStageValue.Layout.Column = 1;
            app.PredStageValue.Text = '';
            app.PredStageValue.FontSize = 22;
            app.PredStageValue.FontWeight = 'bold';
            app.PredStageValue.FontName = 'Segoe UI';
            app.PredStageValue.Visible = 'off';

            app.ConfidenceTitle = uilabel(app.ResultGrid);
            app.ConfidenceTitle.Layout.Row = 6;
            app.ConfidenceTitle.Layout.Column = 1;
            app.ConfidenceTitle.Text = 'CONFIDENCE';
            app.ConfidenceTitle.FontSize = 10;
            app.ConfidenceTitle.FontColor = app.ColorTextSecondary;
            app.ConfidenceTitle.FontName = 'Segoe UI';
            app.ConfidenceTitle.Visible = 'off';

            app.ConfidenceValue = uilabel(app.ResultGrid);
            app.ConfidenceValue.Layout.Row = 7;
            app.ConfidenceValue.Layout.Column = 1;
            app.ConfidenceValue.Text = '';
            app.ConfidenceValue.FontSize = 20;
            app.ConfidenceValue.FontWeight = 'bold';
            app.ConfidenceValue.FontName = 'Segoe UI';
            app.ConfidenceValue.Visible = 'off';

            app.ScreeningTitle = uilabel(app.ResultGrid);
            app.ScreeningTitle.Layout.Row = 9;
            app.ScreeningTitle.Layout.Column = 1;
            app.ScreeningTitle.Text = 'SCREENING DECISION';
            app.ScreeningTitle.FontSize = 10;
            app.ScreeningTitle.FontColor = app.ColorTextSecondary;
            app.ScreeningTitle.FontName = 'Segoe UI';
            app.ScreeningTitle.Visible = 'off';

            app.ScreeningValue = uilabel(app.ResultGrid);
            app.ScreeningValue.Layout.Row = 10;
            app.ScreeningValue.Layout.Column = 1;
            app.ScreeningValue.Text = '';
            app.ScreeningValue.FontSize = 16;
            app.ScreeningValue.FontWeight = 'bold';
            app.ScreeningValue.FontName = 'Segoe UI';
            app.ScreeningValue.Visible = 'off';

            app.ScreeningExplain = uilabel(app.ResultGrid);
            app.ScreeningExplain.Layout.Row = 11;
            app.ScreeningExplain.Layout.Column = 1;
            app.ScreeningExplain.Text = '';
            app.ScreeningExplain.FontSize = 10;
            app.ScreeningExplain.FontColor = app.ColorTextSecondary;
            app.ScreeningExplain.FontName = 'Segoe UI';
            app.ScreeningExplain.Visible = 'off';
            app.ScreeningExplain.WordWrap = 'on';
            app.ScreeningExplain.VerticalAlignment = 'top';

            app.ExportBtn = uibutton(app.ResultGrid, 'push');
            app.ExportBtn.Layout.Row = 12;
            app.ExportBtn.Layout.Column = 1;
            app.ExportBtn.Text = sprintf('%s  Export Report', char(128196));
            app.ExportBtn.FontSize = 12;
            app.ExportBtn.FontColor = [1 1 1];
            app.ExportBtn.BackgroundColor = [0.75 0.78 0.80];
            app.ExportBtn.FontName = 'Segoe UI';
            app.ExportBtn.Enable = 'off';
            app.ExportBtn.ButtonPushedFcn = @(~,~) app.exportReport();

            % =============================================================
            % BOTTOM ROW: Probabilities + History
            % =============================================================
            % Probability chart panel
            probPanel = uipanel(app.AnalyzeGrid);
            probPanel.Layout.Row = 2;
            probPanel.Layout.Column = [1 2];
            probPanel.BackgroundColor = app.ColorCardBg;
            probPanel.BorderType = 'line';
            probPanel.BorderColor = app.ColorBorder;
            probPanel.BorderWidth = 1;

            probGrid = uigridlayout(probPanel, [2 1]);
            probGrid.RowHeight = {'fit', '1x'};
            probGrid.Padding = [14 8 14 8];
            probGrid.RowSpacing = 4;

            app.ProbTitle = uilabel(probGrid);
            app.ProbTitle.Layout.Row = 1;
            app.ProbTitle.Layout.Column = 1;
            app.ProbTitle.Text = 'PROBABILITY DISTRIBUTION';
            app.ProbTitle.FontSize = 11;
            app.ProbTitle.FontWeight = 'bold';
            app.ProbTitle.FontColor = app.ColorTextPrimary;
            app.ProbTitle.FontName = 'Segoe UI';
            app.ProbTitle.Visible = 'off';

            app.ProbAxes = uiaxes(probGrid);
            app.ProbAxes.Layout.Row = 2;
            app.ProbAxes.Layout.Column = 1;
            app.ProbAxes.Visible = 'off';
            app.ProbAxes.FontName = 'Segoe UI';
            app.ProbAxes.FontSize = 10;
            app.ProbAxes.Box = 'off';
            app.ProbAxes.Color = app.ColorCardBg;

            % History panel
            app.HistoryPanel = uipanel(app.AnalyzeGrid);
            app.HistoryPanel.Layout.Row = 2;
            app.HistoryPanel.Layout.Column = 3;
            app.HistoryPanel.BackgroundColor = app.ColorCardBg;
            app.HistoryPanel.BorderType = 'line';
            app.HistoryPanel.BorderColor = app.ColorBorder;
            app.HistoryPanel.BorderWidth = 1;

            app.HistoryGrid = uigridlayout(app.HistoryPanel, [2 1]);
            app.HistoryGrid.RowHeight = {'fit', '1x'};
            app.HistoryGrid.Padding = [14 8 14 8];
            app.HistoryGrid.RowSpacing = 6;

            app.HistoryTitle = uilabel(app.HistoryGrid);
            app.HistoryTitle.Layout.Row = 1;
            app.HistoryTitle.Layout.Column = 1;
            app.HistoryTitle.Text = 'SESSION HISTORY';
            app.HistoryTitle.FontSize = 11;
            app.HistoryTitle.FontWeight = 'bold';
            app.HistoryTitle.FontColor = app.ColorTextPrimary;
            app.HistoryTitle.FontName = 'Segoe UI';

            app.HistoryListBox = uilistbox(app.HistoryGrid);
            app.HistoryListBox.Layout.Row = 2;
            app.HistoryListBox.Layout.Column = 1;
            app.HistoryListBox.Items = {'No analyses yet'};
            app.HistoryListBox.FontSize = 10;
            app.HistoryListBox.FontName = 'Segoe UI';
            app.HistoryListBox.FontColor = app.ColorTextSecondary;
            app.HistoryListBox.ValueChangedFcn = @(src, ~) app.onHistorySelected(src);
        end

        % =================================================================
        % MODEL INSIGHTS PAGE
        % =================================================================
        function createInsightsPage(app)
            % Scrollable grid for insights
            app.InsightsGrid = uigridlayout(app.ContentPanel, [7 4]);
            app.InsightsGrid.RowHeight = {'fit', 'fit', 100, 100, 'fit', '1x', '1x'};
            app.InsightsGrid.ColumnWidth = {'1x', '1x', '1x', '1x'};
            app.InsightsGrid.Padding = [30 20 30 20];
            app.InsightsGrid.RowSpacing = 12;
            app.InsightsGrid.ColumnSpacing = 12;
            app.InsightsGrid.Visible = 'off';
            app.InsightsGrid.BackgroundColor = app.ColorBgLight;
            app.InsightsGrid.Scrollable = 'on';

            % Title
            app.InsightsTitle = uilabel(app.InsightsGrid);
            app.InsightsTitle.Layout.Row = 1;
            app.InsightsTitle.Layout.Column = [1 4];
            app.InsightsTitle.Text = sprintf('%s  Model Performance Insights', char(128202));
            app.InsightsTitle.FontSize = 22;
            app.InsightsTitle.FontWeight = 'bold';
            app.InsightsTitle.FontColor = app.ColorTextPrimary;
            app.InsightsTitle.FontName = 'Segoe UI';

            app.InsightsSubTitle = uilabel(app.InsightsGrid);
            app.InsightsSubTitle.Layout.Row = 2;
            app.InsightsSubTitle.Layout.Column = [1 4];
            app.InsightsSubTitle.Text = 'Final frozen test evaluation  •  AG_V4_5_ReducedAug  •  Test Set N=5,268';
            app.InsightsSubTitle.FontSize = 12;
            app.InsightsSubTitle.FontColor = app.ColorTextSecondary;
            app.InsightsSubTitle.FontName = 'Segoe UI';

            % Metric cards (Row 3: first 4)
            [app.MetricROCAUC, app.MetricROCAUCTitle, app.MetricROCAUCValue] = ...
                app.createMetricCard(app.InsightsGrid, 3, 1, 'ROC-AUC', '—');
            [app.MetricPRAUC, app.MetricPRAUCTitle, app.MetricPRAUCValue] = ...
                app.createMetricCard(app.InsightsGrid, 3, 2, 'PR-AUC', '—');
            [app.MetricSens, app.MetricSensTitle, app.MetricSensValue] = ...
                app.createMetricCard(app.InsightsGrid, 3, 3, 'SENSITIVITY', '—');
            [app.MetricSpec, app.MetricSpecTitle, app.MetricSpecValue] = ...
                app.createMetricCard(app.InsightsGrid, 3, 4, 'SPECIFICITY', '—');

            % Metric cards (Row 4: second 4)
            [app.MetricAcc5, app.MetricAcc5Title, app.MetricAcc5Value] = ...
                app.createMetricCard(app.InsightsGrid, 4, 1, '5-CLASS ACCURACY', '—');
            [app.MetricClinAcc, app.MetricClinAccTitle, app.MetricClinAccValue] = ...
                app.createMetricCard(app.InsightsGrid, 4, 2, 'CLINICAL ACCURACY', '—');
            [app.MetricPPV, app.MetricPPVTitle, app.MetricPPVValue] = ...
                app.createMetricCard(app.InsightsGrid, 4, 3, 'PPV', '—');
            [app.MetricNPV, app.MetricNPVTitle, app.MetricNPVValue] = ...
                app.createMetricCard(app.InsightsGrid, 4, 4, 'NPV', '—');

            % Clinical Confusion Matrix + Model Info (Row 5)
            app.ClinCMPanel = uipanel(app.InsightsGrid);
            app.ClinCMPanel.Layout.Row = 5;
            app.ClinCMPanel.Layout.Column = [1 2];
            app.ClinCMPanel.BackgroundColor = app.ColorCardBg;
            app.ClinCMPanel.BorderType = 'line';
            app.ClinCMPanel.BorderColor = app.ColorBorder;

            cmGrid = uigridlayout(app.ClinCMPanel, [2 1]);
            cmGrid.RowHeight = {'fit', 'fit'};
            cmGrid.Padding = [14 10 14 10];
            cmGrid.RowSpacing = 8;

            app.ClinCMTitle = uilabel(cmGrid);
            app.ClinCMTitle.Layout.Row = 1;
            app.ClinCMTitle.Layout.Column = 1;
            app.ClinCMTitle.Text = 'CLINICAL CONFUSION MATRIX';
            app.ClinCMTitle.FontSize = 11;
            app.ClinCMTitle.FontWeight = 'bold';
            app.ClinCMTitle.FontColor = app.ColorTextPrimary;
            app.ClinCMTitle.FontName = 'Segoe UI';

            app.ClinCMText = uitextarea(cmGrid);
            app.ClinCMText.Layout.Row = 2;
            app.ClinCMText.Layout.Column = 1;
            app.ClinCMText.Value = 'Loading...';
            app.ClinCMText.FontSize = 11;
            app.ClinCMText.FontName = 'Consolas';
            app.ClinCMText.FontColor = app.ColorTextPrimary;
            app.ClinCMText.BackgroundColor = [0.97 0.98 0.99];
            app.ClinCMText.Editable = 'off';

            app.ModelInfoPanel = uipanel(app.InsightsGrid);
            app.ModelInfoPanel.Layout.Row = 5;
            app.ModelInfoPanel.Layout.Column = [3 4];
            app.ModelInfoPanel.BackgroundColor = app.ColorCardBg;
            app.ModelInfoPanel.BorderType = 'line';
            app.ModelInfoPanel.BorderColor = app.ColorBorder;

            miGrid = uigridlayout(app.ModelInfoPanel, [2 1]);
            miGrid.RowHeight = {'fit', 'fit'};
            miGrid.Padding = [14 10 14 10];
            miGrid.RowSpacing = 8;

            app.ModelInfoTitle = uilabel(miGrid);
            app.ModelInfoTitle.Layout.Row = 1;
            app.ModelInfoTitle.Layout.Column = 1;
            app.ModelInfoTitle.Text = 'MODEL INFORMATION';
            app.ModelInfoTitle.FontSize = 11;
            app.ModelInfoTitle.FontWeight = 'bold';
            app.ModelInfoTitle.FontColor = app.ColorTextPrimary;
            app.ModelInfoTitle.FontName = 'Segoe UI';

            modelInfoStr = sprintf([ ...
                'Architecture:     ResNet-101 (~42.56M parameters)\n' ...
                'Framework:        MATLAB Deep Learning Toolbox\n' ...
                'Network Type:     dlnetwork (trainnet)\n' ...
                'Input Size:       224 x 224 x 3 RGB\n' ...
                'Classes:          5 (Mild, Moderate, No_DR, Proliferate_DR, Severe)\n' ...
                'Loss Function:    Cross-Entropy\n' ...
                'Optimizer:        SGDM (Momentum=0.9)\n' ...
                'Augmentation:     Rotation [-5,5], X-Reflection\n' ...
                'Checkpoint:       AG_V4_5_ReducedAug\n' ...
                'Threshold:        0.1199 (Validation-Locked)']);

            app.ModelInfoText = uilabel(miGrid);
            app.ModelInfoText.Layout.Row = 2;
            app.ModelInfoText.Layout.Column = 1;
            app.ModelInfoText.Text = modelInfoStr;
            app.ModelInfoText.FontSize = 11;
            app.ModelInfoText.FontName = 'Consolas';
            app.ModelInfoText.FontColor = app.ColorTextPrimary;

            % ROC/PR Curves figure (Row 6)
            app.ROCPRPanel = uipanel(app.InsightsGrid);
            app.ROCPRPanel.Layout.Row = 6;
            app.ROCPRPanel.Layout.Column = [1 2];
            app.ROCPRPanel.BackgroundColor = app.ColorCardBg;
            app.ROCPRPanel.BorderType = 'line';
            app.ROCPRPanel.BorderColor = app.ColorBorder;

            rocGrid = uigridlayout(app.ROCPRPanel, [2 1]);
            rocGrid.RowHeight = {'fit', '1x'};
            rocGrid.Padding = [14 10 14 10];
            rocGrid.RowSpacing = 6;

            app.ROCPRTitle = uilabel(rocGrid);
            app.ROCPRTitle.Layout.Row = 1;
            app.ROCPRTitle.Layout.Column = 1;
            app.ROCPRTitle.Text = 'ROC & PRECISION-RECALL CURVES';
            app.ROCPRTitle.FontSize = 11;
            app.ROCPRTitle.FontWeight = 'bold';
            app.ROCPRTitle.FontColor = app.ColorTextPrimary;
            app.ROCPRTitle.FontName = 'Segoe UI';

            app.ROCPRImage = uiimage(rocGrid);
            app.ROCPRImage.Layout.Row = 2;
            app.ROCPRImage.Layout.Column = 1;
            app.ROCPRImage.ScaleMethod = 'fit';

            rocFigPath = fullfile(app.AppRootPath, 'ANTIGRAVITY', 'figures', 'final_test_roc_pr_curves.png');
            if isfile(rocFigPath)
                app.ROCPRImage.ImageSource = rocFigPath;
            end

            % Moderate vs NoDR figure (Row 6)
            app.ModNoDRPanel = uipanel(app.InsightsGrid);
            app.ModNoDRPanel.Layout.Row = 6;
            app.ModNoDRPanel.Layout.Column = [3 4];
            app.ModNoDRPanel.BackgroundColor = app.ColorCardBg;
            app.ModNoDRPanel.BorderType = 'line';
            app.ModNoDRPanel.BorderColor = app.ColorBorder;

            modGrid = uigridlayout(app.ModNoDRPanel, [2 1]);
            modGrid.RowHeight = {'fit', '1x'};
            modGrid.Padding = [14 10 14 10];
            modGrid.RowSpacing = 6;

            app.ModNoDRTitle = uilabel(modGrid);
            app.ModNoDRTitle.Layout.Row = 1;
            app.ModNoDRTitle.Layout.Column = 1;
            app.ModNoDRTitle.Text = 'MODERATE vs NO_DR ANALYSIS';
            app.ModNoDRTitle.FontSize = 11;
            app.ModNoDRTitle.FontWeight = 'bold';
            app.ModNoDRTitle.FontColor = app.ColorTextPrimary;
            app.ModNoDRTitle.FontName = 'Segoe UI';

            app.ModNoDRImage = uiimage(modGrid);
            app.ModNoDRImage.Layout.Row = 2;
            app.ModNoDRImage.Layout.Column = 1;
            app.ModNoDRImage.ScaleMethod = 'fit';

            modFigPath = fullfile(app.AppRootPath, 'ANTIGRAVITY', 'figures', 'moderate_nodr_distribution.png');
            if isfile(modFigPath)
                app.ModNoDRImage.ImageSource = modFigPath;
            end
        end

        % =================================================================
        % ABOUT PAGE
        % =================================================================
        function createAboutPage(app)
            app.AboutGrid = uigridlayout(app.ContentPanel, [2 1]);
            app.AboutGrid.RowHeight = {'fit', '1x'};
            app.AboutGrid.Padding = [40 30 40 20];
            app.AboutGrid.RowSpacing = 16;
            app.AboutGrid.Visible = 'off';
            app.AboutGrid.BackgroundColor = app.ColorBgLight;

            app.AboutTitle = uilabel(app.AboutGrid);
            app.AboutTitle.Layout.Row = 1;
            app.AboutTitle.Layout.Column = 1;
            app.AboutTitle.Text = sprintf('%s  About RETINA-AI', char(8505));
            app.AboutTitle.FontSize = 22;
            app.AboutTitle.FontWeight = 'bold';
            app.AboutTitle.FontColor = app.ColorTextPrimary;
            app.AboutTitle.FontName = 'Segoe UI';

            aboutContent = {
                'ABOUT THE PROJECT'
                '═══════════════════════════════════════════════════'
                ''
                'Diabetic retinopathy (DR) is a leading cause of preventable blindness worldwide.'
                'Early detection through automated retinal screening can significantly improve'
                'patient outcomes by enabling timely intervention and referral.'
                ''
                ''
                'PROJECT OBJECTIVE'
                '═══════════════════════════════════════════════════'
                ''
                'Automated 5-class retinal fundus image classification with a primary focus on'
                'identifying referable diabetic retinopathy cases that require ophthalmological'
                'evaluation, while maintaining high specificity to minimize false referrals.'
                ''
                ''
                'CLASSIFICATION STAGES'
                '═══════════════════════════════════════════════════'
                ''
                '  1. Mild                — Early microaneurysms'
                '  2. Moderate            — More extensive microaneurysms, hemorrhages'
                '  3. No_DR              — No diabetic retinopathy detected'
                '  4. Proliferate_DR     — Neovascularization, advanced disease'
                '  5. Severe             — Extensive hemorrhages, venous beading'
                ''
                ''
                'CLINICAL SCREENING GROUPING'
                '═══════════════════════════════════════════════════'
                ''
                '  NON-REFERABLE:   Mild, No_DR'
                '      → Routine follow-up recommended'
                ''
                '  REFERABLE:       Moderate, Proliferate_DR, Severe'
                '      → Ophthalmological referral recommended'
                ''
                ''
                'TECHNOLOGY STACK'
                '═══════════════════════════════════════════════════'
                ''
                '  • MATLAB R2026a'
                '  • Deep Learning Toolbox'
                '  • Image Processing Toolbox'
                '  • ResNet-101 Architecture (~42.56M parameters)'
                '  • GPU-Accelerated Training & Inference'
                ''
                ''
                'DATASET'
                '═══════════════════════════════════════════════════'
                ''
                '  Kaggle: sovitrath/diabetic-retinopathy-2015-data-colored-resized'
                '  Total: 35,126 fundus images (224 × 224 × 3 RGB)'
                '  Split: 24,588 Train / 5,270 Validation / 5,268 Test'
                '  Stratified with fixed rng(42) for reproducibility'
                ''
                ''
                'RESEARCH DISCLAIMER'
                '═══════════════════════════════════════════════════'
                ''
                '  This is a research/experimental diabetic-retinopathy screening model'
                '  developed for academic and dataset-level evaluation purposes.'
                ''
                '  ⚠  It is NOT approved as a medical device.'
                '  ⚠  It is NOT clinically validated.'
                '  ⚠  It is NOT intended for diagnostic use.'
                ''
                '  Always consult a qualified ophthalmologist for clinical decisions'
                '  regarding diabetic retinopathy diagnosis and treatment.'
            };

            app.AboutText = uitextarea(app.AboutGrid);
            app.AboutText.Layout.Row = 2;
            app.AboutText.Layout.Column = 1;
            app.AboutText.Value = aboutContent;
            app.AboutText.FontSize = 12;
            app.AboutText.FontName = 'Consolas';
            app.AboutText.FontColor = app.ColorTextPrimary;
            app.AboutText.BackgroundColor = app.ColorCardBg;
            app.AboutText.Editable = 'off';
        end

        % -----------------------------------------------------------------
        % HISTORY SELECTION CALLBACK
        % -----------------------------------------------------------------
        function onHistorySelected(app, src)
            if isempty(src.Value) || app.HistoryCount == 0
                return;
            end
            idx = str2double(src.Value);
            if ~isnan(idx)
                app.restoreHistoryEntry(idx);
            end
        end

    end % private methods (UI creation)

    % =====================================================================
    % APP LIFECYCLE
    % =====================================================================
    methods (Access = public)

        function app = RetinaAI()
            % Constructor: Build and launch RETINA-AI

            % Determine application root
            appFile = mfilename('fullpath');
            app.AppRootPath = fileparts(appFile);
            if isempty(app.AppRootPath)
                app.AppRootPath = pwd;
            end

            % Add scripts to path
            scriptsPath = fullfile(app.AppRootPath, 'ANTIGRAVITY', 'scripts');
            if isfolder(scriptsPath)
                addpath(scriptsPath);
            end

            % Create UI
            app.createComponents();
            registerApp(app, app.UIFigure);

            % Show dashboard
            app.showPage('dashboard');

            % Make visible
            app.UIFigure.Visible = 'on';
            drawnow;

            % Load metrics for insights page
            app.loadModelMetrics();

            % Load model asynchronously
            app.loadModel();

            if nargout == 0
                clear app;
            end
        end

        function delete(app)
            delete(app.UIFigure);
        end
    end
end
