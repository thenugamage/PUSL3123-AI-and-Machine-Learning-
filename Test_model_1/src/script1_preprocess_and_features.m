%% ================================
%  PUSL3123 - SCRIPT 1
%  Pre-processing + Segmentation + Feature Extraction
%
%  Pipeline:
%   1) Read raw CSVs from data/raw
%   2) Clean, resample to 31 Hz, low-pass filter at 10 Hz
%   3) Compute GLOBAL z-score stats over all filtered data
%   4) Normalise (z-score), segment into 2 s windows (50% overlap)
%   5) Extract features per window and build:
%        - features_all.(mat/csv)      (FD + MD)
%        - reference_templates.(*)     (FD only)
%        - testing_templates.(*)       (MD only)
%   6) Analyse variability & uniqueness of features across users
%
%  Inputs:
%    data/raw/U1NW_FD.csv, U1NW_MD.csv, ..., U10NW_FD.csv, U10NW_MD.csv
%
%  Outputs:
%    data/interim/filtered_data/filtered_*.csv
%    data/interim/normalized_data/norm_*.csv
%    data/processed/features_all.(mat/csv)
%    data/processed/reference_templates.(mat/csv)
%    data/processed/testing_templates.(mat/csv)
%    data/processed/feature_variability_top.csv
%    data/processed/feature_means_per_user.csv
% =================================

clc; clear; close all;

%% --------- PROJECT FOLDER SETUP -----------------------------------------
projectRoot   = fileparts(fileparts(mfilename('fullpath')));  % src/.. -> project root

rawFolder     = fullfile(projectRoot, 'data', 'raw');
interimRoot   = fullfile(projectRoot, 'data', 'interim');
filteredFolder= fullfile(interimRoot, 'filtered_data');
normFolder    = fullfile(interimRoot, 'normalized_data');
processedRoot = fullfile(projectRoot, 'data', 'processed');

% Create folders if needed
if ~exist(interimRoot, 'dir');    mkdir(interimRoot);    end
if ~exist(filteredFolder, 'dir'); mkdir(filteredFolder); end
if ~exist(normFolder, 'dir');     mkdir(normFolder);     end
if ~exist(processedRoot, 'dir');  mkdir(processedRoot);  end

% Clear old CSVs in interim + old feature/template CSVs
delete(fullfile(filteredFolder, '*.csv'));
delete(fullfile(normFolder, '*.csv'));
delete(fullfile(processedRoot, 'features_all*.csv'));
delete(fullfile(processedRoot, 'reference_templates*.csv'));
delete(fullfile(processedRoot, 'testing_templates*.csv'));

% Feature output paths
featuresMatPath = fullfile(processedRoot, 'features_all.mat');
featuresCsvPath = fullfile(processedRoot, 'features_all.csv');
refMatPath      = fullfile(processedRoot, 'reference_templates.mat');
refCsvPath      = fullfile(processedRoot, 'reference_templates.csv');
testMatPath     = fullfile(processedRoot, 'testing_templates.mat');
testCsvPath     = fullfile(processedRoot, 'testing_templates.csv');

%% --- Config -------------------------------------------------------------
forceFs        = 31;   % Hz, forced sampling frequency
fc             = 10;   % Hz, low-pass cutoff
windowLengthSec= 2.0;  % seconds, segment length
overlapFraction= 0.5;  % 50% overlap

% Global z-score accumulators
sensorNames = {'AccX_f','AccY_f','AccZ_f','GyroX_f','GyroY_f','GyroZ_f'};
numSensors  = numel(sensorNames);
globalSum   = zeros(1, numSensors);
globalSqSum = zeros(1, numSensors);
globalN     = 0;

%% --- List raw files -----------------------------------------------------
fileList = dir(fullfile(rawFolder, '*.csv'));

fprintf('\n========================================\n');
fprintf('  SCRIPT 1: PRE-PROCESSING + FEATURE EXTRACTION\n');
fprintf('  Raw folder: %s\n', rawFolder);
fprintf('  Raw files found: %d\n', length(fileList));
fprintf('========================================\n');

if isempty(fileList)
    error('No raw CSV files found in: %s', rawFolder);
end

%% ========================================================================
%%  PASS 1: RAW -> FILTERED (and accumulate global stats)
%% ========================================================================
for k = 1:length(fileList)
    filename = fileList(k).name;
    fprintf('\n[PASS 1: %d/%d] Processing raw file: %s\n', k, length(fileList), filename);
    fprintf('----------------------------------------\n');
    
    filepath = fullfile(rawFolder, filename);
    raw = readtable(filepath, 'VariableNamingRule','preserve', ...
                    'ReadVariableNames', false);
    
    raw.Properties.VariableNames = {'Time','AccX','AccY','AccZ','GyroX','GyroY','GyroZ'};
    
    %% STEP 1: Quality checks
    fprintf('  STEP 1: Quality Checks\n');
    
    % Missing data
    if any(any(ismissing(raw))) || any(any(isnan(table2array(raw))))
        fprintf('    ⚠ Missing data detected → linear interpolate\n');
        raw = fillmissing(raw, 'linear');
    else
        fprintf('    ✓ No missing data\n');
    end
    
    % Exact duplicate rows
    beforeCount = height(raw);
    raw = unique(raw, 'rows', 'stable');
    afterCount  = height(raw);
    if afterCount < beforeCount
        fprintf('    ⚠ Removed %d duplicate rows\n', beforeCount - afterCount);
    else
        fprintf('    ✓ No duplicate rows\n');
    end
    
    % Flat signal warning
    sensorVars = {'AccX','AccY','AccZ','GyroX','GyroY','GyroZ'};
    for v = 1:length(sensorVars)
        s = std(raw.(sensorVars{v}));
        if s < 1e-4
            fprintf('    ⚠ Low variance in %s (σ = %.2e)\n', sensorVars{v}, s);
        end
    end
    
    %% STEP 2: Sort & handle duplicate timestamps, resample to 31 Hz
    fprintf('  STEP 2: Timestamp & Resampling to %.1f Hz\n', forceFs);
    
    % Sort by time
    [raw.Time, sortIdx] = sort(raw.Time);
    raw.AccX  = raw.AccX(sortIdx);
    raw.AccY  = raw.AccY(sortIdx);
    raw.AccZ  = raw.AccZ(sortIdx);
    raw.GyroX = raw.GyroX(sortIdx);
    raw.GyroY = raw.GyroY(sortIdx);
    raw.GyroZ = raw.GyroZ(sortIdx);
    
    % Merge duplicate timestamps (average values)
    tOriginal = raw.Time;
    dataMat   = table2array(raw(:,2:7));  % [AccX..GyroZ]
    [tUnique, ~, ic] = unique(tOriginal);
    if numel(tUnique) < numel(tOriginal)
        fprintf('    ⚠ Duplicate timestamps → averaging duplicates\n');
        Xuniq = zeros(numel(tUnique), 6);
        for c = 1:6
            Xuniq(:, c) = accumarray(ic, dataMat(:,c), [], @mean);
        end
        tOriginal = tUnique;
        dataMat   = Xuniq;
    end
    
    t0   = tOriginal(1);
    tEnd = tOriginal(end);
    
    % For info only
    approxFs = (numel(tOriginal) - 1) / (tEnd - t0 + eps);
    targetFs = forceFs;
    fprintf('    Approx Fs in file: %.2f Hz → Using forced Fs = %d Hz\n', approxFs, targetFs);
    
    % Uniform grid
    t_resampled = (t0 : 1/targetFs : tEnd).';
    data_resampled = interp1(tOriginal, dataMat, t_resampled, 'linear', 'extrap');
    data_resampled = fillmissing(data_resampled, 'linear');
    
    data = table(t_resampled, ...
                 data_resampled(:,1), data_resampled(:,2), data_resampled(:,3), ...
                 data_resampled(:,4), data_resampled(:,5), data_resampled(:,6), ...
                 'VariableNames', {'Time','AccX','AccY','AccZ','GyroX','GyroY','GyroZ'});
    
    fprintf('    ✓ Resampled: %d samples (Fs=%d Hz)\n', height(data), targetFs);
    
    %% STEP 3: Low-pass filter
    fprintf('  STEP 3: Butterworth Low-Pass Filtering (fc=%.1f Hz)\n', fc);
    
    Wn = fc / (targetFs/2);
    if Wn >= 1
        error('Cutoff fc=%.2f Hz too high for Fs=%.2f Hz (Wn=%.2f).', fc, targetFs, Wn);
    end
    [b, a] = butter(4, Wn, 'low');
    
    data.AccX_f  = filtfilt(b, a, data.AccX);
    data.AccY_f  = filtfilt(b, a, data.AccY);
    data.AccZ_f  = filtfilt(b, a, data.AccZ);
    data.GyroX_f = filtfilt(b, a, data.GyroX);
    data.GyroY_f = filtfilt(b, a, data.GyroY);
    data.GyroZ_f = filtfilt(b, a, data.GyroZ);
    
    fprintf('    ✓ Applied 4th-order low-pass\n');
    
    %% STEP 4: Save filtered & accumulate global stats
    fprintf('  STEP 4: Saving filtered & accumulating stats\n');
    
    outFiltered = data(:, {'Time','AccX_f','AccY_f','AccZ_f','GyroX_f','GyroY_f','GyroZ_f'});
    
    Xf = [outFiltered.AccX_f, outFiltered.AccY_f, outFiltered.AccZ_f, ...
          outFiltered.GyroX_f, outFiltered.GyroY_f, outFiltered.GyroZ_f];
    
    globalSum   = globalSum   + sum(Xf, 1, 'omitnan');
    globalSqSum = globalSqSum + sum(Xf.^2, 1, 'omitnan');
    globalN     = globalN     + size(Xf, 1);
    
    filteredPath = fullfile(filteredFolder, ['filtered_' filename]);
    writetable(outFiltered, filteredPath);
    
    fprintf('    ✓ Saved filtered file: %s\n', filteredPath);
end

%% --- Compute global z-score parameters ----------------------------------
fprintf('\n========================================\n');
fprintf('  COMPUTING GLOBAL Z-SCORE PARAMETERS\n');

if globalN == 0
    error('No samples encountered in PASS 1. Cannot compute global stats.');
end

globalMean = globalSum ./ globalN;
globalVar  = globalSqSum ./ globalN - globalMean.^2;
globalStd  = sqrt(globalVar);
globalStd(globalStd == 0) = 1;   % safety

disp('  Global means (per channel):');
disp(array2table(globalMean, 'VariableNames', sensorNames));
disp('  Global stds (per channel):');
disp(array2table(globalStd, 'VariableNames', sensorNames));

%% ========================================================================
%%  PASS 2: FILTERED -> NORMALISED + SEGMENTATION + FEATURES
%% ========================================================================

% Storage for ALL features (FD + MD) for training script
allFeatures    = [];   % [N_all x N_features]
allUserID      = [];   % numeric user ID (1..10)
allDayType     = [];   % 1 = FD, 2 = MD
allSourceFiles = {};   % filename per window

% Reference templates (FD)
refFeatures      = [];
refUserLabels    = {};
refSessionLabels = {};
refFileNames     = {};

% Testing templates (MD)
testFeatures      = [];
testUserLabels    = {};
testSessionLabels = {};
testFileNames     = {};

filteredList = dir(fullfile(filteredFolder, 'filtered_*.csv'));

fprintf('\n========================================\n');
fprintf('  PASS 2: NORMALISATION + SEGMENTATION + FEATURES\n');
fprintf('  Filtered files found: %d\n', length(filteredList));
fprintf('========================================\n');

for f = 1:length(filteredList)
    fname = filteredList(f).name;
    fprintf('\n[PASS 2: %d/%d] Processing filtered file: %s\n', f, length(filteredList), fname);
    
    fpath = fullfile(filteredFolder, fname);
    T = readtable(fpath);
    
    % Build matrix of filtered channels in the same order as sensorNames
    Xf = [T.AccX_f, T.AccY_f, T.AccZ_f, ...
          T.GyroX_f, T.GyroY_f, T.GyroZ_f];
    
    % Global z-score
    Xz = (Xf - globalMean) ./ globalStd;
    
    % Attach z-score columns
    for j = 1:numSensors
        zColName = [sensorNames{j}, '_z'];  % e.g. 'AccX_f_z'
        T.(zColName) = Xz(:, j);
    end
    
    % Save normalised file for traceability
    normName = strrep(fname, 'filtered_', 'norm_');
    normPath = fullfile(normFolder, normName);
    writetable(T, normPath);
    fprintf('  ✓ Saved normalised file: %s\n', normPath);
    
    % --------------- SEGMENTATION & FEATURE EXTRACTION -------------------
    % Use z-score channels
    sensorZVars = {'AccX_f_z','AccY_f_z','AccZ_f_z','GyroX_f_z','GyroY_f_z','GyroZ_f_z'};
    for c = 1:numel(sensorZVars)
        if ~ismember(sensorZVars{c}, T.Properties.VariableNames)
            error('Column %s not found in %s', sensorZVars{c}, normName);
        end
    end
    
    % Sampling frequency from Time
    dt = diff(T.Time);
    Fs_est = 1 / mean(dt);
    windowSize = round(windowLengthSec * Fs_est);
    stepSize   = round(windowSize * (1 - overlapFraction));
    
    if windowSize < 2 || stepSize < 1
        fprintf('  WARNING: window/step too small (Fs=%.2f). Skipping %s.\n', Fs_est, fname);
        continue;
    end
    
    fprintf('  Fs ≈ %.2f Hz | windowSize = %d | stepSize = %d\n', Fs_est, windowSize, stepSize);
    
    Xz_win = [T.AccX_f_z, T.AccY_f_z, T.AccZ_f_z, ...
              T.GyroX_f_z, T.GyroY_f_z, T.GyroZ_f_z];
    numSamples = size(Xz_win, 1);
    
    if numSamples < windowSize
        fprintf('  WARNING: file too short for one window (samples=%d). Skipping.\n', numSamples);
        continue;
    end
    
    % Parse user and session from filename (e.g. filtered_U1NW_FD.csv)
    userID_str = regexp(fname, 'U\d+', 'match', 'once');   % 'U1'
    if isempty(userID_str)
        userID_str = 'U0';
        fprintf('  WARNING: could not parse user from %s, set to U0.\n', fname);
    end
    userNum = str2double(regexprep(userID_str, 'U', ''));
    if isnan(userNum); userNum = 0; end
    
    sessionMatch = regexp(fname, '(FD|MD)', 'match', 'once');
    if isempty(sessionMatch)
        sessionID  = 'Unknown';
        dayTypeVal = 0;
    else
        sessionID = sessionMatch;  % 'FD' or 'MD'
        if strcmpi(sessionID, 'FD')
            dayTypeVal = 1;
        elseif strcmpi(sessionID, 'MD')
            dayTypeVal = 2;
        else
            dayTypeVal = 0;
        end
    end
    
    fprintf('  Parsed userID = %s (num=%d), sessionID = %s (dayType=%d)\n', ...
            userID_str, userNum, sessionID, dayTypeVal);
    
    % Window start indices
    startIdx = 1:stepSize:(numSamples - windowSize + 1);
    fprintf('  Total windows in this file: %d\n', numel(startIdx));
    
    % Loop windows
    for w = 1:numel(startIdx)
        idx = startIdx(w):(startIdx(w) + windowSize - 1);
        winData = Xz_win(idx, :);   % [windowSize x 6]
        
        featVec = extractFeaturesFromWindow(winData);  % 1 x (6*6) vector
        
        % Store in ALL features
        allFeatures    = [allFeatures; featVec];
        allUserID      = [allUserID; userNum];
        allDayType     = [allDayType; dayTypeVal];
        allSourceFiles = [allSourceFiles; {fname}];
        
        % Store in reference/testing templates
        if strcmpi(sessionID, 'FD')
            refFeatures      = [refFeatures; featVec];
            refUserLabels    = [refUserLabels;    {userID_str}];
            refSessionLabels = [refSessionLabels; {sessionID}];
            refFileNames     = [refFileNames;     {fname}];
        elseif strcmpi(sessionID, 'MD')
            testFeatures      = [testFeatures; featVec];
            testUserLabels    = [testUserLabels;    {userID_str}];
            testSessionLabels = [testSessionLabels; {sessionID}];
            testFileNames     = [testFileNames;     {fname}];
        end
    end
end

%% --- Build feature names (6 channels x 6 basic features) ----------------
channels   = {'AccX','AccY','AccZ','GyroX','GyroY','GyroZ'};
basicFeat  = {'mean','std','min','max','range','rms'};
featureNames = {};
for c = 1:numel(channels)
    for b = 1:numel(basicFeat)
        featureNames{end+1} = sprintf('%s_%s', channels{c}, basicFeat{b}); %#ok<SAGROW>
    end
end

%% --- Build and save features_all.(mat/csv) ------------------------------
if isempty(allFeatures)
    error('No windows/features were generated in PASS 2.');
end

featuresTable = array2table(allFeatures, 'VariableNames', featureNames);
featuresTable.userID     = allUserID;
featuresTable.dayType    = allDayType;      % 1 = FD, 2 = MD
featuresTable.sourceFile = allSourceFiles;

save(featuresMatPath, 'featuresTable', 'featureNames', ...
     'allFeatures', 'allUserID', 'allDayType', ...
     'windowLengthSec', 'overlapFraction');

writetable(featuresTable, featuresCsvPath);

%% --- Feature variability & uniqueness across users ----------------------
fprintf('\n========================================\n');
fprintf('  FEATURE VARIABILITY & UNIQUENESS ANALYSIS\n');
fprintf('========================================\n');

uniqueUsers = unique(allUserID);
numUsers    = numel(uniqueUsers);
numFeat     = size(allFeatures, 2);

if numUsers < 2
    warning('Not enough distinct users to analyse variability.');
else
    % Per-user mean and variance for each feature
    userMeans = zeros(numUsers, numFeat);
    userVars  = zeros(numUsers, numFeat);

    for i = 1:numUsers
        u   = uniqueUsers(i);
        idx = (allUserID == u);         % use ALL segments (FD + MD)
        Xi  = allFeatures(idx, :);

        userMeans(i, :) = mean(Xi, 1, 'omitnan');
        userVars(i, :)  = var(Xi, 0, 1, 'omitnan');
    end

    % Between-user vs within-user variance
    betweenVar = var(userMeans, 0, 1, 'omitnan');         % how much user means differ
    withinVar  = mean(userVars, 1, 'omitnan');            % average intra-user variance
    discRatio  = betweenVar ./ (withinVar + eps);         % higher = more "unique"

    % Rank features by discriminative power
    [discSorted, featIdx] = sort(discRatio, 'descend');

    topK = min(15, numFeat);
    topFeatNames = featureNames(featIdx(1:topK))';
    topBetween   = betweenVar(featIdx(1:topK))';
    topWithin    = withinVar(featIdx(1:topK))';

    variabilityTable = table( ...
        topFeatNames, ...
        discSorted(1:topK)', ...
        topBetween, ...
        topWithin, ...
        'VariableNames', {'Feature','BetweenOverWithin','BetweenVar','WithinVar'});

    fprintf('Top %d most discriminative features (between/within variance):\n', topK);
    disp(variabilityTable);

    % Save summary to CSV for the report
    varCsvPath = fullfile(processedRoot, 'feature_variability_top.csv');
    writetable(variabilityTable, varCsvPath);
    fprintf('  ✓ Saved feature variability summary to: %s\n', varCsvPath);

    % Also save full per-user mean feature vectors (nice for tables/plots)
    userID_str = arrayfun(@(u) sprintf('U%d', u), uniqueUsers, 'UniformOutput', false);
    userID_str = userID_str(:);  % force column vector

    userMeansTable = array2table(userMeans, 'VariableNames', featureNames);
    userMeansTable = addvars(userMeansTable, userID_str, ...
                             'NewVariableName', 'UserID', ...
                             'Before', 1);

    userMeansCsvPath = fullfile(processedRoot, 'feature_means_per_user.csv');
    writetable(userMeansTable, userMeansCsvPath);
    fprintf('  ✓ Saved per-user mean feature vectors to: %s\n', userMeansCsvPath);
end

%% --- Save reference templates -------------------------------------------
refUserID_cat    = categorical(refUserLabels);
refSessionID_cat = categorical(refSessionLabels);

if ~isempty(refFeatures)
    fprintf('\nSaving REFERENCE TEMPLATES (FD) to:\n  %s\n  %s\n', refMatPath, refCsvPath);
    
    save(refMatPath, 'refFeatures', 'refUserID_cat', 'refSessionID_cat', ...
         'refFileNames', 'featureNames', 'windowLengthSec', 'overlapFraction');
    
    refTable = array2table(refFeatures, 'VariableNames', featureNames);
    refTable.UserID     = refUserID_cat;
    refTable.SessionID  = refSessionID_cat;
    refTable.SourceFile = refFileNames;
    writetable(refTable, refCsvPath);
else
    fprintf('\nWARNING: No reference templates (FD) were generated.\n');
end

%% --- Save testing templates ---------------------------------------------
testUserID_cat    = categorical(testUserLabels);
testSessionID_cat = categorical(testSessionLabels);

if ~isempty(testFeatures)
    fprintf('\nSaving TESTING TEMPLATES (MD) to:\n  %s\n  %s\n', testMatPath, testCsvPath);
    
    save(testMatPath, 'testFeatures', 'testUserID_cat', 'testSessionID_cat', ...
         'testFileNames', 'featureNames', 'windowLengthSec', 'overlapFraction');
    
    testTable = array2table(testFeatures, 'VariableNames', featureNames);
    testTable.UserID     = testUserID_cat;
    testTable.SessionID  = testSessionID_cat;
    testTable.SourceFile = testFileNames;
    writetable(testTable, testCsvPath);
else
    fprintf('\nWARNING: No testing templates (MD) were generated.\n');
end

%% --- Final summary ------------------------------------------------------
fprintf('\n========================================\n');
fprintf('  SCRIPT 1 COMPLETED\n');
fprintf('  Total segments (ALL):     %d\n', size(allFeatures, 1));
fprintf('  Reference templates (FD): %d\n', size(refFeatures, 1));
fprintf('  Testing templates  (MD):  %d\n', size(testFeatures, 1));
fprintf('  Features per window:      %d\n', numel(featureNames));
fprintf('  features_all MAT: %s\n', featuresMatPath);
fprintf('  reference MAT:   %s\n', refMatPath);
fprintf('  testing MAT:     %s\n', testMatPath);
fprintf('========================================\n');

%% ===== Local function: feature extraction from one window ===============
function feat = extractFeaturesFromWindow(winData)
% winData: [N x 6] matrix for one window (6 channels)
% Returns a 1 x (6*6) feature vector:
%   [mean, std, min, max, range, rms] for each channel

    numCh = size(winData, 2);
    feat  = [];
    for c = 1:numCh
        x  = winData(:, c);
        m  = mean(x);
        s  = std(x);
        mn = min(x);
        mx = max(x);
        rg = mx - mn;
        rv = rms(x);
        feat = [feat, m, s, mn, mx, rg, rv]; %#ok<AGROW>
    end
end
