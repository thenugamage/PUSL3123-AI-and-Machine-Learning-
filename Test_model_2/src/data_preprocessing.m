%% Script 1 — data_preprocessing.m
% Purpose: Import, merge (FD+MD), filter, timestamp, normalize, and save.
% Output: data/interim/preprocessed_data.csv + config/normalizer.mat
% Notes:
% - Filtering: zero-phase 4th-order Butterworth bandpass (0.2–12 Hz) for gait.
% - Normalization: z-score over all rows (report this choice).
% - Optional: set doSegmentInScript1=true to tag samples with window IDs (default off).
% Coursework alignment: Data Collection & Preprocessing; groundwork for supervised ML. 

%% ---------- Configuration ----------
% Try to load your project config; fall back to sensible defaults
thisFile = mfilename('fullpath'); srcDir = fileparts(thisFile); projectRoot = fileparts(srcDir);
cfgPath = fullfile(projectRoot,'config','config.m');
if exist(cfgPath,'file'), run(cfgPath); else, fs=31; win_s=2; hop_s=1; end

% You can toggle segmentation here; normally we segment in Script 2.
doSegmentInScript1 = false;   % <-- set true only if you want window IDs now

% Filter band (Hz) tuned for walking
lowcut = 0.2; highcut = 12;

% Users/sessions expected
userIDs  = 1:10;
sessions = {'FD','MD'};

%% ---------- Paths ----------
rawDir     = fullfile(projectRoot,'data','raw');
interimDir = fullfile(projectRoot,'data','interim');
cfgDir     = fullfile(projectRoot,'config');
if ~exist(interimDir,'dir'), mkdir(interimDir); end
if ~exist(cfgDir,'dir'), mkdir(cfgDir); end

%% ---------- Import, filter, timestamp, merge ----------
ALL = table();
sensorCols = {'AccX','AccY','AccZ','GyroX','GyroY','GyroZ'};

for u = userIDs
    for s = 1:numel(sessions)
        sess = sessions{s};

        % Support both U1NW_FD.csv and U01NW_FD.csv naming
        fp = fullfile(rawDir, sprintf('U%dNW_%s.csv',  u, sess));
        if ~isfile(fp)
            fp = fullfile(rawDir, sprintf('U%02dNW_%s.csv', u, sess));
        end
        if ~isfile(fp)
            warning('Missing file for user %d, session %s', u, sess);
            continue
        end

        % Load with robust headers, clean missing/outliers
        T = readSensorCSV(fp, sensorCols);

        % Time base from nominal fs (31 Hz)
        N = height(T);
        T.SampleIdx = (0:N-1).';
        T.TimeSec   = T.SampleIdx / fs;

        % Zero-phase bandpass filter (0.2–12 Hz)
        T = applyButterBandpass(T, fs, lowcut, highcut, sensorCols);

        % Metadata
        T.UserID  = repmat(u, N, 1);
        T.Session = repmat(string(sess), N, 1);

        ALL = [ALL; T]; %#ok<AGROW>
    end
end

% Keep just the sensor + meta columns (in a consistent order)
metaCols = {'UserID','Session','SampleIdx','TimeSec'};
ALL = ALL(:,[sensorCols, metaCols]);
ALL.Session = categorical(ALL.Session);

%% ---------- (Optional) assign window IDs now ----------
if doSegmentInScript1
    winL = round(win_s*fs); 
    hopL = round(hop_s*fs);
    ALL.WindowID = assignWindowIDs(ALL, winL, hopL);
end

%% ---------- Normalize (z-score) ----------
% Compute global mean/std across all rows (document this in the report).
X = ALL{:,sensorCols};
mu    = mean(X,1,'omitnan');
sigma = std(X,0,1,'omitnan');  sigma(sigma==0) = eps;
ALL{:,sensorCols} = (X - mu) ./ sigma;

% Save the scaler so Script 2/3 use the same normalization
normalizer.mu = mu;
normalizer.sigma = sigma;
normalizer.columns = sensorCols;
save(fullfile(cfgDir,'normalizer.mat'),'normalizer');

%% ---------- Export ----------
outCSV = fullfile(interimDir,'preprocessed_data.csv');
writetable(ALL, outCSV);
fprintf('\nSaved: %s\n', outCSV);
fprintf('Rows: %d | Users present: %s\n', height(ALL), mat2str(unique(ALL.UserID)'));

%% ================= Helper functions =================
function T = readSensorCSV(fp, sensorCols)
    % Robust CSV reader: keeps first 6 columns, renames, fills, de-spikes
    opts = detectImportOptions(fp);
    T = readtable(fp, opts);
    if width(T) < numel(sensorCols)
        error('Expected ≥ %d columns in %s', numel(sensorCols), fp);
    end
    T = T(:,1:numel(sensorCols));
    T.Properties.VariableNames(1:numel(sensorCols)) = sensorCols;

    % Fill missing + de-spike
    for k = 1:numel(sensorCols)
        v = T.(sensorCols{k});
        v = fillmissing(v,'linear','EndValues','nearest');
        idx = isoutlier(v,'movmedian',21);
        if any(idx), v(idx) = median(v(~idx),'omitnan'); end
        T.(sensorCols{k}) = v;
    end
end

function T = applyButterBandpass(T, fs, lowHz, highHz, cols)
    % Zero-phase bandpass using filtfilt
    Wn = [lowHz, highHz] / (fs/2);
    Wn(1) = max(Wn(1), 1e-3);
    Wn(2) = min(Wn(2), 0.999);
    [b,a] = butter(4, Wn, 'bandpass');
    for k = 1:numel(cols)
        x = double(T.(cols{k}));
        T.(cols{k}) = filtfilt(b,a,x);
    end
end

function windowID = assignWindowIDs(ALL, winL, hopL)
    % Integer window IDs per (UserID, Session). No features here—just tags.
    windowID = zeros(height(ALL),1);
    users = unique(ALL.UserID);
    id = 0;
    for u = users'
        sessList = categories(ALL.Session(ALL.UserID==u));
        for j = 1:numel(sessList)
            s = sessList{j};
            rows = find(ALL.UserID==u & ALL.Session==s);
            idxs = ALL.SampleIdx(rows);
            last = idxs(end);
            starts = 0:hopL:(last - winL);
            for k = 1:numel(starts)
                id = id + 1;
                inWin = idxs >= starts(k) & idxs < (starts(k)+winL);
                windowID(rows(inWin)) = id;
            end
        end
    end
end
