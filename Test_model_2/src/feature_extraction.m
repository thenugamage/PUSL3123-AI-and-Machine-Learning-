%% Script 2 — feature_extraction.m  (Step 06 ready)
% Adds a toggle to include orientation-invariant magnitudes (AccMag, GyroMag).
% Reads data/interim/preprocessed_data.csv; writes data/processed/features.csv

%% ---------- Configuration ----------
thisFile = mfilename('fullpath'); srcDir = fileparts(thisFile); projectRoot = fileparts(srcDir);
cfgPath = fullfile(projectRoot,'config','config.m');
if exist(cfgPath,'file'), run(cfgPath); else, fs=31; win_s=2; hop_s=1; end

% Toggle can be injected by run_step06.m
if exist('add_magnitude_features','var'), addMag = logical(add_magnitude_features);
else, addMag = false; end   % baseline = false; Step06 may set true

winL = max(1, round(win_s * fs));
hopL = max(1, round(hop_s * fs));

rawInterim = fullfile(projectRoot,'data','interim','preprocessed_data.csv');
outDir     = fullfile(projectRoot,'data','processed'); if ~exist(outDir,'dir'), mkdir(outDir); end
outCSV     = fullfile(outDir,'features.csv');

sensorCols = {'AccX','AccY','AccZ','GyroX','GyroY','GyroZ'};
metaKeep   = {'UserID','Session','SampleIdx','TimeSec'};

%% ---------- Load ----------
if ~isfile(rawInterim), error('Could not find %s. Run Script 1 first.', rawInterim); end
T = readtable(rawInterim);
for c = 1:numel(sensorCols), assert(ismember(sensorCols{c}, T.Properties.VariableNames)); end
assert(all(ismember(metaKeep, T.Properties.VariableNames)));
if ~iscategorical(T.Session), T.Session = categorical(T.Session); end
T = sortrows(T, {'UserID','Session','SampleIdx'});

%% ---------- Feature names ----------
baseStats = {'mean','std','var','min','max','range','energy','entropy'};
corrNames = {'AccCorr_XY','AccCorr_XZ','AccCorr_YZ','GyroCorr_XY','GyroCorr_XZ','GyroCorr_YZ'};

if addMag
    axesForStats = [sensorCols, {'AccMag','GyroMag'}];
else
    axesForStats = sensorCols;
end
featNames = makeFeatureNames(axesForStats, baseStats, corrNames);
featDim   = numel(featNames);

%% ---------- Count windows ----------
users = unique(T.UserID); totalWins = 0;
for u = users'
    sessList = unique(T.Session(T.UserID==u));
    for s = 1:numel(sessList)
        rows = find(T.UserID==u & T.Session==sessList(s));
        N = numel(rows);
        totalWins = totalWins + max(0, floor((N - winL)/hopL) + 1);
    end
end
if totalWins==0, error('No windows to extract.'); end

% Preallocate
X  = zeros(totalWins, featDim);
Y  = zeros(totalWins, 1);
SS = strings(totalWins,1);
WinStartIdx = zeros(totalWins,1); WinEndIdx = zeros(totalWins,1); WinCenterT = zeros(totalWins,1);

%% ---------- Window & features ----------
w = 0;
for u = users'
    sessList = unique(T.Session(T.UserID==u));
    for s = 1:numel(sessList)
        sess = sessList(s);
        rows = find(T.UserID==u & T.Session==sess); TT = T(rows,:);
        N = height(TT); if N < winL, continue; end
        starts = 1:hopL:(N - winL + 1);

        for k = 1:numel(starts)
            a = starts(k); b = a + winL - 1;
            W = TT{a:b, sensorCols};  % winL x 6 (already z-scored)

            % --- Optionally add magnitudes (AccMag, GyroMag) ---
            if addMag
                AccMag = sqrt(sum(W(:,1:3).^2,2));
                GyroMag= sqrt(sum(W(:,4:6).^2,2));
                Wext = [W, AccMag, GyroMag];   % 8 signals
            else
                Wext = W;                       % 6 signals
            end

            % --- Per-signal stats (mean, std, var, min, max, range, energy, entropy) ---
            mu   = mean(Wext,1);
            sd   = std(Wext,0,1);
            vr   = var(Wext,0,1);
            mn   = min(Wext,[],1);
            mx   = max(Wext,[],1);
            rngV = mx - mn;
            eng  = sum(Wext.^2,1) / size(Wext,1);
            ent  = zeros(1, size(Wext,2));
            for j = 1:size(Wext,2), ent(j) = shannon_entropy(Wext(:,j), 20); end

            % --- Correlations within Acc and within Gyro (use original W) ---
            acc = W(:,1:3); gyr = W(:,4:6);
            cAcc = [safe_corr(acc(:,1),acc(:,2)), safe_corr(acc(:,1),acc(:,3)), safe_corr(acc(:,2),acc(:,3))];
            cGyr = [safe_corr(gyr(:,1),gyr(:,2)), safe_corr(gyr(:,1),gyr(:,3)), safe_corr(gyr(:,2),gyr(:,3))];

            featRow = [mu, sd, vr, mn, mx, rngV, eng, ent, cAcc, cGyr];
            if numel(featRow) ~= featDim, error('Feature length %d != expected %d', numel(featRow), featDim); end

            w = w + 1;
            X(w,:) = featRow; Y(w) = u; SS(w)=string(sess);
            WinStartIdx(w)=TT.SampleIdx(a); WinEndIdx(w)=TT.SampleIdx(b);
            WinCenterT(w)=mean(TT.TimeSec([a b]));
        end
    end
end

% Trim
X=X(1:w,:); Y=Y(1:w); SS=SS(1:w);
WinStartIdx=WinStartIdx(1:w); WinEndIdx=WinEndIdx(1:w); WinCenterT=WinCenterT(1:w);

%% ---------- Save ----------
FeatTbl = array2table(X, 'VariableNames', featNames);
FeatTbl.UserID=Y; FeatTbl.Session=categorical(SS);
FeatTbl.WinStartIdx=WinStartIdx; FeatTbl.WinEndIdx=WinEndIdx; FeatTbl.CenterTimeSec=WinCenterT;

writetable(FeatTbl, outCSV);
fprintf('Saved features: %s | add_magnitude_features=%d\n', outCSV, addMag);

%% ---------- Helpers ----------
function names = makeFeatureNames(sigCols, baseStats, corrNames)
    names = strings(0,1);
    for i = 1:numel(sigCols)
        for b = 1:numel(baseStats)
            names(end+1,1) = sprintf('%s_%s', sigCols{i}, baseStats{b}); %#ok<AGROW>
        end
    end
    names = [names; string(corrNames(:))];
    names = cellstr(names)';  % 1-by-N for table headers
end
function H = shannon_entropy(x, nbins)
    if nargin<2, nbins=20; end
    if std(x)==0, H = 0; return; end
    p = histcounts(x, nbins, 'Normalization','probability'); p = p + eps;
    H = -sum(p .* log2(p));
end
function r = safe_corr(a,b)
    if numel(a)<2 || numel(b)<2 || std(a)==0 || std(b)==0, r=0; return; end
    C = corrcoef(a,b); r = C(1,2);
end
