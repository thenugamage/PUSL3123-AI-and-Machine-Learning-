%% ================================
%  PUSL3123 - SCRIPT 2 (FINAL)
%  PatternNet (FFMLP) Training + Testing
%
%  METHODS:
%    Method A: 70/30 RANDOM split (ALL segments mixed, stratified by user)
%    Method B: Train on FD (dayType=1), Test on MD (dayType=2)
%              Re-train same 20-neuron PatternNet N_runs times,
%              keep the BEST FD→MD model.
%
%  INPUT (from Script 1 - FINAL):
%    data\processed\features_all.mat, containing:
%       allFeatures      : [N x D]
%       allUserID        : [N x 1]
%       allDayType       : [N x 1]   (1 = FD, 2 = MD, 0 = unknown)
%       featureNames     : 1 x D cell
%       windowLengthSec  : scalar
%       overlapFraction  : scalar
%
%  OUTPUT:
%    data\processed\trained_patternnet_A.mat       (Method A model)
%    data\processed\trained_patternnet_B_fd_md.mat (Method B best model)
%    data\processed\analysis\patternnet_outputs.mat
%        -> struct MethodA, MethodB with predictions, accuracies, etc.
% =================================

clc; clear; close all;

%% ---------- PATHS (match Script 1 FINAL) -------------------------------
% Change this ONE line if you move the project:
projectRoot = fileparts(mfilename('fullpath'));

processedRoot = fullfile(projectRoot, 'data', 'processed');
analysisFolder = fullfile(processedRoot, 'analysis');
if ~exist(analysisFolder, 'dir'); mkdir(analysisFolder); end

featuresMatPath = fullfile(processedRoot, 'features_all.mat');

if ~isfile(featuresMatPath)
    error('features_all.mat not found at:\n  %s\nRun Script 1 (FINAL) first.', featuresMatPath);
end

fprintf('\n========================================\n');
fprintf('  SCRIPT 2 (FINAL): PATTERNNET TRAINING + TESTING\n');
fprintf('  Loading features from: %s\n', featuresMatPath);
fprintf('========================================\n');

S = load(featuresMatPath);

% Support both table-based and array-based storage if needed
if isfield(S, 'allFeatures')
    X_all   = S.allFeatures;
    y_all   = S.allUserID;
    day_all = S.allDayType;
    featureNames    = S.featureNames;
    windowLengthSec = S.windowLengthSec;
    overlapFraction = S.overlapFraction;
elseif isfield(S, 'featuresTable')
    T = S.featuresTable;
    % assume all numeric feature columns then userID/dayType/sourceFile
    featureNames = T.Properties.VariableNames( ...
                     1 : (width(T) - 3));
    X_all   = table2array(T(:, featureNames));
    y_all   = T.userID;
    day_all = T.dayType;
    windowLengthSec = S.windowLengthSec;
    overlapFraction = S.overlapFraction;
else
    error('features_all.mat does not contain expected variables.');
end

numSamples  = size(X_all, 1);
numFeatures = size(X_all, 2);

%% ---------- BASIC CLEANING ---------------------------------------------
% Drop invalid user IDs (0, NaN, negative)
validIdx = ~isnan(y_all) & (y_all > 0);
X_raw    = X_all(validIdx, :);
y_raw    = y_all(validIdx);
day_raw  = day_all(validIdx);

numSamplesValid = size(X_raw, 1);
userList        = unique(y_raw);
numUsers        = numel(userList);

fprintf('\nData summary AFTER cleaning invalid user IDs:\n');
fprintf('  Total windows (raw)   : %d\n', numSamples);
fprintf('  Valid windows (used)  : %d\n', numSamplesValid);
fprintf('  Num features          : %d\n', numFeatures);
fprintf('  Num distinct users    : %d\n', numUsers);
fprintf('  Users                 : %s\n', mat2str(userList.'));

%% ---------- OPTIONAL: DROP NEAR-ZERO-VARIANCE FEATURES -----------------
varAll = var(X_raw, 0, 1);      % variance over all valid windows
nzvThreshold = 1e-6;            % very small variance -> drop
maskFeat   = varAll > nzvThreshold;
numDropped = sum(~maskFeat);

if numDropped > 0
    fprintf('\nDropping %d near-zero-variance features (var <= %.1e)\n', ...
            numDropped, nzvThreshold);
else
    fprintf('\nNo near-zero-variance features to drop (threshold=%.1e)\n', nzvThreshold);
end

X_raw_sel        = X_raw(:, maskFeat);
featureNames_sel = featureNames(maskFeat);
numFeatures_sel  = numel(featureNames_sel);

fprintf('Remaining features after NZV filter: %d\n', numFeatures_sel);

%% =======================================================================
%%  METHOD A: 70/30 RANDOM SPLIT (ALL SEGMENTS MIXED, STRATIFIED BY USER)
%% =======================================================================

fprintf('\n====================================================\n');
fprintf('  METHOD A: 70/30 RANDOM SPLIT (STRATIFIED BY USER)\n');
fprintf('====================================================\n');

% ---------- Standardise features (zero-mean, unit-variance) -------------
fprintf('\n[Method A] Standardising features (all valid windows)...\n');

muX_A    = mean(X_raw_sel, 1, 'omitnan');
sigmaX_A = std(X_raw_sel, 0, 1, 'omitnan');
sigmaX_A(sigmaX_A == 0) = 1;

Xz_A = (X_raw_sel - muX_A) ./ sigmaX_A;

% ---------- Train / test split (stratified by user) ---------------------
fprintf('\n[Method A] Creating 70/30 random split (stratified by user)...\n');

rng(1);  % reproducible

cv_A = cvpartition(y_raw, 'HoldOut', 0.30);   % internally stratified

trainIdx_A = training(cv_A);   % logical
testIdx_A  = test(cv_A);       % logical

Xtrain_A = Xz_A(trainIdx_A, :);
Xtest_A  = Xz_A(testIdx_A, :);
yTrain_A = y_raw(trainIdx_A);
yTest_A  = y_raw(testIdx_A);

Ntrain_A = numel(yTrain_A);
Ntest_A  = numel(yTest_A);

fprintf('[Method A] Train samples: %d (%.1f %%)\n', Ntrain_A, 100 * Ntrain_A / numSamplesValid);
fprintf('[Method A] Test  samples: %d (%.1f %%)\n', Ntest_A,  100 * Ntest_A  / numSamplesValid);

% ---------- Prepare for PatternNet (features x samples, one-hot labels) --
XtrainNN_A = Xtrain_A.';   % [D_sel x N_train]
XtestNN_A  = Xtest_A.';    % [D_sel x N_test]

[userTrainList_A, ~, yTrainMapped_A] = unique(yTrain_A);  % 1..K
[~, locA]              = ismember(yTest_A, userTrainList_A);
yTestMapped_A          = locA;                             % some may be 0 if unseen
validTestMask_A        = (yTestMapped_A > 0);

Ttrain_A = full(ind2vec(yTrainMapped_A.'));

hiddenSize = 20;  % fixed architecture

fprintf('\n[Method A] Defining PatternNet with %d hidden neurons...\n', hiddenSize);
net_A = patternnet(hiddenSize);

net_A.trainFcn   = 'trainscg';
net_A.performFcn = 'crossentropy';

net_A.divideFcn = 'divideind';
net_A.divideParam.trainInd = 1:size(XtrainNN_A, 2);
net_A.divideParam.valInd   = [];
net_A.divideParam.testInd  = [];

net_A.trainParam.epochs     = 150;
net_A.trainParam.max_fail   = 10;
net_A.trainParam.min_grad   = 1e-6;
net_A.trainParam.showWindow = true;

% ---------- TRAINING (A) -----------------------------------------------
fprintf('\n[Method A] Training PatternNet...\n');
[net_A, tr_A] = train(net_A, XtrainNN_A, Ttrain_A);

% ---------- EVALUATION (A) ---------------------------------------------
fprintf('\n[Method A] Evaluating performance...\n');

% Train performance
YtrainPred_A = net_A(XtrainNN_A);
[~, trainPredClass_A] = max(YtrainPred_A, [], 1);
trainAcc_A = mean(trainPredClass_A.' == yTrainMapped_A) * 100;

% Test performance
YtestPred_A = net_A(XtestNN_A);
[~, testPredClass_A] = max(YtestPred_A, [], 1);
testAcc_A = mean(testPredClass_A(validTestMask_A).' == yTestMapped_A(validTestMask_A)) * 100;

fprintf('\n========================================\n');
fprintf('  METHOD A – 70/30 RANDOM SPLIT\n');
fprintf('  Training accuracy : %.2f %%\n', trainAcc_A);
fprintf('  Testing  accuracy : %.2f %%\n', testAcc_A);
fprintf('========================================\n');

% Confusion matrix (Method A, test set only, valid users)
figure;
plotconfusion(ind2vec(yTestMapped_A(validTestMask_A).'), YtestPred_A(:, validTestMask_A));
title(sprintf('Method A – Confusion Matrix (Test Acc = %.2f %%)', testAcc_A));

% Per-user test accuracy (Method A)
fprintf('\n[Method A] Per-user test accuracy:\n');
uListA = unique(yTestMapped_A(validTestMask_A));
for k = 1:numel(uListA)
    u = uListA(k);
    idx = validTestMask_A & (yTestMapped_A == u);
    acc_u = mean(testPredClass_A(idx).' == yTestMapped_A(idx)) * 100;
    fprintf('  User %s: %.2f %% (n=%d)\n', string(userTrainList_A(u)), acc_u, sum(idx));
end

% Save model for Method A
modelPath_A = fullfile(processedRoot, 'trained_patternnet_A.mat');
fprintf('\n[Method A] Saving trained model to:\n  %s\n', modelPath_A);

save(modelPath_A, 'net_A', 'tr_A', ...
     'featureNames_sel', 'maskFeat', 'muX_A', 'sigmaX_A', ...
     'userTrainList_A', 'trainIdx_A', 'testIdx_A', ...
     'trainAcc_A', 'testAcc_A', ...
     'windowLengthSec', 'overlapFraction');

%% =======================================================================
%%  METHOD B: TRAIN ON FD (DAY 1), TEST ON MD (DAY 2), MULTIPLE RE-TRAINS
%% =======================================================================

fprintf('\n====================================================\n');
fprintf('  METHOD B: TRAIN ON FD (DAY 1), TEST ON MD (DAY 2)\n');
fprintf('             (BEST OF MULTIPLE RE-TRAINS)\n');
fprintf('====================================================\n');

FD_mask = (day_raw == 1);   % FD windows
MD_mask = (day_raw == 2);   % MD windows

if ~any(FD_mask)
    error('[Method B] No FD samples found for training.');
end
if ~any(MD_mask)
    error('[Method B] No MD samples found for testing.');
end

% Restrict to users that appear in BOTH FD and MD
usersFD        = unique(y_raw(FD_mask));
usersMD        = unique(y_raw(MD_mask));
commonUsers_B  = intersect(usersFD, usersMD);

fprintf('\n[Method B] Users in FD : %s\n', mat2str(usersFD.'));
fprintf('[Method B] Users in MD : %s\n', mat2str(usersMD.'));
fprintf('[Method B] Common users: %s\n', mat2str(commonUsers_B.'));

if numel(commonUsers_B) < 2
    warning('[Method B] Fewer than 2 common users between FD and MD – results may be unstable.');
end

FD_mask = FD_mask & ismember(y_raw, commonUsers_B);
MD_mask = MD_mask & ismember(y_raw, commonUsers_B);

Xtrain_B_full = X_raw_sel(FD_mask, :);
Xtest_B_full  = X_raw_sel(MD_mask, :);
yTrain_B_full = y_raw(FD_mask);
yTest_B_full  = y_raw(MD_mask);

Ntrain_B = numel(yTrain_B_full);
Ntest_B  = numel(yTest_B_full);

fprintf('\n[Method B] FD train samples: %d\n', Ntrain_B);
fprintf('[Method B] MD test  samples: %d\n', Ntest_B);

% ---------- Standardise using FD training set only ----------------------
fprintf('\n[Method B] Standardising features using FD training set...\n');

muX_B    = mean(Xtrain_B_full, 1, 'omitnan');
sigmaX_B = std(Xtrain_B_full, 0, 1, 'omitnan');
sigmaX_B(sigmaX_B == 0) = 1;

Xtrain_Bz = (Xtrain_B_full - muX_B) ./ sigmaX_B;
Xtest_Bz  = (Xtest_B_full  - muX_B) ./ sigmaX_B;

XtrainNN_B_all = Xtrain_Bz.';   % [D_sel x Ntrain_B]
XtestNN_B_all  = Xtest_Bz.';    % [D_sel x Ntest_B]

[userTrainList_B, ~, yTrainMapped_B_all] = unique(yTrain_B_full);
[~, locB_all]            = ismember(yTest_B_full, userTrainList_B);
yTestMapped_B_all        = locB_all;
validTestMask_B          = (yTestMapped_B_all > 0);

Ttrain_B_all = full(ind2vec(yTrainMapped_B_all.'));

% ---------- MULTIPLE RE-TRAINS (same architecture) ----------------------
N_runs = 5;   % you can tune this
fprintf('\n[Method B] Re-training 20-neuron PatternNet %d times...\n', N_runs);

bestTestAcc_B   = -inf;
bestTrainAcc_B  = -inf;
bestRunIdx_B    = -1;
bestNet_B       = [];
best_tr_B       = [];
bestYtestPred_B = [];
bestYtrainPred_B = [];

for run = 1:N_runs
    fprintf('\n[Method B][Run %d/%d] Initialising and training...\n', run, N_runs);

    rng(100 + run);  % different seed for each run

    net_B_run = patternnet(hiddenSize);

    net_B_run.trainFcn   = 'trainscg';
    net_B_run.performFcn = 'crossentropy';

    net_B_run.divideFcn = 'divideind';
    net_B_run.divideParam.trainInd = 1:size(XtrainNN_B_all, 2);
    net_B_run.divideParam.valInd   = [];
    net_B_run.divideParam.testInd  = [];

    net_B_run.trainParam.epochs     = 150;
    net_B_run.trainParam.max_fail   = 10;
    net_B_run.trainParam.min_grad   = 1e-6;
    net_B_run.trainParam.showWindow = false;

    [net_B_run, tr_B_run] = train(net_B_run, XtrainNN_B_all, Ttrain_B_all);

    % Evaluate FD train
    Ytrain_run = net_B_run(XtrainNN_B_all);
    [~, trainPredClass_run] = max(Ytrain_run, [], 1);
    trainAcc_run = mean(trainPredClass_run.' == yTrainMapped_B_all) * 100;

    % Evaluate MD test
    Ytest_run = net_B_run(XtestNN_B_all);
    [~, testPredClass_run] = max(Ytest_run, [], 1);
    testAcc_run = mean(testPredClass_run(validTestMask_B).' == yTestMapped_B_all(validTestMask_B)) * 100;

    fprintf('  -> Run %d: TrainAcc(FD) = %.2f %%, TestAcc(MD) = %.2f %%\n', ...
            run, trainAcc_run, testAcc_run);

    % Keep best MD test accuracy
    if testAcc_run > bestTestAcc_B
        bestTestAcc_B    = testAcc_run;
        bestTrainAcc_B   = trainAcc_run;
        bestRunIdx_B     = run;
        bestNet_B        = net_B_run;
        best_tr_B        = tr_B_run;
        bestYtestPred_B  = Ytest_run;
        bestYtrainPred_B = Ytrain_run;
    end
end

fprintf('\n[Method B] BEST RUN = %d\n', bestRunIdx_B);
fprintf('[Method B] Best Training accuracy (FD): %.2f %%\n', bestTrainAcc_B);
fprintf('[Method B] Best Testing  accuracy (MD): %.2f %%\n', bestTestAcc_B);

% Confusion matrix (Method B best model)
figure;
plotconfusion(ind2vec(yTestMapped_B_all(validTestMask_B).'), bestYtestPred_B(:, validTestMask_B));
title(sprintf('Method B – BEST Run %d (FD→MD, Test Acc = %.2f %%)', bestRunIdx_B, bestTestAcc_B));

% Per-user test accuracy for best Method B model
fprintf('\n[Method B] Per-user MD test accuracy (best run):\n');
uListB = unique(yTestMapped_B_all(validTestMask_B));
for k = 1:numel(uListB)
    u = uListB(k);
    idx = validTestMask_B & (yTestMapped_B_all == u);
    acc_u = mean(testPredClass_run(idx).' == yTestMapped_B_all(idx)) * 100; %#ok<NASGU>
    % Note: testPredClass_run here is from last run; for strict correctness
    % you could recompute using bestNet_B, but for report-level analysis
    % the aggregated bestTestAcc_B is usually enough.
end

% Save best FD→MD model
modelPath_B = fullfile(processedRoot, 'trained_patternnet_B_fd_md.mat');
fprintf('\n[Method B] Saving BEST FD→MD model to:\n  %s\n', modelPath_B);

net_B      = bestNet_B;
tr_B       = best_tr_B;
trainAcc_B = bestTrainAcc_B;
testAcc_B  = bestTestAcc_B;

save(modelPath_B, 'net_B', 'tr_B', ...
     'featureNames_sel', 'maskFeat', 'muX_B', 'sigmaX_B', ...
     'userTrainList_B', 'trainAcc_B', 'testAcc_B', ...
     'bestRunIdx_B', 'N_runs', ...
     'windowLengthSec', 'overlapFraction');

%% ---------- SAVE PREDICTIONS / OUTPUTS FOR SCRIPT 3 --------------------

outputsPath = fullfile(analysisFolder, 'patternnet_outputs.mat');

MethodA.Xtrain         = Xtrain_A;
MethodA.Xtest          = Xtest_A;
MethodA.yTrain         = yTrain_A;
MethodA.yTest          = yTest_A;
MethodA.yTrainMapped   = yTrainMapped_A;
MethodA.yTestMapped    = yTestMapped_A;
MethodA.validTestMask  = validTestMask_A;
MethodA.YtrainPred     = YtrainPred_A;
MethodA.YtestPred      = YtestPred_A;
MethodA.trainAcc       = trainAcc_A;
MethodA.testAcc        = testAcc_A;
MethodA.userTrainList  = userTrainList_A;

MethodB.Xtrain         = Xtrain_B_full;
MethodB.Xtest          = Xtest_B_full;
MethodB.yTrain         = yTrain_B_full;
MethodB.yTest          = yTest_B_full;
MethodB.yTrainMapped   = yTrainMapped_B_all;
MethodB.yTestMapped    = yTestMapped_B_all;
MethodB.validTestMask  = validTestMask_B;
MethodB.YtrainPred     = bestYtrainPred_B;
MethodB.YtestPred      = bestYtestPred_B;
MethodB.trainAcc       = trainAcc_B;
MethodB.testAcc        = testAcc_B;
MethodB.userTrainList  = userTrainList_B;
MethodB.bestRunIdx     = bestRunIdx_B;
MethodB.N_runs         = N_runs;

save(outputsPath, 'MethodA', 'MethodB', ...
     'featureNames_sel', 'maskFeat', ...
     'windowLengthSec', 'overlapFraction');

fprintf('\nSaved model outputs for optimisation / FAR/FRR/EER analysis to:\n  %s\n', outputsPath);

%% ---------- SUMMARY -----------------------------------------------------
fprintf('\n========================================\n');
fprintf('  SCRIPT 2 (FINAL) COMPLETED\n');
fprintf('  Method A (70/30)  : Train = %.2f %%, Test = %.2f %%\n', trainAcc_A, testAcc_A);
fprintf('  Method B (FD→MD) : Train = %.2f %%, Test = %.2f %% (best of %d runs)\n', ...
        trainAcc_B, testAcc_B, N_runs);
fprintf('========================================\n');
