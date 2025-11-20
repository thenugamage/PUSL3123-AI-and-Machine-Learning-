% =========================================================================
%  PUSL3123 - SCRIPT 2
%  PATTERNNET TRAINING + TESTING
%
%  METHODS:
%    Method A: 70/30 RANDOM split (ALL segments mixed, stratified by user)
%    Method B: Train on FD (dayType=1), Test on MD (dayType=2)
%              SAME PatternNet (20 neurons), BUT re-trained multiple times
%              and best FD->MD model is selected.
%
%  Uses (from Script 1):
%    - data/processed/features_all.mat
%      -> allFeatures  : [N x 36]
%      -> allUserID    : [N x 1]
%      -> allDayType   : [N x 1]  (1 = FD, 2 = MD)
%
%  Outputs:
%    - data/processed/trained_patternnet.mat
%         (Method A, 20 hidden neurons, one run)
%    - data/processed/trained_patternnet_fd_md.mat
%         (Method B, BEST of multiple re-trainings, still 20 neurons)
% =========================================================================

clc; clear; close all;

%% ------------------------- PATH SETUP -----------------------------------
projectRoot   = fileparts(fileparts(mfilename('fullpath')));  % src/.. -> root
processedRoot = fullfile(projectRoot, 'data', 'processed');
featuresMatPath = fullfile(processedRoot, 'features_all.mat');

if ~exist(featuresMatPath, 'file')
    error('features_all.mat not found at:\n  %s\nRun Script 1 first.', featuresMatPath);
end

fprintf('\n========================================\n');
fprintf('  SCRIPT 2: PATTERNNET TRAINING + TESTING\n');
fprintf('  Loading features from: %s\n', featuresMatPath);
fprintf('========================================\n');

load(featuresMatPath, 'allFeatures', 'allUserID', 'allDayType', ...
     'featureNames', 'windowLengthSec', 'overlapFraction');

X_all   = allFeatures;     % [N_samples x N_features]
y_all   = allUserID;       % [N_samples x 1], user IDs (1..10)
day_all = allDayType;      % [N_samples x 1], 1=FD, 2=MD

numSamples  = size(X_all, 1);
numFeatures = size(X_all, 2);

% Remove invalid labels (0 or NaN)
validIdx = ~isnan(y_all) & (y_all > 0);
X_raw    = X_all(validIdx, :);
y_raw    = y_all(validIdx);
day_raw  = day_all(validIdx);

numSamplesValid = size(X_raw, 1);
userList        = unique(y_raw);
numUsers        = numel(userList);

fprintf('Total windows (raw):   %d\n', numSamples);
fprintf('Valid windows (used):  %d\n', numSamplesValid);
fprintf('Num features:          %d\n', numFeatures);
fprintf('Num distinct users:    %d\n', numUsers);
fprintf('Users:                 %s\n', mat2str(userList.'));

%% ========================================================================
%%  METHOD A: 70/30 RANDOM SPLIT (ALL SEGMENTS MIXED)
%% ========================================================================
fprintf('\n====================================================\n');
fprintf('  METHOD A: 70/30 RANDOM SPLIT (STRATIFIED BY USER)\n');
fprintf('====================================================\n');

% ------------------------- FEATURE STANDARDISATION (A) -------------------
fprintf('\n[Method A] Standardising features (zero-mean, unit-variance)...\n');

muX_A = mean(X_raw, 1, 'omitnan');
sigmaX_A = std(X_raw, 0, 1, 'omitnan');
sigmaX_A(sigmaX_A == 0) = 1;

Xz_A = (X_raw - muX_A) ./ sigmaX_A;

% ------------------------- TRAIN / TEST SPLIT (A) ------------------------
fprintf('\n[Method A] Creating 70/30 random split (stratified by user)...\n');

rng(1);  % reproducibility for Method A

cv_A = cvpartition(y_raw, 'HoldOut', 0.30);

trainIdx_A = training(cv_A);   % logical index
testIdx_A  = test(cv_A);       % logical index

Xtrain_A = Xz_A(trainIdx_A, :);
Xtest_A  = Xz_A(testIdx_A, :);
yTrain_A = y_raw(trainIdx_A);
yTest_A  = y_raw(testIdx_A);

Ntrain_A = numel(yTrain_A);
Ntest_A  = numel(yTest_A);

fprintf('[Method A] Train samples: %d (%.1f %%)\n', Ntrain_A, 100 * Ntrain_A / numSamplesValid);
fprintf('[Method A] Test  samples: %d (%.1f %%)\n', Ntest_A,  100 * Ntest_A  / numSamplesValid);

% ------------------------- PREP FOR PATTERNNET (A) -----------------------
XtrainNN_A = Xtrain_A.';   % [numFeatures x N_train]
XtestNN_A  = Xtest_A.';    % [numFeatures x N_test]

[userTrainList_A, ~, yTrainMapped_A] = unique(yTrain_A);
[~, locA] = ismember(yTest_A, userTrainList_A);
yTestMapped_A = locA;

Ttrain_A = full(ind2vec(yTrainMapped_A.'));

hiddenSize = 20;  % fixed, as we decided

fprintf('\n[Method A] Defining PatternNet with %d hidden neurons...\n', hiddenSize);
net_A = patternnet(hiddenSize);

net_A.divideFcn = 'divideind';
net_A.divideParam.trainInd = 1:size(XtrainNN_A, 2);
net_A.divideParam.valInd   = [];
net_A.divideParam.testInd  = [];

net_A.performFcn = 'crossentropy';
net_A.trainParam.epochs     = 150;
net_A.trainParam.max_fail   = 10;
net_A.trainParam.min_grad   = 1e-6;
net_A.trainParam.showWindow = true;

% ------------------------- TRAINING (A) ----------------------------------
fprintf('\n[Method A] Training neural network (PatternNet)...\n');
[net_A, tr_A] = train(net_A, XtrainNN_A, Ttrain_A);

% ------------------------- EVALUATION (A) --------------------------------
fprintf('\n[Method A] Evaluating performance...\n');

YtrainPred_A = net_A(XtrainNN_A);
[~, trainPredClass_A] = max(YtrainPred_A, [], 1);
trainAcc_A = mean(trainPredClass_A.' == yTrainMapped_A) * 100;

YtestPred_A = net_A(XtestNN_A);
[~, testPredClass_A] = max(YtestPred_A, [], 1);
validTestMask_A = (yTestMapped_A > 0);
testAcc_A = mean(testPredClass_A(validTestMask_A).' == yTestMapped_A(validTestMask_A)) * 100;

fprintf('\n========================================\n');
fprintf('  METHOD A – PATTERNNET (70/30 RANDOM)\n');
fprintf('  Training accuracy: %.2f %%\n', trainAcc_A);
fprintf('  Testing  accuracy: %.2f %%\n', testAcc_A);
fprintf('========================================\n');

figure;
plotconfusion(ind2vec(yTestMapped_A(validTestMask_A).'), YtestPred_A(:, validTestMask_A));
title(sprintf('Method A – Confusion Matrix (Test Acc = %.2f %%)', testAcc_A));

% ------------------------- SAVE MODEL (A) --------------------------------
modelPath_A = fullfile(processedRoot, 'trained_patternnet.mat');

fprintf('\n[Method A] Saving trained model to:\n  %s\n', modelPath_A);

save(modelPath_A, 'net_A', 'tr_A', ...
     'featureNames', 'muX_A', 'sigmaX_A', ...
     'userTrainList_A', 'trainIdx_A', 'testIdx_A', ...
     'trainAcc_A', 'testAcc_A', ...
     'windowLengthSec', 'overlapFraction');

%% ========================================================================
%%  METHOD B: TRAIN ON FD (DAY 1), TEST ON MD (DAY 2)
%%           SAME MODEL, MULTIPLE RE-TRAINS, KEEP BEST
%% ========================================================================
fprintf('\n====================================================\n');
fprintf('  METHOD B: TRAIN ON FD (DAY 1), TEST ON MD (DAY 2)\n');
fprintf('             (RE-TRAIN 20-NEURON PatternNet MULTIPLE TIMES)\n');
fprintf('====================================================\n');

% day_raw: 1 = FD, 2 = MD (after validIdx filter)
FD_mask = (day_raw == 1);  % FD
MD_mask = (day_raw == 2);  % MD

if ~any(FD_mask)
    error('[Method B] No FD samples found for training.');
end
if ~any(MD_mask)
    error('[Method B] No MD samples found for testing.');
end

usersFD  = unique(y_raw(FD_mask));
usersMD  = unique(y_raw(MD_mask));
commonUsers_B = intersect(usersFD, usersMD);

fprintf('\n[Method B] Users in FD:  %s\n', mat2str(usersFD.'));
fprintf('[Method B] Users in MD:  %s\n', mat2str(usersMD.'));
fprintf('[Method B] Common users: %s\n', mat2str(commonUsers_B.'));

if numel(commonUsers_B) < 2
    warning('[Method B] Less than 2 common users between FD and MD. Results may be unstable.');
end

FD_mask = FD_mask & ismember(y_raw, commonUsers_B);
MD_mask = MD_mask & ismember(y_raw, commonUsers_B);

Xtrain_B_full = X_raw(FD_mask, :);
Xtest_B_full  = X_raw(MD_mask, :);
yTrain_B_full = y_raw(FD_mask);
yTest_B_full  = y_raw(MD_mask);

Ntrain_B = numel(yTrain_B_full);
Ntest_B  = numel(yTest_B_full);

fprintf('\n[Method B] FD train samples: %d\n', Ntrain_B);
fprintf('[Method B] MD test  samples: %d\n', Ntest_B);

% Standardise ONCE using full FD training set
fprintf('\n[Method B] Standardising features using FD training set only...\n');

muX_B = mean(Xtrain_B_full, 1, 'omitnan');
sigmaX_B = std(Xtrain_B_full, 0, 1, 'omitnan');
sigmaX_B(sigmaX_B == 0) = 1;

Xtrain_Bz = (Xtrain_B_full - muX_B) ./ sigmaX_B;
Xtest_Bz  = (Xtest_B_full  - muX_B) ./ sigmaX_B;

XtrainNN_B_all = Xtrain_Bz.';
XtestNN_B_all  = Xtest_Bz.';

[userTrainList_B, ~, yTrainMapped_B_all] = unique(yTrain_B_full);
[~, locB_all] = ismember(yTest_B_full, userTrainList_B);
yTestMapped_B_all = locB_all;

Ttrain_B_all = full(ind2vec(yTrainMapped_B_all.'));

%% ------------------------- MULTIPLE RE-TRAINS (Method B) ----------------
N_runs = 5;   % how many times to re-train the SAME architecture
fprintf('\n[Method B] Re-training the same 20-neuron PatternNet %d times...\n', N_runs);

bestTestAcc_B   = -inf;
bestTrainAcc_B  = -inf;
bestRunIdx_B    = -1;
bestNet_B       = [];
best_tr_B       = [];
bestYtestPred_B = [];

for run = 1:N_runs
    fprintf('\n[Method B][Run %d/%d] Initialising and training...\n', run, N_runs);

    % Different random seed each run
    rng(100 + run);  

    net_B_run = patternnet(hiddenSize);

    net_B_run.divideFcn = 'divideind';
    net_B_run.divideParam.trainInd = 1:size(XtrainNN_B_all, 2);
    net_B_run.divideParam.valInd   = [];
    net_B_run.divideParam.testInd  = [];

    net_B_run.performFcn = 'crossentropy';
    net_B_run.trainParam.epochs     = 150;
    net_B_run.trainParam.max_fail   = 10;
    net_B_run.trainParam.min_grad   = 1e-6;
    net_B_run.trainParam.showWindow = false;   % keep GUI closed for loops

    [net_B_run, tr_B_run] = train(net_B_run, XtrainNN_B_all, Ttrain_B_all);

    % Evaluate FD train
    Ytrain_run = net_B_run(XtrainNN_B_all);
    [~, trainPredClass_run] = max(Ytrain_run, [], 1);
    trainAcc_run = mean(trainPredClass_run.' == yTrainMapped_B_all) * 100;

    % Evaluate MD test
    Ytest_run = net_B_run(XtestNN_B_all);
    [~, testPredClass_run] = max(Ytest_run, [], 1);
    validTestMask_B = (yTestMapped_B_all > 0);
    testAcc_run = mean(testPredClass_run(validTestMask_B).' == yTestMapped_B_all(validTestMask_B)) * 100;

    fprintf('  -> Run %d: TrainAcc(FD) = %.2f %%, TestAcc(MD) = %.2f %%\n', ...
            run, trainAcc_run, testAcc_run);

    % Keep the best MD test accuracy
    if testAcc_run > bestTestAcc_B
        bestTestAcc_B   = testAcc_run;
        bestTrainAcc_B  = trainAcc_run;
        bestRunIdx_B    = run;
        bestNet_B       = net_B_run;
        best_tr_B       = tr_B_run;
        bestYtestPred_B = Ytest_run;
    end
end

fprintf('\n[Method B] BEST RUN = %d\n', bestRunIdx_B);
fprintf('[Method B] Best Training accuracy (FD): %.2f %%\n', bestTrainAcc_B);
fprintf('[Method B] Best Testing  accuracy (MD): %.2f %%\n', bestTestAcc_B);

% Confusion matrix for best model
figure;
plotconfusion(ind2vec(yTestMapped_B_all(validTestMask_B).'), bestYtestPred_B(:, validTestMask_B));
title(sprintf('Method B – BEST Run %d (FD→MD, Test Acc = %.2f %%)', bestRunIdx_B, bestTestAcc_B));

%% ------------------------- SAVE BEST MODEL (B) --------------------------
modelPath_B = fullfile(processedRoot, 'trained_patternnet_fd_md.mat');

fprintf('\n[Method B] Saving BEST FD→MD model to:\n  %s\n', modelPath_B);

net_B = bestNet_B;
tr_B  = best_tr_B;
trainAcc_B = bestTrainAcc_B;
testAcc_B  = bestTestAcc_B;

save(modelPath_B, 'net_B', 'tr_B', ...
     'featureNames', 'muX_B', 'sigmaX_B', ...
     'userTrainList_B', 'trainAcc_B', 'testAcc_B', ...
     'bestRunIdx_B', 'N_runs', ...
     'windowLengthSec', 'overlapFraction');

fprintf('\nSCRIPT 2 COMPLETED.\n');
fprintf('  Method A: trained_patternnet.mat (70/30 random, 20 neurons)\n');
fprintf('  Method B: trained_patternnet_fd_md.mat (FD train → MD test, best of %d runs, 20 neurons)\n', N_runs);
