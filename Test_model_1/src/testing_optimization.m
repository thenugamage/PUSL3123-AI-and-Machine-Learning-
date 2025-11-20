% =========================================================================
%  PUSL3123 - SCRIPT 3
%  EVALUATION (FAR/FRR/EER) + OPTIMISATION
%
%  Uses ONLY:
%   - data/processed/features_all.mat      (from Script 1)
%   - data/processed/trained_patternnet_fd_md.mat (from Script 2, Method B)
%
%  PART 1A: FD-only (First Day) DET curve
%  PART 1B: FD -> MD evaluation (per-user and global FAR/FRR/EER)
%  PART 2 : Optimisation (hidden neurons, KNN, top features)
%
%  NOTE: FD-only DET is an optimistic same-day scenario
%        FD->MD is the realistic cross-day scenario for the report
% =========================================================================

clc; clear; close all;

%% -------------------------- PATHS ---------------------------------------
projectRoot    = fileparts(fileparts(mfilename('fullpath')));  % src/.. -> root
processedRoot  = fullfile(projectRoot, 'data', 'processed');

featuresMatPath = fullfile(processedRoot, 'features_all.mat');
modelFDMDPath   = fullfile(processedRoot, 'trained_patternnet_fd_md.mat');
varCsvPath      = fullfile(processedRoot, 'feature_variability_top.csv');

if ~exist(featuresMatPath, 'file')
    error('features_all.mat not found at:\n  %s\nRun Script 1 first.', featuresMatPath);
end
if ~exist(modelFDMDPath, 'file')
    error('trained_patternnet_fd_md.mat not found at:\n  %s\nRun Script 2 (Method B) first.', modelFDMDPath);
end

fprintf('\n========================================\n');
fprintf('  SCRIPT 3: EVALUATION + OPTIMISATION\n');
fprintf('========================================\n');

%% ------------------------ LOAD DATA & MODEL -----------------------------
load(featuresMatPath, 'allFeatures', 'allUserID', 'allDayType', 'featureNames');
load(modelFDMDPath, 'net_B', 'muX_B', 'sigmaX_B', ...
                    'userTrainList_B', 'trainAcc_B', 'testAcc_B', ...
                    'windowLengthSec', 'overlapFraction');

X_all   = allFeatures;
y_all   = allUserID;
day_all = allDayType;

[N_all,D] = size(X_all);
fprintf('Total windows (ALL):   %d\n', N_all);
fprintf('Num features:          %d\n', D);
fprintf('Users (model):         %s\n', mat2str(userTrainList_B.'));

% Keep only valid labelled samples
validIdx = ~isnan(y_all) & (y_all > 0) & ~isnan(day_all) & (day_all > 0);
X   = X_all(validIdx,:);
y   = y_all(validIdx);
day = day_all(validIdx);

% Masks for FD (day=1) and MD (day=2)
FD_mask = (day == 1);
MD_mask = (day == 2);

usersFD  = unique(y(FD_mask));
usersMD  = unique(y(MD_mask));
commonUsers = intersect(intersect(usersFD, usersMD), userTrainList_B);

if numel(commonUsers) < 2
    warning('Less than 2 common users between FD and MD.');
end

FD_mask = FD_mask & ismember(y, commonUsers);
MD_mask = MD_mask & ismember(y, commonUsers);

Xtrain_FD = X(FD_mask,:);
Xtest_MD  = X(MD_mask,:);
yTrain_FD = y(FD_mask);
yTest_MD  = y(MD_mask);

fprintf('FD train samples: %d\n', numel(yTrain_FD));
fprintf('MD test  samples: %d\n', numel(yTest_MD));

% Standardise using stats from Script 2 (Method B)
Xtrain_FD_z = (Xtrain_FD - muX_B) ./ sigmaX_B;
Xtest_MD_z  = (Xtest_MD  - muX_B) ./ sigmaX_B;

XtrainNN = Xtrain_FD_z.';   % [D x N_FD]
XtestNN  = Xtest_MD_z.';    % [D x N_MD]

% Threshold grid (0..1 with step 0.01)
thresholds = 0:0.01:1;
numTh      = numel(thresholds);

numUsers   = numel(userTrainList_B);
numClasses = net_B.numOutputs;

if numClasses ~= numUsers
    warning('net_B outputs (%d) != number of users (%d).', numClasses, numUsers);
end

%% ========================================================================
%%  PART 1A – FD-ONLY DET CURVE (FIRST DAY, SAME-DAY SCENARIO)
%% ========================================================================
fprintf('\n========================================\n');
fprintf('  PART 1A – FD-ONLY DET CURVE (FIRST DAY)\n');
fprintf('========================================\n');

% Scores for FD windows (network trained on FD in Script 2)
Y_FD = net_B(XtrainNN);      % [numClasses x N_FD]

allGenuine_FD  = [];
allImpostor_FD = [];

for uIdx = 1:numUsers
    uLabel = userTrainList_B(uIdx);

    genuineMask_fd  = (yTrain_FD == uLabel);
    impostorMask_fd = (yTrain_FD ~= uLabel);
    if ~any(genuineMask_fd) || ~any(impostorMask_fd)
        continue;
    end

    scores = Y_FD(uIdx,:);   % score for class u over all FD samples
    allGenuine_FD  = [allGenuine_FD;  scores(genuineMask_fd).'];
    allImpostor_FD = [allImpostor_FD; scores(impostorMask_fd).'];
end

FAR_global_FD = zeros(1,numTh);
FRR_global_FD = zeros(1,numTh);

for t = 1:numTh
    tau = thresholds(t);
    FAR_global_FD(t) = mean(allImpostor_FD >= tau);
    FRR_global_FD(t) = mean(allGenuine_FD  <  tau);
end

% EER for FD-only
[~, idxFD] = min(abs(FAR_global_FD - FRR_global_FD));
EER_FD        = 0.5*(FAR_global_FD(idxFD) + FRR_global_FD(idxFD));
thrEER_FD     = thresholds(idxFD);

FAR_FD_pct = FAR_global_FD * 100;
FRR_FD_pct = FRR_global_FD * 100;

fprintf('  FD-only EER: %.4f  (%.2f %%) at tau = %.2f\n', EER_FD, 100*EER_FD, thrEER_FD);

% Plot FD-only DET (axes based on actual data)
figure('Color','w');
semilogx(FAR_FD_pct, FRR_FD_pct, 'LineWidth', 1.5); hold on;
plot(FAR_FD_pct(idxFD), FRR_FD_pct(idxFD), 'ro', 'MarkerSize', 8, 'LineWidth', 1.5);
text(FAR_FD_pct(idxFD)*1.05, FRR_FD_pct(idxFD), ...
    sprintf('EER = %.2f%% (\\tau = %.2f)', 100*EER_FD, thrEER_FD), ...
    'FontSize', 11, 'FontWeight', 'bold');

xlabel('False Positive Rate (%)');
ylabel('False Negative Rate (%)');
title('DET Curve - Time Domain - First Day (FD)');
grid on;

% axis limits: 0 → 1.2×max error, no fake zooming
xmaxFD = max(FAR_FD_pct);
ymaxFD = max(FRR_FD_pct);
xlim([max(0.1, 0.5*min(FAR_FD_pct(FAR_FD_pct>0))), min(100, 1.2*xmaxFD)]);
ylim([0, min(100, 1.2*ymaxFD)]);

%% ========================================================================
%%  PART 1B – FD -> MD: PER-USER + GLOBAL FAR / FRR / EER
%% ========================================================================
fprintf('\n========================================\n');
fprintf('  PART 1B – FD -> MD EVALUATION\n');
fprintf('========================================\n');

Ytest = net_B(XtestNN);                 % [numClasses x N_MD]
[~, testPredClass] = max(Ytest, [], 1); % winning class index per test sample

FAR_user      = zeros(numUsers,numTh);
FRR_user      = zeros(numUsers,numTh);
EER_user      = nan(numUsers,1);
ThrEER_user   = nan(numUsers,1);
FAR_atEER     = nan(numUsers,1);
FRR_atEER     = nan(numUsers,1);
userAccFrac   = nan(numUsers,1);

allGenuineScores  = [];
allImpostorScores = [];

fprintf('\nPer-user metrics (MD test set):\n');
for uIdx = 1:numUsers
    uLabel = userTrainList_B(uIdx);

    genuineMask = (yTest_MD == uLabel);
    impostorMask = (yTest_MD ~= uLabel);
    if ~any(genuineMask) || ~any(impostorMask)
        fprintf('  User %2d: no genuine or impostor samples in MD. Skipping.\n', uLabel);
        continue;
    end

    scores = Ytest(uIdx,:);   % scores for class u
    s_g = scores(genuineMask);
    s_i = scores(impostorMask);

    allGenuineScores  = [allGenuineScores;  s_g(:)];
    allImpostorScores = [allImpostorScores; s_i(:)];

    % classification accuracy (who wins)
    userAccFrac(uIdx) = mean(testPredClass(genuineMask) == uIdx);

    for t = 1:numTh
        tau = thresholds(t);
        FAR_user(uIdx,t) = mean(s_i >= tau);
        FRR_user(uIdx,t) = mean(s_g <  tau);
    end

    [~,idxu] = min(abs(FAR_user(uIdx,:) - FRR_user(uIdx,:)));
    EER_user(uIdx)    = 0.5*(FAR_user(uIdx,idxu) + FRR_user(uIdx,idxu));
    ThrEER_user(uIdx) = thresholds(idxu);
    FAR_atEER(uIdx)   = FAR_user(uIdx,idxu);
    FRR_atEER(uIdx)   = FRR_user(uIdx,idxu);

    fprintf('  User %2d: Acc = %6.2f %%, EER = %6.2f %%, FAR@EER = %.3f, FRR@EER = %.3f\n',...
        uLabel, 100*userAccFrac(uIdx), 100*EER_user(uIdx), FAR_atEER(uIdx), FRR_atEER(uIdx));
end

% Global FAR/FRR/EER (all users pooled)
FAR_global = zeros(1,numTh);
FRR_global = zeros(1,numTh);
for t = 1:numTh
    tau = thresholds(t);
    FAR_global(t) = mean(allImpostorScores >= tau);
    FRR_global(t) = mean(allGenuineScores  <  tau);
end

[~,idxG] = min(abs(FAR_global - FRR_global));
EER_global    = 0.5*(FAR_global(idxG) + FRR_global(idxG));
ThrEER_global = thresholds(idxG);

fprintf('\nFD->MD PatternNet (overall):\n');
fprintf('  Training accuracy (FD): %.2f %%\n', trainAcc_B);
fprintf('  Testing  accuracy (MD): %.2f %%\n', testAcc_B);
fprintf('  Global EER:             %.4f  (%.2f %%)\n', EER_global, 100*EER_global);
fprintf('  Threshold at EER:       %.2f\n', ThrEER_global);

% Save per-user EER + accuracy
eerTable = table;
eerTable.UserID          = userTrainList_B(:);
eerTable.UserAcc         = userAccFrac;
eerTable.UserAcc_percent = 100*userAccFrac;
eerTable.EER             = EER_user;
eerTable.EER_percent     = 100*EER_user;
eerTable.Threshold_EER   = ThrEER_user;
eerTable.FAR_at_EER      = FAR_atEER;
eerTable.FRR_at_EER      = FRR_atEER;

eerCsvOut = fullfile(processedRoot, 'eer_results_patternnet_fd_md.csv');
writetable(eerTable, eerCsvOut);

matOut = fullfile(processedRoot, 'far_frr_eer_patternnet_fd_md.mat');
save(matOut, 'thresholds','FAR_user','FRR_user',...
             'FAR_global','FRR_global',...
             'EER_user','ThrEER_user',...
             'FAR_atEER','FRR_atEER',...
             'EER_global','ThrEER_global',...
             'userTrainList_B','userAccFrac');

fprintf('  Saved EER table to:\n    %s\n', eerCsvOut);
fprintf('  Saved FAR/FRR data to:\n    %s\n', matOut);

% Plot FD->MD DET with real axis limits
figure('Color','w');
FARp = FAR_global * 100;
FRRp = FRR_global * 100;
semilogx(FARp, FRRp, 'LineWidth', 1.5); hold on;
plot(FARp(idxG), FRRp(idxG), 'ro', 'MarkerSize', 8, 'LineWidth', 1.5);
text(FARp(idxG)*1.05, FRRp(idxG), ...
    sprintf('EER = %.2f%% (\\tau = %.2f)', 100*EER_global, ThrEER_global), ...
    'FontSize', 11, 'FontWeight', 'bold');

xlabel('False Positive Rate (%)');
ylabel('False Negative Rate (%)');
title('DET Curve - Time Domain - FD \rightarrow MD');
grid on;

xmax = max(FARp);
ymax = max(FRRp);
xlim([max(0.1, 0.5*min(FARp(FARp>0))), min(100, 1.2*xmax)]);
ylim([0, min(100, 1.2*ymax)]);

%% ========================================================================
%%  PART 2 – OPTIMISATION (same data, no dummy values)
%% ========================================================================

fprintf('\n========================================\n');
fprintf('  PART 2 – OPTIMISATION EXPERIMENTS\n');
fprintf('========================================\n');

Xtrain_opt = Xtrain_FD_z;
Xtest_opt  = Xtest_MD_z;
yTrain     = yTrain_FD;
yTest      = yTest_MD;

XtrainNN_opt = Xtrain_opt.';
XtestNN_opt  = Xtest_opt.';

%% 2.1 PatternNet hidden neuron sweep
fprintf('\n[OPT 2.1] PatternNet hidden neuron sweep (FD->MD)\n');

hiddenList = [10 20 40];
opt_results_A = table('Size',[numel(hiddenList) 3], ...
    'VariableTypes',{'double','double','double'}, ...
    'VariableNames',{'HiddenNeurons','TrainAcc','TestAcc'});

for i = 1:numel(hiddenList)
    h = hiddenList(i);
    fprintf('  > %d hidden neurons...\n', h);

    net_opt = patternnet(h);
    net_opt.divideFcn = 'divideind';
    net_opt.divideParam.trainInd = 1:size(XtrainNN_opt,2);
    net_opt.divideParam.valInd   = [];
    net_opt.divideParam.testInd  = [];
    net_opt.performFcn = 'crossentropy';
    net_opt.trainParam.epochs     = 150;
    net_opt.trainParam.max_fail   = 10;
    net_opt.trainParam.min_grad   = 1e-6;
    net_opt.trainParam.showWindow = false;

    [~,~,yTrainMap] = unique(yTrain);
    Ttrain = full(ind2vec(yTrainMap.'));

    net_opt = train(net_opt, XtrainNN_opt, Ttrain);

    Ytr  = net_opt(XtrainNN_opt);
    [~,predTr] = max(Ytr,[],1);
    trainAcc = mean(predTr.' == yTrainMap)*100;

    [uTrain,~,~] = unique(yTrain);
    [~,locT] = ismember(yTest, uTrain);
    yTestMap = locT;

    Yte  = net_opt(XtestNN_opt);
    [~,predTe] = max(Yte,[],1);
    valid = (yTestMap>0);
    testAcc = mean(predTe(valid).' == yTestMap(valid))*100;

    opt_results_A.HiddenNeurons(i) = h;
    opt_results_A.TrainAcc(i)      = trainAcc;
    opt_results_A.TestAcc(i)       = testAcc;

    fprintf('    TrainAcc = %.2f %%, TestAcc = %.2f %%\n', trainAcc, testAcc);
end

optA_csv = fullfile(processedRoot, 'opt_patternnet_hidden_fd_md.csv');
writetable(opt_results_A, optA_csv);
fprintf('  Saved hidden-neuron sweep to: %s\n', optA_csv);

%% 2.2 KNN comparison
fprintf('\n[OPT 2.2] KNN classifier comparison (FD->MD)\n');

if license('test','Statistics_Toolbox')
    kList = [1 3 5];
    opt_results_K = table('Size',[numel(kList) 2], ...
        'VariableTypes',{'double','double'}, ...
        'VariableNames',{'K','TestAcc'});

    for i = 1:numel(kList)
        k = kList(i);
        fprintf('  > k = %d\n', k);

        mdl = fitcknn(Xtrain_opt, yTrain, ...
                      'NumNeighbors',k, ...
                      'Distance','euclidean', ...
                      'Standardize',false);

        yPred = predict(mdl, Xtest_opt);
        acc   = mean(yPred == yTest)*100;

        opt_results_K.K(i)       = k;
        opt_results_K.TestAcc(i) = acc;

        fprintf('    TestAcc = %.2f %%\n', acc);
    end

    optK_csv = fullfile(processedRoot, 'opt_knn_fd_md.csv');
    writetable(opt_results_K, optK_csv);
    fprintf('  Saved KNN results to: %s\n', optK_csv);
else
    warning('Statistics and Machine Learning Toolbox not available. Skipping KNN.');
end

%% 2.3 Feature-selection experiment using real variability CSV
fprintf('\n[OPT 2.3] PatternNet with top discriminative features\n');

if exist(varCsvPath, 'file')
    Tvar = readtable(varCsvPath);
    if ismember('Feature', Tvar.Properties.VariableNames)
        Kfeat = min(10, height(Tvar));
        topFeat = Tvar.Feature(1:Kfeat);

        [~,featIdx] = ismember(topFeat, featureNames);
        featIdx = featIdx(featIdx>0);

        if isempty(featIdx)
            warning('No matching features found in featureNames.');
        else
            Xtr_FS = Xtrain_FD(:,featIdx);
            Xte_FS = Xtest_MD(:,featIdx);

            mu_FS = mean(Xtr_FS,1,'omitnan');
            sd_FS = std(Xtr_FS,0,1,'omitnan'); sd_FS(sd_FS==0)=1;

            Xtr_FS_z = (Xtr_FS - mu_FS)./sd_FS;
            Xte_FS_z = (Xte_FS - mu_FS)./sd_FS;

            XtrNN_FS = Xtr_FS_z.';
            XteNN_FS = Xte_FS_z.';

            [~,~,yTrMap_FS] = unique(yTrain);
            Ttr_FS = full(ind2vec(yTrMap_FS.'));

            net_FS = patternnet(20);
            net_FS.divideFcn = 'divideind';
            net_FS.divideParam.trainInd = 1:size(XtrNN_FS,2);
            net_FS.divideParam.valInd   = [];
            net_FS.divideParam.testInd  = [];
            net_FS.performFcn = 'crossentropy';
            net_FS.trainParam.epochs     = 150;
            net_FS.trainParam.max_fail   = 10;
            net_FS.trainParam.min_grad   = 1e-6;
            net_FS.trainParam.showWindow = false;

            net_FS = train(net_FS, XtrNN_FS, Ttr_FS);

            Ytr_FS = net_FS(XtrNN_FS);
            [~,predTr_FS] = max(Ytr_FS,[],1);
            trainAcc_FS = mean(predTr_FS.' == yTrMap_FS)*100;

            [uTr_FS,~,~] = unique(yTrain);
            [~,locFS] = ismember(yTest,uTr_FS);
            yTeMap_FS = locFS;

            Yte_FS = net_FS(XteNN_FS);
            [~,predTe_FS] = max(Yte_FS,[],1);
            validFS = (yTeMap_FS>0);
            testAcc_FS = mean(predTe_FS(validFS).' == yTeMap_FS(validFS))*100;

            opt_FS = table(Kfeat, trainAcc_FS, testAcc_FS, ...
                           'VariableNames',{'NumFeatures','TrainAcc','TestAcc'});

            optFS_csv = fullfile(processedRoot, 'opt_patternnet_topfeat_fd_md.csv');
            writetable(opt_FS, optFS_csv);
            fprintf('  Saved top-feature results to: %s\n', optFS_csv);
        end
    else
        warning('feature_variability_top.csv has no "Feature" column.');
    end
else
    fprintf('  feature_variability_top.csv not found. Skip top-feature experiment.\n');
end

fprintf('\nSCRIPT 3 COMPLETED.\n');
