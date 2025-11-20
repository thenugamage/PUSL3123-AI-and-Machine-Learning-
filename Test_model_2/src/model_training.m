%% Script 3 — model_training.m
% FF-MLP classifier + verification metrics.
% Prints per-user FAR/FRR/EER for:
%   (A) TEST sweep at EER point
%   (B) Operational (threshold set on TRAIN), evaluated on TEST
% Also logs mean FAR/FRR/EER into results/metrics/summary_log.csv

%% ---------- Configuration ----------
clearvars -except exp_name split_mode holdout hiddenSizes trainFcn use_mrmr Ksel
thisFile = mfilename('fullpath'); srcDir = fileparts(thisFile); projectRoot = fileparts(srcDir);
cfgPath = fullfile(projectRoot,'config','config.m');
if exist(cfgPath,'file'), run(cfgPath); else, fs=31; win_s=2; hop_s=1; rng(42); end %#ok<NASGU>

% Injectables (safe defaults)
if ~exist('exp_name','var'), exp_name = 'NN_experiment'; end
if ~exist('split_mode','var'), split_mode = 'session'; end   % 'session' or 'random'
if ~exist('holdout','var'), holdout = 0.20; end
if ~exist('hiddenSizes','var'), hiddenSizes = [64 32]; end
if ~exist('trainFcn','var'), trainFcn = 'trainscg'; end      % 'trainscg' or 'trainbr'
if ~exist('use_mrmr','var'), use_mrmr = false; end
if ~exist('Ksel','var'), Ksel = 40; end
reStandardizeFeatures = true;

inCSV  = fullfile(projectRoot,'data','processed','features.csv');
figDir = fullfile(projectRoot,'results','figures'); if ~exist(figDir,'dir'), mkdir(figDir); end
metDir = fullfile(projectRoot,'results','metrics'); if ~exist(metDir,'dir'), mkdir(metDir); end
modelDir = fullfile(projectRoot,'models'); if ~exist(modelDir,'dir'), mkdir(modelDir); end
modelOut = fullfile(modelDir, sprintf('trained_model_%s.mat', exp_name));
summaryCSV = fullfile(metDir,'summary_log.csv');

%% ---------- Load features ----------
if ~isfile(inCSV), error('features.csv not found. Run Script 2 first.'); end
T = readtable(inCSV);
exclude = {'UserID','Session','WinStartIdx','WinEndIdx','CenterTimeSec'};
featCols = setdiff(T.Properties.VariableNames, exclude, 'stable');
Xall = T{:,featCols}; Yall = T.UserID;
if ~iscategorical(T.Session), T.Session=categorical(T.Session); end
classes = unique(Yall); C = numel(classes);

%% ---------- Split ----------
switch lower(split_mode)
    case 'session'
        if ~any(T.Session=='FD') || ~any(T.Session=='MD')
            error('Need both FD and MD sessions for session-based split.');
        end
        trainIdx = (T.Session=='FD'); testIdx = (T.Session=='MD');
    case 'random'
        cv = cvpartition(Yall,'HoldOut',holdout); trainIdx=training(cv); testIdx=test(cv);
    otherwise, error('Unknown split_mode: %s', split_mode);
end
Xtr = Xall(trainIdx,:); Xte = Xall(testIdx,:); Ytr = Yall(trainIdx); Yte = Yall(testIdx);

% Standardize on TRAIN only
featScaler = [];
if reStandardizeFeatures
    mu = mean(Xtr,1,'omitnan'); sg = std(Xtr,0,1,'omitnan'); sg(sg==0)=eps;
    Xtr = (Xtr - mu)./sg; Xte = (Xte - mu)./sg;
    featScaler.mu=mu; featScaler.sigma=sg; featScaler.columns=featCols;
end

% mRMR on TRAIN (optional)
if use_mrmr && exist('fscmrmr','file')==2
    [idx,~] = fscmrmr(Xtr, Ytr);
    sel = idx(1:min(Ksel,size(Xtr,2)));
    Xtr = Xtr(:,sel); Xte = Xte(:,sel); featCols = featCols(sel);
elseif use_mrmr
    warning('fscmrmr not found. Skipping mRMR.'); use_mrmr=false;
end

% One-hot targets
[~, trIdx] = ismember(Ytr, classes); Ttr = full(ind2vec(trIdx'));

%% ---------- Train FF-MLP ----------
net = patternnet(hiddenSizes, trainFcn);
net.performFcn = 'crossentropy';
net.trainParam.epochs = 300; net.trainParam.max_fail = 10;
net.divideFcn = 'dividerand'; net.divideParam.trainRatio=0.85; net.divideParam.valRatio=0.15; net.divideParam.testRatio=0.0;
[net, tr] = train(net, Xtr', Ttr);

%% ---------- Classification metrics ----------
scores = net(Xte'); [~, predIdx] = max(scores,[],1); Ypred = classes(predIdx)';
cm = confusionmat(Yte, Ypred, 'order', classes); Ntest = sum(cm(:));
acc = sum(diag(cm)) / Ntest;

prec=zeros(C,1); rec=zeros(C,1); f1=zeros(C,1); supp=zeros(C,1);
for i=1:C, tp=cm(i,i); fp=sum(cm(:,i))-tp; fn=sum(cm(i,:))-tp;
    prec(i)=tp/(tp+fp+eps); rec(i)=tp/(tp+fn+eps); f1(i)=2*prec(i)*rec(i)/(prec(i)+rec(i)+eps); supp(i)=sum(cm(i,:));
end
macroF1 = mean(f1); weightedF1 = sum(f1 .* supp)/sum(supp);

writetable(table(classes,prec,rec,f1,supp,'VariableNames',{'UserID','Precision','Recall','F1','Support'}), ...
    fullfile(metDir, sprintf('classification_per_class_%s.csv',exp_name)));
writetable(table(acc,macroF1,weightedF1,'VariableNames',{'Accuracy','MacroF1','WeightedF1'}), ...
    fullfile(metDir, sprintf('classification_overall_%s.csv',exp_name)));

fprintf('\n=== [%s] Classification (TEST) ===\n', exp_name);
fprintf('Accuracy: %.4f | Macro-F1: %.4f | Weighted-F1: %.4f\n', acc, macroF1, weightedF1);

%% ---------- Verification A: TEST sweep at EER (per-user prints as %) ----------
thresholds = linspace(0,1,1001);
EER=zeros(C,1); tauEER=zeros(C,1); FAR_E=zeros(C,1); FRR_E=zeros(C,1);

for i=1:C
    pos=(Yte==classes(i)); gen=scores(i,pos); imp=scores(i,~pos);
    FAR=zeros(numel(thresholds),1); FRR=zeros(numel(thresholds),1);
    for t=1:numel(thresholds), tau=thresholds(t); FRR(t)=mean(gen<tau); FAR(t)=mean(imp>=tau); end
    [~,idx]=min(abs(FAR-FRR));
    EER(i)    = (FAR(idx)+FRR(idx))/2;
    tauEER(i) = thresholds(idx);
    FAR_E(i)  = FAR(idx);
    FRR_E(i)  = FRR(idx);
end
meanEER_curve = mean(EER);
meanFAR_curve = mean(FAR_E);
meanFRR_curve = mean(FRR_E);

fprintf('\n=== [%s] Verification (TEST sweep @ EER) — per-user ===\n', exp_name);
fprintf('%6s | %9s | %9s | %9s | %9s\n','User','Tau_EER','FAR_EER','FRR_EER','EER');
for i=1:C
    fprintf('%6d | %9.4f | %8.2f%% | %8.2f%% | %8.2f%%\n', ...
        classes(i), tauEER(i), 100*FAR_E(i), 100*FRR_E(i), 100*EER(i));
end
fprintf('MEANS (TEST sweep) -> FAR: %.2f%% | FRR: %.2f%% | EER: %.2f%%\n', ...
    100*meanFAR_curve, 100*meanFRR_curve, 100*meanEER_curve);

writetable(table(classes,tauEER,FAR_E,FRR_E,EER, ...
    'VariableNames',{'UserID','Tau_EER','FAR_EER','FRR_EER','EER'}), ...
    fullfile(metDir, sprintf('FAR_FRR_EER_curve_test_%s.csv',exp_name)));

%% ---------- Verification B: Operational (τ from TRAIN), evaluated on TEST ----------
scores_tr = net(Xtr'); tau_train=zeros(C,1);
for i=1:C
    pos_tr=(Ytr==classes(i)); gen_tr=scores_tr(i,pos_tr); imp_tr=scores_tr(i,~pos_tr);
    [FAR_tr,FRR_tr]=far_frr_curve(gen_tr,imp_tr,thresholds); [~,idx]=min(abs(FAR_tr-FRR_tr)); tau_train(i)=thresholds(idx);
end
FAR_test=zeros(C,1); FRR_test=zeros(C,1);
for i=1:C
    pos_te=(Yte==classes(i)); gen_te=scores(i,pos_te); imp_te=scores(i,~pos_te);
    FRR_test(i)=mean(gen_te<tau_train(i)); FAR_test(i)=mean(imp_te>=tau_train(i));
end
EER_test   = (FAR_test+FRR_test)/2;
meanFAR_op = mean(FAR_test); 
meanFRR_op = mean(FRR_test); 
meanEER_op = mean(EER_test);

fprintf('\n=== [%s] Verification (Operational τ from TRAIN) — per-user ===\n', exp_name);
fprintf('%6s | %9s | %9s | %9s | %9s\n','User','Tau_train','FAR_test','FRR_test','EER_test');
for i=1:C
    fprintf('%6d | %9.4f | %8.2f%% | %8.2f%% | %8.2f%%\n', ...
        classes(i), tau_train(i), 100*FAR_test(i), 100*FRR_test(i), 100*EER_test(i));
end
fprintf('MEANS (Operational) -> FAR: %.2f%% | FRR: %.2f%% | EER: %.2f%%\n', ...
    100*meanFAR_op, 100*meanFRR_op, 100*meanEER_op);

writetable(table(classes,tau_train,FAR_test,FRR_test,EER_test, ...
    'VariableNames',{'UserID','Tau_train','FAR_test','FRR_test','EER_test'}), ...
    fullfile(metDir, sprintf('FAR_FRR_operational_%s.csv',exp_name)));

%% ---------- Summary row (Accuracy + mean FAR/FRR/EER for both modes) ----------
Summ = table(string(exp_name), ...
             acc*100, macroF1*100, weightedF1*100, ...
             meanFAR_curve*100, meanFRR_curve*100, meanEER_curve*100, ...
             meanFAR_op*100,    meanFRR_op*100,    meanEER_op*100, ...
    'VariableNames',{'Experiment','Accuracy_pct','MacroF1_pct','WeightedF1_pct', ...
                     'MeanFAR_curve_pct','MeanFRR_curve_pct','MeanEER_curve_pct', ...
                     'MeanFAR_oper_pct','MeanFRR_oper_pct','MeanEER_oper_pct'});
append_summary_row(summaryCSV, Summ);
fprintf('\n--- SUMMARY ROW ---\n'); disp(Summ);

%% ---------- Save model ----------
training_config = struct('exp_name',exp_name,'split_mode',split_mode,'hiddenSizes',hiddenSizes, ...
    'trainFcn',trainFcn,'use_mrmr',use_mrmr,'Ksel',Ksel);
save(modelOut,'net','classes','featCols','featScaler','training_config','tr');
fprintf('Model saved: %s\nMetrics -> %s\n', modelOut, metDir);

%% ---------- Helpers ----------
function [FAR, FRR] = far_frr_curve(gen, imp, thresholds)
FAR=zeros(numel(thresholds),1); FRR=zeros(numel(thresholds),1);
for t=1:numel(thresholds), tau=thresholds(t); FRR(t)=mean(gen<tau); FAR(t)=mean(imp>=tau); end
end

function append_summary_row(csvPath, Summ)
if isfile(csvPath)
    Prev = readtable(csvPath);
    allNames = unique([Prev.Properties.VariableNames, Summ.Properties.VariableNames],'stable');
    for k = 1:numel(allNames)
        name = allNames{k};
        if ~ismember(name, Prev.Properties.VariableNames), Prev.(name) = NaN(height(Prev),1); end
        if ~ismember(name, Summ.Properties.VariableNames), Summ.(name) = NaN(height(Summ),1); end
    end
    Prev = Prev(:, allNames); Summ = Summ(:, allNames);
    Out = [Prev; Summ]; writetable(Out, csvPath);
else
    writetable(Summ, csvPath);
end
end
