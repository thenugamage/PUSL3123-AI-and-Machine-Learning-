%% model_training_knn.m — KNN baseline with per-user FAR/FRR/EER (percent formatting)
% Optional base-workspace vars:
%   exp_name (default 'KNN_experiment'), K_for_knn (default 7),
%   use_mrmr (default false), Ksel (default 40).
% Appends unified summary to results/metrics/summary_log.csv.

clearvars -except exp_name K_for_knn use_mrmr Ksel
thisFile=mfilename('fullpath'); srcDir=fileparts(thisFile); projectRoot=fileparts(srcDir);
inCSV = fullfile(projectRoot,'data','processed','features.csv');
metDir=fullfile(projectRoot,'results','metrics'); if ~exist(metDir,'dir'), mkdir(metDir); end
summaryCSV = fullfile(metDir,'summary_log.csv');

if ~exist('exp_name','var'), exp_name = 'KNN_experiment'; end
if ~exist('K_for_knn','var'), K_for_knn = 7; end
if ~exist('use_mrmr','var'), use_mrmr = false; end
if ~exist('Ksel','var'), Ksel = 40; end

T = readtable(inCSV);
exclude={'UserID','Session','WinStartIdx','WinEndIdx','CenterTimeSec'};
featCols = setdiff(T.Properties.VariableNames, exclude, 'stable');
X = T{:,featCols}; Y = T.UserID; classes=unique(Y); C=numel(classes);
if ~iscategorical(T.Session), T.Session=categorical(T.Session); end

% FD->MD split
trainIdx=(T.Session=='FD'); testIdx=(T.Session=='MD');
Xtr=X(trainIdx,:); Xte=X(testIdx,:); Ytr=Y(trainIdx); Yte=Y(testIdx);

% Standardize TRAIN only
mu=mean(Xtr,1,'omitnan'); sg=std(Xtr,0,1,'omitnan'); sg(sg==0)=eps;
Xtr=(Xtr-mu)./sg; Xte=(Xte-mu)./sg;

% Optional mRMR
if use_mrmr && exist('fscmrmr','file')==2
    [idx,~]=fscmrmr(Xtr,Ytr); sel=idx(1:min(Ksel,size(Xtr,2)));
    Xtr=Xtr(:,sel); Xte=Xte(:,sel); featCols=featCols(sel);
elseif use_mrmr
    warning('fscmrmr not found. Skipping mRMR.'); use_mrmr=false;
end

% Train KNN
mdl = fitcknn(Xtr, Ytr, 'NumNeighbors',K_for_knn, 'Distance','euclidean', 'DistanceWeight','inverse');

% Predict on TEST
[Yhat, score] = predict(mdl, Xte);    % score: Ntest x C
cm=confusionmat(Yte, Yhat,'order',classes); acc=sum(diag(cm))/sum(cm(:));

% Macro/Weighted F1
prec=zeros(C,1); rec=zeros(C,1); f1=zeros(C,1); supp=zeros(C,1);
for i=1:C, tp=cm(i,i); fp=sum(cm(:,i))-tp; fn=sum(cm(i,:))-tp;
    prec(i)=tp/(tp+fp+eps); rec(i)=tp/(tp+fn+eps); f1(i)=2*prec(i)*rec(i)/(prec(i)+rec(i)+eps); supp(i)=sum(cm(i,:));
end
macroF1 = mean(f1); weightedF1 = sum(f1 .* supp)/sum(supp);

fprintf('\n=== [%s | K=%d] Classification (TEST) ===\nAccuracy: %.4f | Macro-F1: %.4f | Weighted-F1: %.4f\n', ...
    exp_name, K_for_knn, acc, macroF1, weightedF1);

%% ---------- Verification A: TEST sweep at EER (per-user prints as %) ----------
thresholds=linspace(0,1,1001); EER=zeros(C,1); FAR_E=zeros(C,1); FRR_E=zeros(C,1); tauEER=zeros(C,1);
for i=1:C
    s=score(:,i); pos=(Yte==classes(i)); gen=s(pos); imp=s(~pos);
    FAR=zeros(numel(thresholds),1); FRR=zeros(numel(thresholds),1);
    for t=1:numel(thresholds), tau=thresholds(t); FRR(t)=mean(gen<tau); FAR(t)=mean(imp>=tau); end
    [~,idx]=min(abs(FAR-FRR)); EER(i)=(FAR(idx)+FRR(idx))/2; FAR_E(i)=FAR(idx); FRR_E(i)=FRR(idx); tauEER(i)=thresholds(idx);
end
meanEER_curve = mean(EER); meanFAR_curve=mean(FAR_E); meanFRR_curve=mean(FRR_E);

fprintf('\n=== [%s | K=%d] Verification (TEST sweep @ EER) — per-user ===\n', exp_name, K_for_knn);
fprintf('%6s | %9s | %9s | %9s | %9s\n','User','Tau_EER','FAR_EER','FRR_EER','EER');
for i=1:C
    fprintf('%6d | %9.4f | %8.2f%% | %8.2f%% | %8.2f%%\n', classes(i), tauEER(i), 100*FAR_E(i), 100*FRR_E(i), 100*EER(i));
end
fprintf('MEANS (TEST sweep) -> FAR: %.2f%% | FRR: %.2f%% | EER: %.2f%%\n', 100*meanFAR_curve, 100*meanFRR_curve, 100*meanEER_curve);

writetable(table(classes,tauEER,FAR_E,FRR_E,EER, ...
    'VariableNames',{'UserID','Tau_EER','FAR_EER','FRR_EER','EER'}), ...
    fullfile(metDir, sprintf('KNN_FAR_FRR_EER_curve_test_%s.csv',exp_name)));

%% ---------- Verification B: Operational (τ from TRAIN), evaluated on TEST ----------
[~, score_tr] = predict(mdl, Xtr);
tau_train=zeros(C,1);
for i=1:C
    pos_tr=(Ytr==classes(i)); gen_tr=score_tr(pos_tr,i); imp_tr=score_tr(~pos_tr,i);
    FAR_tr=zeros(numel(thresholds),1); FRR_tr=zeros(numel(thresholds),1);
    for t=1:numel(thresholds), tau=thresholds(t); FRR_tr(t)=mean(gen_tr<tau); FAR_tr(t)=mean(imp_tr>=tau); end
    [~,idx]=min(abs(FAR_tr-FRR_tr)); tau_train(i)=thresholds(idx);
end
FAR_test=zeros(C,1); FRR_test=zeros(C,1);
for i=1:C
    pos_te=(Yte==classes(i)); gen_te=score(pos_te,i); imp_te=score(~pos_te,i);
    FRR_test(i)=mean(gen_te<tau_train(i)); FAR_test(i)=mean(imp_te>=tau_train(i));
end
EER_test=(FAR_test+FRR_test)/2;
meanFAR_op=mean(FAR_test); meanFRR_op=mean(FRR_test); meanEER_op=mean(EER_test);

fprintf('\n=== [%s | K=%d] Verification (Operational τ from TRAIN) — per-user ===\n', exp_name, K_for_knn);
fprintf('%6s | %9s | %9s | %9s | %9s\n','User','Tau_train','FAR_test','FRR_test','EER_test');
for i=1:C
    fprintf('%6d | %9.4f | %8.2f%% | %8.2f%% | %8.2f%%\n', classes(i), tau_train(i), 100*FAR_test(i), 100*FRR_test(i), 100*EER_test(i));
end
fprintf('MEANS (Operational) -> FAR: %.2f%% | FRR: %.2f%% | EER: %.2f%%\n', 100*meanFAR_op, 100*meanFRR_op, 100*meanEER_op);

% Unified summary row
Summ = table(string(exp_name), ...
             acc*100, macroF1*100, weightedF1*100, ...
             meanFAR_curve*100, meanFRR_curve*100, meanEER_curve*100, ...
             meanFAR_op*100,    meanFRR_op*100,    meanEER_op*100, ...
    'VariableNames',{'Experiment','Accuracy_pct','MacroF1_pct','WeightedF1_pct', ...
                     'MeanFAR_curve_pct','MeanFRR_curve_pct','MeanEER_curve_pct', ...
                     'MeanFAR_oper_pct','MeanFRR_oper_pct','MeanEER_oper_pct'});
append_summary_row(summaryCSV, Summ);
disp(Summ);

%% ---------- Helper ----------
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
