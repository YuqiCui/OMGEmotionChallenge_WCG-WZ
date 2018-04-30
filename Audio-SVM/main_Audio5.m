clc; clearvars; close all; rng('default');
dqWorker = parallel.pool.DataQueue; afterEach(dqWorker, @(ids) fprintf('%d-%d ', ids(1),ids(2))); % individual worker

YTrain=xlsread('data\omg_TrainVideos.csv','F2:G2443');
YVal=xlsread('data\omg_ValidationVideos.csv','F2:G622');

load('data\feature_Train_76'); XTrain= FeatureMatrix;
load('data\feature_Val_76'); XVal= FeatureMatrix;
load('data\featureMatrix_test');XTest=FeatureMatrix;

%一起归一化
data=cat(1,XTrain,XVal);
idsTrain=1:size(XTrain,1);
idsVal=size(XTrain,1)+1:size(data,1);
[data,PS1]=mapminmax(data');
data=data';
XTrain=data(idsTrain,:); XVal=data(idsVal,:);
XTest=mapminmax('apply',XTest',PS1)';

%% Clip the features 截断，并且截断之后再一次进行了归一化
for i=1:size(XTrain,2)
    lowerX=prctile(data(:,i),2); upperX=prctile(data(:,i),98);
    XTrain(XTrain(:,i)<lowerX,i)=lowerX;
    XTrain(XTrain(:,i)>upperX,i)=upperX;
    XVal(XVal(:,i)<lowerX,i)=lowerX;
    XVal(XVal(:,i)>upperX,i)=upperX;
    XTest(XTest(:,i)<lowerX,i)=lowerX;
    XTest(XTest(:,i)>upperX,i)=upperX;
end
data=cat(1,XTrain,XVal);
[data,PS2]=mapminmax(data');
data=data';
XTrain=data(idsTrain,:); XVal=data(idsVal,:);
XTest=mapminmax('apply',XTest',PS2)';

% ccc=nan(1,2);
% m=2;
% [ccc(1),bestModelA,rA,bestK_A]=regressions(XTrain,YTrain(:,1),XVal,YVal(:,1),m); ccc
% [ccc(2),bestModelV,rV,bestK_V]=regressions(XTrain,YTrain(:,2),XVal,YVal(:,2),m); ccc
% save('AudioResults.mat','ccc','bestModelA','bestModelV','bestK_A','bestK_V','rA','rV');


%% bestModelA and bestModelV can then be used in svmpredict for prediction %%%
load('AudioResults.mat');
% XValA=XVal;
% XValV=XVal;
% XValA=XValA(:,rA); 
% XValV=XValV(:,rV);
% [arousal, accuracy1, decision_values1] = svmpredict(YVal(:,1),XValA(:,1:bestK_A),bestModelA);
% [valence, accuracy2, decision_values2] = svmpredict(YVal(:,2),XValV(:,1:bestK_V),bestModelV);
% save('pred_valLabel','arousal','valence');
% ccc1=CCC(YVal(:,1),arousal);
% ccc2=CCC(YVal(:,2),valence);
XTestA=XTest;
XTestV=XTest;
XTestA=XTestA(:,rA); 
XTestV=XTestV(:,rV);
YTest=zeros(size(XTestA,1),1);
[arousal,~,~] = svmpredict(YTest,XTestA(:,1:bestK_A),bestModelA);
[valence,~,~] = svmpredict(YTest,XTestV(:,1:bestK_V),bestModelV);
save('pred_TestLabel','arousal','valence');

function [ccc,bestModel,r,bestK]=regressions(XTrain,yTrain,XTest,yTest,idxMethod)
%% 3 feature selection and regression approaches:
% 1. No feature selection
% 2. relieff
% 3. PCA
switch idxMethod
    case 1 % No feature selection
        
    case 2 % relieff
        r=relieff(XTrain,yTrain,10,'method','regression');
        XTrain=XTrain(:,r);
        XTest=XTest(:,r);
        maxCCC=-1;
        for k=1:size(XTrain,2)
            ccc=optSVM(XTrain(:,1:k),yTrain,XTest(:,1:k),yTest);
            if ccc>maxCCC
                maxCCC=ccc;       bestK=k;
            end
        end
        XTrain=XTrain(:,1:bestK);
        XTest=XTest(:,1:bestK);
end
[ccc,bestModel]=optSVM(XTrain,yTrain,XTest,yTest);
end

function [maxCCC,bestModel,bestC,bestG]=optSVM(XTrain,yTrain,XTest,yTest)
% find the best parameters for SVR
maxCCC=-1;
for log2c=-2:2 
    C=2^log2c;
    for log2g=-2:2
        Gamma=2^log2g;
        model=svmtrain(yTrain,XTrain,['-h 0 -t 2 -s 3 -c ', num2str(C) ,' -g ' ,num2str(Gamma)]);
        yPred=svmpredict(yTest,XTest,model);
        yPred=max(min(yTrain),min(max(yTrain),yPred));
        ccc=CCC(yTest,yPred);
        if ccc>maxCCC
            maxCCC=ccc; bestModel=model; bestC=C; bestG=Gamma;
        end
    end
end
end

function ccc=CCC(yTrue,yPred)
mTrue=mean(yTrue); stdTrue=std(yTrue);
mPred=mean(yPred); stdPred=std(yPred);
rho=corr(yTrue,yPred);
ccc=2*rho*stdTrue*stdPred/(stdPred^2+stdTrue^2+(mPred-mTrue)^2);
end

