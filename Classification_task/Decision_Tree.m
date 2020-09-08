clc;
clear;
close all;

%% Initializing parameters and data required.
numRight=0;
wrong=0;

load MOCA1

X =data(:,3:end-1);      % Sample
Y = data(:,end);         % Targets
    
[R C]=size(data);

% produce  Train data 
numTrain=ceil(80*R/100);
XTrainData=X(1:numTrain,:);
YTrainData=Y(1:numTrain,:);
 
% produce test data
XTestData=X(numTrain:end,:); 
YTestData=Y(numTrain:end,:);
nytst=size(YTestData,1);

%% Train Decision Tree Classifier

t=ClassificationTree.fit(XTrainData,YTrainData);
disp('Resub. Loss =');
disp(resubLoss(t));

%% Test Decision Tree Classifier

Yhat=t.predict(XTestData);

%Calculate Accuracy
for i = 1 : nytst
    if Yhat(i)==YTestData(i)
        numRight = numRight + 1;
    else
        wrong = wrong + 1;
    end
end

Accuracy = numRight/nytst
disp('Accuracy test =');
disp(ACC);

view(t,'mode','graph')



%% Cross-validation

% cvmodel=crossval(t);
% 
% disp('k-Fold Loss =');
% disp(kfoldLoss(cvmodel));

