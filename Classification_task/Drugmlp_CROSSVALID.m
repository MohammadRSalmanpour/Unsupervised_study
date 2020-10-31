clear
clc
% load Xtrain_MLP_81
% load Xtest_MLP_81
% load Ytrain_MLP_81
% load Ytest_MLP_81 

% load TrainData81
% load TestData81
% load lblTrain81
% load lblTest81 

datatotal=xlsread('drug490.xlsx','Sheet1');
datatotal=datatotal';

n=size(datatotal,1) ;
R = randperm(n)';
c=0.8;
TrainData=datatotal(round(R(1:c*n)),1:end-1);
lblTrain=datatotal(round(R(1:c*n)),end);
TestData=datatotal(R(c*n+1:end),1:end-1);
lblTest=datatotal(round(R(c*n+1:end)),end);
TrainData=TrainData';
TestData=TestData';
lblTest=lblTest';
lblTrain=lblTrain';

% data_datatotal=datatotal(:,1:end-1);
% data_label=datatotal(:,end);

% k=5;
% index=crossvalind('kfold',size(datatotal,1),k);
% make ANN
performFcn = 'crossentropy';  % Cross-Entropy

trainFcn = 'trainscg';% Scaled conjugate gradient backpropagation.
net3 = patternnet([100 100],trainFcn,performFcn);
%net3.divideParam.valRatio = 0/100;
view(net3)
net3 = train(net3,TrainData,lblTrain);
output = net3(TestData);
output1=round(output);

plotconfusion(lblTest,output1)
[cc,cm]=confusion(lblTest,output1)
figure;
plot(output1,'r*');
hold on;
plot(lblTest,'go');
legend('output','lblTest');










