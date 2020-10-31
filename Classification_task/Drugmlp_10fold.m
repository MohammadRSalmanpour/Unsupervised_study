clear
clc
load Xtrain_MLP_81
load Xtest_MLP_81
load Ytrain_MLP_81
load Ytest_MLP_81 

% load TrainData81
% load TestData81
% load lblTrain81
% load lblTest81 
% 
% datatotal=xlsread('drug490.xlsx','Sheet1');
% datatotal=datatotal';
% 
% n=size(datatotal,1) ;
% R = randperm(n)';
% c=0.8;


% Acc_MLP=zeros(10,1);
% SN_MLP=zeros(10,1);
% SP_MLP=zeros(10,1);
% F1_MLP=zeros(10,1);
% MCC_MLP=zeros(10,1);
% CM_MLP=zeros(10,4);
% data_datatotal=datatotal(:,1:end-1);
% data_label=datatotal(:,end);

k=10;
% index=crossvalind('kfold',size(datatotal,1),k);
% make ANN
performFcn = 'crossentropy';  % Cross-Entropy

trainFcn = 'trainscg';% Scaled conjugate gradient backpropagation.
net3 = patternnet([100 100],trainFcn,performFcn);
net3.divideParam.valRatio = 5/100;
view(net3)
%     TrainData=datatotal(round(R(1:c*n)),1:end-1);
%     lblTrain=datatotal(round(R(1:c*n)),end);
%     TestData=datatotal(R(c*n+1:end),1:end-1);
%     lblTest=datatotal(round(R(c*n+1:end)),end);
%     TrainData=TrainData';
%     TestData=TestData';
%     lblTest=lblTest';
%     lblTrain=lblTrain';
%       train1 = (index==i);     
%       test = ~train1;
%       f1=datatotal(train1,1:end-1);
%       f2=datatotal(train1,end);
    net3 = train(net3,TrainData,lblTrain);
    output = net3(TestData);
    output1=round(output);
    [cc,cm]=confusion(lblTest,output1)
    
%     CM_MLP(i,1)=cm(1,1);
%     CM_MLP(i,2)=cm(1,2);
%     CM_MLP(i,3)=cm(2,1);
%     CM_MLP(i,4)=cm(2,2);


    Acc_MLP=(cm(1,1)+cm(2,2))/(cm(1,1)+cm(1,2)+cm(2,1)+cm(2,2))
    SN_MLP=cm(1,1)/(cm(1,1)+cm(1,2))
    SP_MLP=cm(2,2)/(cm(2,2)+cm(2,1))
    F1_MLP = 2*cm(1,1)/(cm(1,2)+cm(2,1)+(2*cm(1,1)))
    MCC_MLP=((cm(2,2)*cm(1,1))-(cm(1,2)*cm(2,1)))/sqrt((cm(1,1)+cm(1,2))*(cm(1,1)+cm(2,1))*(cm(2,2)+cm(2,1))*(cm(2,2)+cm(1,2)))

 
     
% plotconfusion(lblTest,output1)
% [cc,cm]=confusion(lblTest,output1)
% figure;
% plot(output1,'r*');
% hold on;
% plot(lblTest,'go');
% legend('output','lblTest');










