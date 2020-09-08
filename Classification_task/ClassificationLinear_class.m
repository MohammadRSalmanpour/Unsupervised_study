function [TestResult,TrainResult]=ClassificationLinear_class(X,Y,No_of_folds,i,p,o)

No_of_class=max(Y);
InputNum=size(X,2);

data=[X Y];


Accurtrain= zeros(No_of_folds,1);
Sensittrain = zeros(No_of_class,No_of_folds);
Specitrain = zeros(No_of_class,No_of_folds);
Fscoretrain = zeros(No_of_class,No_of_folds);
Pesrcitrain =zeros(No_of_class,No_of_folds);
Recalltrain =zeros(No_of_class,No_of_folds);

Accurtest = zeros(No_of_folds,1);
Sensittest =zeros(No_of_class,No_of_folds);
Specitest = zeros(No_of_class,No_of_folds);
Fscoretest = zeros(No_of_class,No_of_folds);
Pesrcitest =zeros(No_of_class,No_of_folds);
Recalltest = zeros(No_of_class,No_of_folds);

[test_data,train_data] = KFoldCrossValidation(data,No_of_folds);
for K =1 : No_of_folds
    
 
    Train_Validedata=train_data(K);
    Train_Validedatase=cell2mat(Train_Validedata);
    Xtrn=Train_Validedatase(:,1:end-1);
    Ytrn=Train_Validedatase(:,end);
    
    test_datatest=cell2mat(test_data(K));
    Xtst= test_datatest(:,1:end-1);
    Ytest=test_datatest(:,end);
    ctr=size(Xtrn,2);
    for l=1:ctr
        if std( Xtrn(:,l))==0
            Xtrn(1,l)=Xtrn(1,l)+( 2);
        end
    end
    cte=size(Xtst,2);
    
    for l=1:cte
        if std( Xtst(:,l))==0
            Xtst(1,l)=Xtst(1,l)+(2);
        end
    end

%     Ytrn1 =Ytrn == 'stats';
    Mdl = fitclinear(Xtrn,Ytrn);
    yhtrn = predict(Mdl,Xtrn);

    %%%%%%%%TRAIN Measurments  Accuracy, Specifisity, Sensivitivity, f_SCORE,.....
    statsTrain = confusionmatStats(Ytrn,round(yhtrn));
    
    Accurtrain(K,1) = statsTrain.accuracy;
    % Sensittrain(:,K) = statsTrain.sensitivity;
    % Specitrain(:,K) = statsTrain.specificity;
    % Fscoretrain(:,K) = statsTrain.Fscore;
    % Pesrcitrain(:,K) = statsTrain.precision;
    % Recalltrain(:,K) = statsTrain.recall;
    
    
    % test step
%  Yh11 =Yh1 == 'stats';
    Yh1 = predict(Mdl,Xtst);
    %%%%%%%%TEST Measurments  Accuracy, Specifisity, Sensivitivity, f_SCORE,.....
    statsTest = confusionmatStats(Ytest,round(Yh1));
    
    %%%%%%%%%%%%Filling measurements%%%%%%%%%%%%%%%%%
    Accurtest(K,1) = statsTest.accuracy;
    % Sensittest(:,K) = statsTest.sensitivity;
    % Specitest(:,K) = statsTest.specificity;
    % Fscoretest(:,K) = statsTest.Fscore;
    % Pesrcitest(:,K) = statsTest.precision;
    % Recalltest(:,K) = statsTest.recall;
    
end%%%For for Kfold

TestResult.Accurtest=(Accurtest);
% TrainResult.Sensittest=(Sensittest);
% TrainResult.specificity=(Specitrain);
% TrainResult.Fscore=(Fscoretest);
% TrainResult.precision=(Pesrcitest);
% TrainResult.recall=(Recalltest);


TrainResult.Accurtrain=(Accurtrain);
% TestResult.Sensittest=(Sensittrain);
% TestResult.specificity=(specificitytarin);
% TestResult.Fscore=(Fscoretrain);
% TestResult.precision=(Pesrcitrain);
% TestResult.recall=(Recalltrain);

close all;
end %function

