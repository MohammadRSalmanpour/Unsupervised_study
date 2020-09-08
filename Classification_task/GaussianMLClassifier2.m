function [TestResult,TrainResult]=GaussianMLClassifier2(X,Y,No_of_folds,i,p,o)
%%%https://www.researchgate.net/publication/308927930_Comparison_of_Feature_Reduction_Approaches_and_Classification_Approaches_for_Pattern_Recognition
%%%https://github.com/Xiaoyang-Rebecca/PatternRecognition_Matlab

%%%% https://www.researchgate.net/publication/308927930_Comparison_of_Feature_Reduction_Approaches_and_Classification_Approaches_for_Pattern_Recognition
 

No_of_class=max(Y);
InputNum=size(X,2);

data=[X Y];

MAXY=max(Y);


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
    Xtrn=Train_Validedatase(:,1:end-1)';
    Ytrn=Train_Validedatase(:,end)';
    
    test_datatest=cell2mat(test_data(K));
    Xtst= test_datatest(:,1:end-1)';
    Ytest=test_datatest(:,end)';
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
    
    %%%%%%%train process
  
  
[~  ,NoTrS] = size(Xtrn); 
[NoF,NoTeS] = size(Xtst);
[NoFtr,NoTeStr] = size(Xtrn);
Yh_Test=zeros(NoTeS,1);
Yh_Train=zeros(NoTeStr,1);
%Assume that the data is labelled in continuous integer: 1,2,3...,Noc
%NoC = max(Y_Train);   %Number of class(label)

%%If the data is not labelled in continuous integer,but 0,2,5,7,8...
% we use class()
class_label = min(Ytrn); 
i = 0; 
class = zeros(max(Ytrn),2);
for j= 1: max(Ytrn)
       class_temp = find( Ytrn == class_label); 	 % account the number 
       i = i+1; 									 % i to account the number of class
       class(i,1)= class_label  ;     				 % the true lable of class i 
       class(i,2)= length(class_temp ); 			 %the No.of elements in class i(labeled as class(i,1))
       class_label = class_label+1;   
end 
class(class(:,2)==0,:)=[]; 						     % delete the extra row if the number of this class is zero. 
NoC = length(class);

%%
%------------------Get the estimation value of training data--------------
Mu = zeros(NoF,NoC);  				%Number of Feature * Number of class
%Sigma = zeros(NoF,NoF,NoC); %covariance matrix for NoC pages
P_w = zeros(NoC,1);   				%Number of class * 1, storage the prior posibility of class i
for i = class(:,1)'    				%for class(1)
     Index_class = (Ytrn == i);  % all the index of ith class 
     X_Train_classi = (Xtrn(:,Index_class))';  %all the data of ith class 
	 
     % the output Mu(:,i), Sigma(:,:,i) are estimation value of training data of class i
     [Mu_temp, ~] = GaussianMLEstimator(X_Train_classi);
   
     Mu(:,i)= Mu_temp';
     P_w(i)=  class(i,2)/ NoTrS; %the number of data of class i in trainng data 
end
%[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[Differenc to GaussianMLClassifier3]]]]]]]]]]]]]]]]]]] 
%by lumping all the classes together,we can use the covariance from the global dataset (without class labels)
    [~, cov_mat] = GaussianMLEstimator(Xtrn');

%[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[Differenc to GaussianMLClassifier3]]]]]]]]]]]]]]]]]]]  




%%%%%%Train evaluation%%%%TRAIN Measurments  Accuracy, Specifisity, Sensivitivity, f_SCORE,.....
g_C2 = zeros(NoC,1) ;
for Idx_train = 1:(NoTeStr)     %for each data point in training sample
    %----------------------------------------------------------------------
    %-----Generate the classifier for class i ,g(i) is the distriminate----
    %funciton
    x = Xtrn(:,Idx_train);  % NoF*1
    for i = class(:,1)'       %Put in to the classifier of each class
 %[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[Differenc to GaussianMLClassifier3]]]]]]]]]]]]]]]]]]] 
        w= (cov_mat ^(-1))* Mu(:,i);
        w_0 = -1/2* Mu(:,i)'*(cov_mat ^(-1))* Mu(:,i) + log(P_w(i));
        g_C2(i)= w'* x + w_0;
%[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[Differenc to GaussianMLClassifier3]]]]]]]]]]]]]]]]]]] 
    end
   %-----------------------------------------------------------------------
   Yh_Train(Idx_train) = find(g_C2 == max(g_C2)); % if gi(x)>gj(x) (for all j ~=i) classify this data as class i

end




statsTrain = confusionmatStats(Yh_Train,Ytrn');
    
    Accurtrain(K,1) = statsTrain.accuracy;
    % Sensittrain(:,K) = statsTrain.sensitivity;
    % Specitrain(:,K) = statsTrain.specificity;
    % Fscoretrain(:,K) = statsTrain.Fscore;
    % Pesrcitrain(:,K) = statsTrain.precision;
    % Recalltrain(:,K) = statsTrain.recall;
    
    
%%%%%%Test evaluation%%%%TRAIN Measurments  Accuracy, Specifisity, Sensivitivity, f_SCORE,.....
 %----------------------- Put testing data into classifier------
% decide  wi(class i) 
%if        P(wi|x)>P(wj|x)          (for all j ~=i)
%=>  P(x|wi)*P(wi)>P(x|wj)*P(wj)     (for all j ~=i)

%introduce: gi(x)= ln(P(x|wi))+ln(P(wi))
%The final classifier is :
%    decide  wi(class i)  if  gi(x)>gj(x) (for all j ~=i)

%for Gaussian case,  we use log likelihood discriminant functions g(i)
g_C2 = zeros(NoC,1) ;
for Idx_test = 1:(NoTeS)     %for each data point in training sample
    %----------------------------------------------------------------------
    %-----Generate the classifier for class i ,g(i) is the distriminate----
    %funciton
    x = Xtst(:,Idx_test);  % NoF*1
    for i = class(:,1)'       %Put in to the classifier of each class
 %[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[Differenc to GaussianMLClassifier3]]]]]]]]]]]]]]]]]]] 
        w= (cov_mat ^(-1))* Mu(:,i);
        w_0 = -1/2* Mu(:,i)'*(cov_mat ^(-1))* Mu(:,i) + log(P_w(i));
        g_C2(i)= w'* x + w_0;
%[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[Differenc to GaussianMLClassifier3]]]]]]]]]]]]]]]]]]] 
    end
   %-----------------------------------------------------------------------
   Yh_Test(Idx_test) = find(g_C2 == max(g_C2)); % if gi(x)>gj(x) (for all j ~=i) classify this data as class i

end


   
    statsTest = confusionmatStats(Ytest,Yh_Test');
    
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

end %function

