function [TestResult,TrainResult]=GMMClassifier(X,Y,No_of_folds,i,p,o)

%%%https://www.researchgate.net/publication/308927930_Comparison_of_Feature_Reduction_Approaches_and_Classification_Approaches_for_Pattern_Recognition
%%%https://github.com/Xiaoyang-Rebecca/PatternRecognition_Matlab

%%%% https://www.researchgate.net/publication/308927930_Comparison_of_Feature_Reduction_Approaches_and_Classification_Approaches_for_Pattern_Recognition

No_of_class=max(Y);
InputNum=size(X,2);

data=[X Y];

MAXY=max(Y);
kk=MAXY;

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
%     ctr=size(Xtrn,1);
%     for l=1:ctr
%         if std( Xtrn(:,l))==0
%             Xtrn(1,l)=Xtrn(1,l)+( 2);
%         end
%     end
%     cte=size(Xtst,2);
%     
%     for l=1:cte
%         if std( Xtst(:,l))==0
%             Xtst(1,l)=Xtst(1,l)+(2);
%         end
%     end
    
    %%%%%%%train process
  
  
[~  ,NoTrS] = size(Xtrn'); 
[NoF,NoTeS] = size(Xtst');
[NoFtr,NoTeStr] = size(Xtrn');
Yh_Test=zeros(NoTeS,1);
Yh_Train=zeros(NoTeStr,1);
%Assume that the data is labelled in continuous integer: 1,2,3...,Noc
%NoC = max(Y_Train);   %Number of class(label)

%%If the data is not labelled in continuous integer,but 0,2,5,7,8...
% we use class()
% [NoTeS,NoF]=size(TestSample);
Yh_train=zeros(NoTeStr,1);
Yh_test=zeros(NoTeS,1);
%%
%fit GMM to trainsample,generate parameter for each clusters in each class
for i= unique(Ytrn)' %i index of class

    x= (Xtrn((Ytrn==i),:))';
    %for Samples in each class, implement GMM to get parameters
    
    MODEL=gmm(x,kk); 
        
    field.para(i)=MODEL;
    %  - PX: N-by-K matrix indicating the probability of each
    %       component generating each point.(responsibility matrix)
    %  [~, cls_ind] = max(px,[],2); %cls_ind = cluster label  
    %  - MODEL: a structure containing the parameters for a GMM:
    %       MODEL.Miu: a D by K matrix.
    %       MODEL.Sigma: a D-by-D-by-K matrix.
    %       MODEL.Pi: a 1-by-K vector. (Posterior probability of each component  )
end    
NoClass=i;


%%
%Train prediction and evaluation
%Method 1  Maximum Likelihood
    for Idx_train = 1:NoFtr     %for each data point in training sample
            %----------------------------------------------------------------------
            %-----Generate the classifier for class i ,g(i) is the distriminate----
            %funciton
          x = Xtrn(Idxtrain,:)';  % NoF*1
          g=zeros(NoClass,kk);
          
          classlabel=0;
          maxg=-inf;
          for i= 1: NoClass%i index of class
              for j=1:kk
                  Mu   =field.para(i).Miu; 
                  Sigma=field.para(i).Sigma;% a D-by-D-by-K matrix.   
                  P_w  = field.para(i).Pi;   % : a 1-by-K vector. (Posterior probability of each component  )
                  % sum up the posibility of all components in this class
                  g(i,j)= (-0.5* (x - Mu(:,j))'*(Sigma(:,:,j)^(-1))*(x - Mu(:,j))- NoF/2*log(2*pi)-0.5*log(det(Sigma(:,:,j)))+log(P_w(j)));
                  %g(i,j)= log(mvnpdf(x,Mu(:,j),(Sigma(:,:,j)))+log(P_w(j)));
                  if g(i,j)>maxg
                    maxg=g(i,j);
                    classlabel=i;
                  end
              end
          end
          Yh_train(Idx_train) = classlabel;
    end 

   
statsTrain = confusionmatStats(Yh_train,Ytrn');
    
    Accurtrain(K,1) = statsTrain.accuracy;
    % Sensittrain(:,K) = statsTrain.sensitivity;
    % Specitrain(:,K) = statsTrain.specificity;
    % Fscoretrain(:,K) = statsTrain.Fscore;
    % Pesrcitrain(:,K) = statsTrain.precision;
    % Recalltrain(:,K) = statsTrain.recall; 
    
    
    
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%Test Evaluation




%%
%predict labels for testing samples

%Method 1  Maximum Likelihood
    for Idx_test = 1:NoTeS     %for each data point in training sample
            %----------------------------------------------------------------------
            %-----Generate the classifier for class i ,g(i) is the distriminate----
            %funciton
          x = Xtst(Idx_test,:)';  % NoF*1
          g=zeros(NoClass,Kk);
          
          classlabel=0;
          maxg=-inf;
          for i= 1: NoClass%i index of class
              for j=1:Kk
                  Mu   =field.para(i).Miu; 
                  Sigma=field.para(i).Sigma;% a D-by-D-by-K matrix.   
                  P_w  = field.para(i).Pi;   % : a 1-by-K vector. (Posterior probability of each component  )
                  % sum up the posibility of all components in this class
                  g(i,j)= (-0.5* (x - Mu(:,j))'*(Sigma(:,:,j)^(-1))*(x - Mu(:,j))- NoF/2*log(2*pi)-0.5*log(det(Sigma(:,:,j)))+log(P_w(j)));
                  %g(i,j)= log(mvnpdf(x,Mu(:,j),(Sigma(:,:,j)))+log(P_w(j)));
                  if g(i,j)>maxg
                    maxg=g(i,j);
                    classlabel=i;
                  end
              end
          end
         Yh_test(Idx_test) = classlabel;
    end 
statsTest = confusionmatStats(Ytest,Yh_test');
    
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

