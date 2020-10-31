function stats = confusionmatStats(group,grouphat)
2 % http://www.mathworks.com/matlabcentral/fileexchange/46035-confusion-matrix--accuracy--precision--specificity--sensitivity--recall--f-score
3 %
4 % INPUT
5 % group = true class labels
6 % grouphat = predicted class labels
7 %
8 % OR INPUT
9 % stats = confusionmatStats(group);
10 % group = confusion matrix from matlab function (confusionmat)
11 %
12 % OUTPUT
13 % stats is a structure array
14 % stats.confusionMat
15 %               Predicted Classes
16 %                    p'    n'
17 %              ___|_____|_____|
18 %       Actual  p |     |     |
19 %      Classes  n |     |     |
20 %
21 % stats.accuracy = (TP + TN)/(TP + FP + FN + TN) ; the average accuracy is returned
22 % stats.precision = TP / (TP + FP)                  % for each class label
23 % stats.sensitivity = TP / (TP + FN)                % for each class label
24 % stats.specificity = TN / (FP + TN)                % for each class label
25 % stats.recall = sensitivity                        % for each class label
26 % stats.Fscore = 2*TP /(2*TP + FP + FN)            % for each class label
27 %
28 % TP: true positive, TN: true negative,
29 % FP: false positive, FN: false negative
30 


field1 = 'confusionMat';
if nargin < 2
    value1 = group;
else
    value1 = confusionmat(group,grouphat);
end


numOfClasses = size(value1,1);
totalSamples = sum(sum(value1));

field2 = 'accuracy';  value2 = (2*trace(value1)+sum(sum(2*value1)))/(numOfClasses*totalSamples);


[TP,TN,FP,FN,sensitivity,specificity,precision,f_score] = deal(zeros(numOfClasses,1));
for class = 1:numOfClasses
    TP(class) = value1(class,class);
    tempMat = value1;
    tempMat(:,class) = []; % remove column
    tempMat(class,:) = []; % remove row
    TN(class) = sum(sum(tempMat));
    FP(class) = sum(value1(:,class))-TP(class);
    FN(class) = sum(value1(class,:))-TP(class);
end


for class = 1:numOfClasses
    sensitivity(class) = TP(class) / (TP(class) + FN(class));
    specificity(class) = TN(class) / (FP(class) + TN(class));
    precision(class) = TP(class) / (TP(class) + FP(class));
    f_score(class) = 2*TP(class)/(2*TP(class) + FP(class) + FN(class));
end


field3 = 'sensitivity';  value3 = sensitivity;
field4 = 'specificity';  value4 = specificity;
field5 = 'precision';  value5 = precision;
field6 = 'recall';  value6 = sensitivity;
field7 = 'Fscore';  value7 = f_score;
stats = struct(field1,value1,field2,value2,field3,value3,field4,value4,field5,value5,field6,value6,field7,value7);
