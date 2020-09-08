function [Xtst Ytst Xtrn Ytrn Xvld Yvld ] = initData()

s     = 1400;
s_trn = 410;
%s_vld = 238;

data = load('data2.dat');
data = data(1:2048,:);

tau = 2;
dim = 3;
% X(:,1) = data(1     :end-tau*(dim-1),:)
% X(:,2) = data(tau+1 :end-tau*(dim-2),:)
% X(:,3) = data(2*tau+1:end-tau*(dim-3),:)

for i=1:dim
    X(:,i) = data(i*tau:end-tau*(dim-i),:);
end
%plot(X(:,1),X(:,3))
Xtst = X(1:s           ,1:end-1);
Ytst = X(1:s           ,end);
Xtrn = X(s+1:s+s_trn   ,1:end-1);
Ytrn = X(s+1:s+s_trn   ,end);
Xvld = X(s+s_trn+1:end ,1:end-1);
Yvld = X(s+s_trn+1:end ,end);