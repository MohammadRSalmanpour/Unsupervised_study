function [Xtrn Ytrn Xvld Yvld Xtst Ytst ] = initNN5()
% close all;
% clear all;
% clc;
data = load ('data.dat');
index = find(isnan(data(:,1)));
data(index,1) = 0;
m = mean(data(:,1));
index = find(data(:,1)==0);
data(index,1) = m;

X = data(:,1);
% Xprd = data(end-55:end,1);
Xvld = [];
Yvld = [];
% for i=1:(size(X,1)-14)
%     Xtrn(i,:) = X(i:i+13,:);
%     Ytrn(i,:) = X(i+14);
% end
for i=1:size(X,1)-14
    Xtrn(i,:) = X(i:i+13,:);
    Ytrn(i,:) = X(i+14);
end
Ytst = Ytrn(end-55:end,:);
Xtst = Xtrn(end-55:end,:);
Xtrn = Xtrn(1:end-56,:);
Ytrn = Ytrn(1:end-56,:);