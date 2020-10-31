function [X Y Xtrn Ytrn Xvld Yvld ] = initMGData()

data = load('mgdata.dat');
%%Generating 4D Mackey Glass data.
for i=1:240
    for ii=1:3
        
        X(i,ii)     = data(i+(ii-1)*(ii-1),2);
        X(i,ii+1)   = data(i+8,2);
    end
end
h = ones(size(X,1),1);
X = (X-h*mean(X))./(h*var(X).^.5);

Xtrn    = X(1:180,1:3);
Ytrn    = X(1:180,4);

Xvld    = X(181:240,1:3);
Yvld    = X(181:240,4);

Y = X(:,4);
X = X(:,1:3);