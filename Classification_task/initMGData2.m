function [ Xtrn Ytrn Xvld Yvld Xtest Ytest] = initMGData2(PC)
%data = PC;
data = load('mgdata.dat');
s = size(data);
dim = 4;                %%The number of dimensions of input
distance = 10;           %% The distance between steps
endstep = 1+distance*dim;
%%Generating 4D Mackey Glass data.
for i=1:s(1,1)-endstep
    k = 1;
    for ii=1:distance:endstep
        X(i,k) = data((i-1)+ii,2);
        k = k+1;
    end
end

 %h = ones(size(X,1),1);
 %X = (X-h*mean(X))./(h*var(X).^.5);

Xtrn    = X(1:200,1:dim);
Ytrn    = X(1:200,dim+1);

Xvld    = X(501:600,1:dim);
Yvld    = X(501:600,dim+1);

Xtest   = X(601:700,1:dim);
Ytest   = X(601:700,dim+1);