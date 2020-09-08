function [X Y X2 Y2] = initData1()
    X  = 0:.05:2;
    X  = X';
    Y  = (sin(-X)+1);
    X2 = (2*rand(1,25))';
    Y2 = sin(-X2)+1;
    
    s1 = size(X);
    index = randint(1,s1(1,1),[1 s1(1,1)]);
    X     = X(index,:);
    Y     = Y(index,:);
    s1 = size(X2);
    index = randint(1,s1(1,1),[1 s1(1,1)]);
    X2     = X2(index,:);
    Y2     = Y2(index,:);    
end