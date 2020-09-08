function [C I LB M UB V W] = FLoLiMoT(Xtrn, Ytrn, Xvld, Yvld,ncount)
%Offline Flat LoLiMoT ver 0.15
%This version includes RLS for estimating LLM parameters.
%Input arguments:
%   Xtrn: Training input data set.
%   Ytrn: Training desigred output data set.
%   Xvld: Input validation data set.
%   Yvld: Output validation data set.
%Output arguments:
%   M: Number of generated neurons.
%   C:
%   I:
%   LB:
%   UB:
%   V:
%   W:
%%
    global M1;
    M    = 1;            %We have one universal approximator at zeroth itteration.
    p    = size(Xtrn,2); %Input dimension size.
    n1   = size(Xtrn,1);
    n2   = size(Xvld,1);
    tmp1 = [];
    tmp2 = [];
    %% total structure of the network.
    C   = [];
    I   = [];
    LB  = [];
    UB  = [];
    V   = [];
    W   = [];
    %% structure of each division.
    C1  = [];
    C2  = [];
    I1  = [];
    I2  = [];
    LB1 = [];
    LB2 = [];
    UB1 = [];
    UB2 = [];
    V1  = [];
    V2  = [];
    W1  = [];
    W2  = [];
    P   = {};
    %% getting upper bound and lower bound.
    for i=1:p
        LB(1,i) = min(Xtrn(:,i));
        UB(1,i) = max(Xtrn(:,i));
    end
    %% initializing the first neuron.
    C(M,:)              = (LB+UB)./2;
    V(M,:)              = (UB-LB)./2;
    M1                  = phi(Xtrn,C(M,:),V(M,:));

    %W(M,:)              = LE (M1,Xtrn,Ytrn);
    
    [W(M,:) P{1,1}]     = RLE(M1,Xtrn,Ytrn);
    I(M,1)              = LLS(M1,Xtrn,Ytrn,W);
%     celldisp(P);
    while(M<ncount)
        [Iworst lworst] = max(I);
        UBt             = UB(lworst,:);
        LBt             = LB(lworst,:);
        C (lworst,:)    = [];
        I (lworst,:)    = [];
        LB(lworst,:)    = [];
        UB(lworst,:)    = [];
        V (lworst,:)    = [];
        W (lworst,:)    = [];
        for ii=1:p
            UB2(ii,:)           = UBt;
            LB2(ii,:)           = [LBt(1,1:ii-1) ((UBt(1,ii)+LBt(1,ii))/2) LBt(1,ii+1:end)];
            UB1(ii,:)           = [UBt(1,1:ii-1) ((UBt(1,ii)+LBt(1,ii))/2) UBt(1,ii+1:end)];
            LB1(ii,:)           = LBt;
            [C1(ii,:) V1(ii,:)] = initNeuron(LB1(ii,:), UB1(ii,:));
            [C2(ii,:) V2(ii,:)] = initNeuron(LB2(ii,:), UB2(ii,:));
            M1                  = phi(Xtrn,[C ; C1(ii,:) ; C2(ii,:)],[V; V1(ii,:);V2(ii,:)]);
            W1(ii,:)            = RLE (M1(:,end-1),Xtrn,Ytrn);
            W2(ii,:)            = RLE (M1(:,end  ),Xtrn,Ytrn);
            I1(ii,:)            = LLS(M1,Xtrn,Ytrn,[W; W1(ii,:); W2(ii,:)]);
        end
        [Ibest lbest] = min(I1);
        C (M,:)     = C1 (lbest,:);
        LB(M,:)     = LB1(lbest,:);
        UB(M,:)     = UB1(lbest,:);
        V (M,:)     = V1 (lbest,:);
        W (M,:)     = W1 (lbest,:);
        I1          = LLS(M1(:,end-1),Xtrn,Ytrn,W1(lbest,:));
        I (M,:)     = I1;
        M           = M+1;
        C (M,:)     = C2 (lbest,:);
        LB(M,:)     = LB2(lbest,:);
        UB(M,:)     = UB2(lbest,:);
        V (M,:)     = V2 (lbest,:);
        W (M,:)     = W2 (lbest,:);
        I2          = LLS(M1(:,end)  ,Xtrn,Ytrn,W2(lbest,:));
        I (M,:)     = I2;
%%        
        Yh_trn      = netFeed(C,V,W,Xtrn);
        etrn        = mse(Ytrn-Yh_trn);
        tmp1        = [tmp1 etrn];
        Yh_vld      = netFeed(C,V,W,Xvld);
        evld        = mse(Yvld-Yh_vld);
        tmp2        = [tmp2 evld];
%         hold on
%         plot(tmp1,'b.-');
%         plot(tmp2,'r*-');
%         legend('Training error','Validation error');
%         xlabel('Number of neurons.');
%         ylabel('Mean squared error.');    
%         hold off
%         drawnow();
%         
%         fprintf('MSE train:%d,\nMSE validation:%d,\nevld/etrn:%d\n',etrn,evld,evld/etrn);
%         fprintf('%d Neurons\n===========================\n',M)        
%% calculating SMAPE(Symmetric Mean Absolute Percent Error).
%         Yh_trn      = netFeed(C,V,W,Xtrn);
%         etrn        = 200/n1*sum(abs(Ytrn-Yh_trn)./(Ytrn+Yh_trn));
%         tmp1        = [tmp1 etrn];
%         Yh_vld      = netFeed(C,V,W,Xvld);
%         evld        = 200/n2*sum(abs(Yvld-Yh_vld)./(Yvld+Yh_vld));        
%         tmp2        = [tmp2 evld];
%         hold on
%         plot(tmp1,'b.-');
%         plot(tmp2,'r*-');
%         legend('Training error');
%         xlabel('Number of neurons.');
%         ylabel('SMAPE.');    
%         hold off
%         drawnow();
%         fprintf('MSE train:%d,\nMSE validation:%d,\nevld/etrn:%d\n',etrn);
%         fprintf('%d Neurons\n===========================\n',M)
%% NSME
%         Yh_trn      = netFeed(C,V,W,Xtrn);
%         etrn        = sum((Ytrn-Yh_trn).^2)/sum((Ytrn-mean(Ytrn)).^2);
%         tmp1        = [tmp1 etrn];
% %         Yh_vld      = netFeed(C,V,W,Xvld);
% %         evld        = sum((Yvld-Yh_vld).^2)/sum((Yvld-mean(Yvld)).^2);
% %         tmp2        = [tmp2 evld];
%         hold on
%         plot(tmp1,'b.-');
% %         plot(tmp2,'r*-');
%         legend('Training error');%,'Validation error');
%         xlabel('Number of neurons.');
%         ylabel('Mean squared error.');    
%         hold off
%         drawnow();
%         
%         fprintf('MSE train:%d,\nMSE validation:%d,\nevld/etrn:%d\n',etrn);
%         fprintf('%d Neurons\n===========================\n',M);
        
    end
end