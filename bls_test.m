function [pred] = bls_test(test_x, N1, N2, beta11, ps , wh, l2, beta)
% Learning Process of the proposed broad learning system
%Input: 
%---train_x,test_x : the training data and learning data 
%---train_y,test_y : the label 
%---We: the randomly generated coefficients of feature nodes
%---wh:the randomly generated coefficients of enhancement nodes
%----s: the shrinkage parameter for enhancement nodes
%----C: the regularization parameter for sparse regualarization
%----N11: the number of feature nodes  per window
%----N2: the number of windows of feature nodes

load bboxes.mat;
%%%%%%%%%%%%%%%%%%%%%%Testing Process%%%%%%%%%%%%%%%%%%%
test_x = zscore(test_x');
HH1 = [test_x .1 * ones(size(test_x,1),1)];
size(HH1)
%clear test_x;
yy1=zeros(size(test_x,1),N2*N1);
for i=1:N2
    beta1=beta11{i};ps1=ps(i); %%% save them
    TT1 = HH1 * beta1;
    TT1  =  mapminmax('apply',TT1',ps1)';

clear beta1; clear ps1;
%yy1=[yy1 TT1];
yy1(:,N1*(i-1)+1:N1*i)=TT1;
end
clear TT1;clear HH1;
HH2 = [yy1 .1 * ones(size(yy1,1),1)]; 
TT2 = tansig(HH2 * wh * l2);TT3=[yy1 TT2];
clear HH2;clear TT2;

%%%%%%%%%%%%%%%%% testing accuracy%%%%%%%%%%%%%%%%%%%%%%%%%%%



x = TT3 * beta;
pred = result(x);
% test_yy = result(test_y);
% save('test_yy')
% TestingAccuracy = length(find(y == test_yy))/size(test_yy,1);
clear TT3;
