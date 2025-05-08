function [LR, UP, PE]=computeLR(outcomes, predictions, newBlock)

%keyboard
%% find the last trial of each block
a=find(newBlock)-1;
a=a(a>0);

UP=nan(length(outcomes),1);    
UP(1:end-1)=predictions(2:end)-predictions(1:end-1);%predicted outcome t+1 - predicted outcome t
PE=outcomes-predictions;
LR=(UP./PE);%added abs as there was the possibility of neg LR

UP(a)=nan;%why is this set to NaN
PE(a)=nan;
LR(a)=nan;
