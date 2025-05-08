function [datastruct_nd,datastruct_24] = initializeCpmScript(whichExp)
%Initialization to run before every script.

set(0,'defaultAxesFontSize',20);

% Load data
if whichExp==1
    load('allData1.mat');
elseif whichExp==2
    load('allData2.mat');
end

end

