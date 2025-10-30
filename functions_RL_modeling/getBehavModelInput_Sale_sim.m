function input = getBehavModelInput_Sale_sim(datastruct,paramsToFit,paramValue)
% keyboard
% Function to generate the input for modeling behavior
% Input
% datastruct
% paramsToFit: cell array of strings
% paramValue: if you want to get prior pdf of a certain parameter

input = struct;
input.datastruct = datastruct;
input.paramNames = paramsToFit;

input.outcome           = datastruct.outcome;
input.choice            = datastruct.choice;
input.reversal          = datastruct.IndexFirstShow;
input.Prob              = datastruct.Prob;
input.stimNr            = datastruct.Stimno;
input.LR                = datastruct.LR;
input.IndexFirstShow    = datastruct.IndexFirstShow;
input.event             = datastruct.event;
input.RewardMagnitude   = datastruct.magnitude;
input.VisualSurprise    = datastruct.flash;

% Feedback Confirmation - 0 = disconfirmational feedback; 1 =
% confirmational feedback
input.feedConfir = ones(length(datastruct.outcome),1);
input.feedConfir(datastruct.outcome==1&datastruct.choice==0) = 0;
input.feedConfir(datastruct.outcome==0&datastruct.choice==1) = 0;


input.usePrior = true;

input.pointEstFlag = true;
input.genDataFlag  = true;

% get new start point from prior distribution for each parameter
input.newStartPoint = [];
input.priorPdf = [];

input.learningMat = [];

% Use these values for all other LR parameters
general_randPriorVal = normrnd(0, 5);
general_priorPdf=pdf('norm', paramValue, 0, 5);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% PARAMETERS:

for p = 1:length(paramsToFit)
    
    % Inverse temperature
    if strcmp(paramsToFit{p},'invT')
        input.startPoint(:,p) = 1;
        input.lb(:,p) = 0;
        input.ub(:,p) = 100;
        input.newStartPoint(p) = gamrnd(2, 1);
        input.priorPdf(p) = pdf('Gamma',paramValue, 2, 1);
               
    % Play bias
    elseif strcmp(paramsToFit{p},'playBias')
        input.startPoint(:,p) = 0;
        input.lb(:,p) = -200;
        input.ub(:,p) = 100;
        input.newStartPoint(p) = normrnd(0, 10);
        input.priorPdf(p) = pdf('norm',paramValue, 0, 3);


    % LR intercept
    elseif strcmp(paramsToFit{p},'intcpt')
        input.learningMat(:,p) = ones(length(datastruct.CPP),1);
        input.startPoint(:,p) = 0;
        input.lb(:,p) = -100;
        input.ub(:,p) = 100;
        input.newStartPoint(p) = normrnd(0, 5);
        input.priorPdf(p) = pdf('norm', paramValue, 0, 5);
        
    % CPP
    elseif strcmp(paramsToFit{p},'CPP')
        input.learningMat(:,p) = datastruct.CPP;
        input.startPoint(:,p) = 0;
        input.lb(:,p) = -100;
        input.ub(:,p) = 100;
        input.newStartPoint(p) = general_randPriorVal;
        input.priorPdf(p) = general_priorPdf;
        
    % RU
    elseif strcmp(paramsToFit{p},'RU')
        input.learningMat(:,p) = datastruct.RU;
        input.startPoint(:,p) = 0;
        input.lb(:,p) = -100;
        input.ub(:,p) = 100;
        input.newStartPoint(p) = general_randPriorVal;
        input.priorPdf(p) = general_priorPdf;
        
    % ideal LR
    elseif strcmp(paramsToFit{p},'idealLR')
        input.learningMat(:,p) = datastruct.LR;
        input.startPoint(:,p) = 0;
        input.lb(:,p) = -100;
        input.ub(:,p) = 100;
        
        input.newStartPoint(p) = normrnd(1, 5);
        input.priorPdf(p) = pdf('norm', paramValue, 1, 5);
   
%      input.newStartPoint(p) = gamrnd(2, 1);
%      input.priorPdf(p) = pdf('Gamma',paramValue, 2, 1);  

    % Reward Magnitude
    elseif strcmp(paramsToFit{p},'RewardMagnitude')
        input.learningMat(:,p) = input.RewardMagnitude;
        input.startPoint(:,p) = 0;
        input.lb(:,p) = -100;
        input.ub(:,p) = 100;
   
        input.newStartPoint(p) = normrnd(3, 5);
        input.priorPdf(p) = pdf('norm', paramValue, 3, 3);


%       input.newStartPoint(p) = gamrnd(4, 1);
%       input.priorPdf(p) = pdf('Gamma',paramValue, 4, 1);
    
    % Visual Surprise
    elseif strcmp(paramsToFit{p},'VisualSurprise')
        input.learningMat(:,p) = input.VisualSurprise;
        input.startPoint(:,p) = 0;
        input.lb(:,p) = -100;
        input.ub(:,p) = 100;
        input.newStartPoint(p) = normrnd(0, 5);
        input.priorPdf(p) = pdf('norm', paramValue, 0, 5);
        
    % Feed Confrimation
    elseif strcmp(paramsToFit{p},'FeedConfrim')
        input.learningMat(:,p) = input.feedConfir;
        input.startPoint(:,p) = 0;
        input.lb(:,p) = -100;
        input.ub(:,p) = 100;
        input.newStartPoint(p) = general_randPriorVal;
        input.priorPdf(p) = general_priorPdf;
    
    % Feed Reality
    elseif strcmp(paramsToFit{p},'FeedReality')
        input.learningMat(:,p) = input.choice;
        input.startPoint(:,p) = 0;
        input.lb(:,p) = -100;
        input.ub(:,p) = 100;
        input.newStartPoint(p) = general_randPriorVal;
        input.priorPdf(p) = general_priorPdf;
                
    end
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% z score learning rate explanatory matrix
input.learningMat(:,1:2) = [];
input.learningMat=zScoreX(input.learningMat);