function [negLogLike, allLR, expP, ChoiceProb, input, logLike_woPrior, RPE] = learningFunc_Sale(params, input)  %input comes from fmincon (initially start_point)
 
%keyboard

minLogLike=-10e10;

% parameters:

% 1) inverse temperature
% 2) play bias
% 3) N parameters governing the effects of columns of matrix learningMat on
%    learning rate. First one should be intercept.

% First 2 parameters are fixed
invT= params(strcmp(input.paramNames,'invT'));

playBias= params(strcmp(input.paramNames,'playBias'));

intcpt = params(strcmp(input.paramNames,'intcpt'));

idealLR = params(strcmp(input.paramNames,'idealLR'));

RewardMagnitude = params(strcmp(input.paramNames,'RewardMagnitude'));

VisualSurprise = params(strcmp(input.paramNames,'VisualSurprise'));

% rest of parameters control flexible learning rate
LR_coeffWts=params(~ismember(input.paramNames,{'invT','playBias'}));

allLR=nan(size(input.outcome));
expP=zeros(size(input.outcome));
RPE=nan(size(input.outcome));
logPChoice=nan(size(input.outcome));
ChoiceProb = nan(size(input.outcome))';

reward = ones(length(input.outcome),1)*10;
reward(input.outcome==0) = -10;

for i = 1:length(input.outcome)
    
    % if this is a new stimuli restet trialProb
    if input.reversal(i)==1
        expP(i)=0;
    end
       
    % If the subject is biased toward playing then they could be described as attributing a higher value
    % to the play option.

    biasedTrialVs=[playBias+expP(i),0];

    % first column computes bet probability, second column = pass
    [pBet_pass, ~]=softMax(biasedTrialVs, invT);
    
    ChoiceProb(i)=pBet_pass(1);

    if input.genDataFlag==true
       input.choice(i)=rand>pBet_pass(2);
    end
    
    % Want this to be close to 0
    % See if the subject made the higher probability choice. If subject
    % chose the higher prob option, logPChoice will be smaller negative
    % value.
    logPChoice(i)=log(pBet_pass( (input.choice(i)==0) +1));
    

    LR_term=input.learningMat(i,:)*LR_coeffWts'; % input.learningMat should have column of ones as first column
    LR=exp(LR_term)./(1+exp(LR_term)); %map unbounded LR_term onto to [0 1] interval
    allLR(i)=LR;
    
    % compute feedback: i.e. delta rule
    % update according to learning rate times prediction error
    % This should be the outcome of the trial that the subject saw.
    newTrialProb= expP(i)+LR.*(reward(i)-expP(i));

    RPE(i)= reward(i)-expP(i);

    % stick updated probability in appropriate direction:

    expP(i+1)= newTrialProb;

end


logPChoice(logPChoice<minLogLike)=minLogLike;
logPChoice(~isfinite(logPChoice))=minLogLike;

logLike_woPrior=nansum(logPChoice);

%keyboard

if input.usePrior==true
    allPriors = nan(1,length(input.paramNames));
    for p = 1:length(input.paramNames)
        thisParamStr = input.paramNames{p};
        if strcmp(thisParamStr,'invT'), paramValue = invT;
        elseif strcmp(thisParamStr,'playBias'), paramValue = playBias;
        elseif strcmp(thisParamStr,'intcpt'), paramValue = intcpt;
        elseif strcmp(thisParamStr,'idealLR'), paramValue = idealLR;
        elseif strcmp(thisParamStr,'RewardMagnitude'), paramValue = RewardMagnitude;
        elseif strcmp(thisParamStr,'VisualSurprise'), paramValue = VisualSurprise;        
        else
            paramValue = params(p);
        end
        
        inputForPrior = getBehavModelInput_Sale(input.datastruct,input.paramNames,paramValue);
        allPriors(p) = inputForPrior.priorPdf(strcmp(input.paramNames,thisParamStr));
        allPriors(p) = max([allPriors(p),10e-10]);
    end
    
    totLogPrior = sum(log(allPriors));
else
    totLogPrior=1;
end

% log posterior probability is log prior plus log likelihood:
postProb = logLike_woPrior+totLogPrior;
negLogLike =-postProb;


if negLogLike==0
    keyboard
end


if ~isfinite(negLogLike)
    disp('something fishy happened');
    keyboard
end



end
