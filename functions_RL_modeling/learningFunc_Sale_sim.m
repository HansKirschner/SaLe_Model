function output = learningFunc_Sale_sim(params, input)  %input comes from fmincon (initially start_point)
 
%keyboard

% parameters:

% 1) inverse temperature
% 2) play bias
% 3) N parameters governing the effects of columns of matrix learningMat on
%    learning rate. First one should be intercept.

% First 2 parameters are fixed
invT= params(strcmp(input.paramNames,'invT'));
%invT = 3;

playBias= params(strcmp(input.paramNames,'playBias'));


% rest of parameters control flexible learning rate
LR_coeffWts=params(~ismember(input.paramNames,{'invT','playBias'}));

allLR       = nan(size(input.outcome));
expP        = zeros(size(input.outcome));
RPE         = nan(size(input.outcome));
logPChoice  = nan(size(input.outcome));
ChoiceProb  = nan(size(input.outcome))';

reward = ones(length(input.outcome),1)*10;
reward(input.outcome==0) = -10;

BonusSum = nan(size(input.outcome))';
points   = nan(size(input.outcome))';

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

    if input.pointEstFlag

        if input.choice(i) == 1 && input.RewardMagnitude(i) == 0
            points(i) = reward(i)*10;
        
        elseif input.choice(i) == 1 && input.RewardMagnitude(i) == 1
            points(i) = reward(i)*80;
        
        else
            points(i) = 0;
        end

        BonusSum(i) = sum(points(1:i));
    end

end

output = struct;
output.choice       = input.choice;
output.ChoiceProb   = ChoiceProb;
output.outcome      = input.outcome;
output.BounsSum     = BonusSum(end);
output.Prob         = input.Prob;
output.Stimno       = input.stimNr;
output.reversal             = input.reversal;
output.RewardMagnitude      = input.RewardMagnitude;
output.VisualSurprise       = input.VisualSurprise;
output.LR                   = input.LR;
output.IndexFirstShow       = input.IndexFirstShow;
output.magnitude            = input.RewardMagnitude;
output.flash                = input.VisualSurprise;
output.event                = input.event;
end
