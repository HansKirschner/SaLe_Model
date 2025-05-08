%% Script to fit a hybrid RWPH model to REFIT MOCA data
% using a constrained search algorithm (fmincon)
clear;
rootDir = pwd;
addpath(genpath(rootDir));
set(0,'defaultAxesFontSize',20);
fminconDir = fullfile(rootDir,'data/fminconResults');
savepath = fullfile(rootDir,'data');

% All parameters that can potentially influence decision behavior
% All parameters that can potentially influence decision behavior
allParams = {'invT','playBias','intcpt','CPP','RU','idealLR','RewardMagnitude','VisualSurprise','FeedConfrim','FeedReality'};

load('data/Data_wPrior_1110011100.mat')
load('data/fminconResult/fminconResult_wPrior_1110011100.mat')

paramset = {'invT','playBias','intcpt','idealLR','RewardMagnitude','VisualSurprise'};
%paramset = {'invT','playBias','intcpt','idealLR','RewardMagnitude'};


allParamSets = {paramset};

Input = dstruct;

for tt = 1:length(allParamSets)

    paramsToUse = allParamSets{tt};

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    paramsToFit    = false(length(allParams),1)';
    paramsToFit(ismember(allParams,paramsToUse)) = true;
    paramCount     = sum(paramsToFit);
    allParamsToFit = paramsToFit;
    allParamsToFit = logical(allParamsToFit);

    thisParamsBin = allParamsToFit;
    paramStr = '';
    for k = 1:length(thisParamsBin), paramStr = cat(2,paramStr,num2str(double(thisParamsBin(k)))); end

    for vp = 1:length(Input)

        vp
        simChoice=[];
        PC=[];

        for s = 1:100


            % generate synthetic data
            inputSym    = getBehavModelInput_Sale_sim(Input(vp),allParams(thisParamsBin),nan);
            SymData     = learningFunc_Sale_sim(fminconResult.fitParams(vp,:), inputSym);

            simChoice(s,:)  = SymData.choice;
            PC(s,:)         = SymData.ChoiceProb;
            BonusSum(vp,s)  = SymData.BounsSum;
        end
        
        SEM = std(simChoice)/sqrt(size(simChoice,1));           % Standard Error
        ts = tinv([0.00001  0.99999],size(simChoice,1)-1);      % T-Score
        CI = mean(simChoice)' + ts.*SEM';               % Confidence Intervals
        
        Input(vp).lowCIChoice       = CI(:,1);
        Input(vp).highCIChoice      = CI(:,2);

        Input(vp).minSimPC          = min(PC);
        Input(vp).maxSimPC          = max(PC);
        Input(vp).meanSimPC         = mean(PC);
        Input(vp).SimBonus_median   = median(BonusSum(vp,:));
        Input(vp).SimBonus_mean     = mean(BonusSum(vp,:));
        
        clear PC
    end
end

%% save data
save("data/Data_and_Sim_Sale.mat","Input")

%%
clear

load('data/Data_and_Sim_Sale.mat')
load('data/all_EndPOints.mat')
load('data/fminconResult/fminconResult_wPrior_1110011100.mat')

% Subject exclusion
Criterion = 200;
ExcludeBH = find(EndPOints <= Criterion);

EndPOints(ExcludeBH) = [];Input(ExcludeBH) = []; fminconResult.fitParams(ExcludeBH,:)=[];

% quantile split on magnitude LR bias - plot boht LR
y = quantile(fminconResult.fitParams(:,5),3);
y = median(fminconResult.fitParams(:,5));

LowBias     = find(fminconResult.fitParams(:,5)<y);
HighBias    = find(fminconResult.fitParams(:,5)>y);


D = Input;

for i = 1:length(D)


    D(i).ChoiceProb     = D(i).ChoiceProb';
    D(i).lowCIChoice    = D(i).lowCIChoice';
    D(i).highCIChoice   = D(i).highCIChoice';
    D(i).RL_LR          = D(i).RL_LR';

end

SymStage = [];
n = size(D,2);
for c =  1 : n
    US = D(c).SymStage'; If = 1; AllI = [];
    for c2 = 2 : length(US)
        if US(c2)<=US(c2-1) %end of stage
            if US(c2-1)>25
                
                AllI = [AllI If:c2-1];
            end
            If = c2;
        end
    end
    length(AllI);
    AllIAll(c) = {AllI};
    SymStage = [SymStage D(c).SymStage(AllI)'];
end
SymStage

% Change all fieldnames into equally size arrays
FN = fieldnames(Input); FN([1 17 31 32]) = [];
for c = 1 : length(FN)
    eval([FN{c} ' = [];'])
    for c2 = 1 : n
        eval([FN{c} ' = [' FN{c} ' D(c2).(FN{c})(AllIAll{c2})''];'])
    end
end

cols    = get(groot,'DefaultAxesColorOrder');
figure(1);clf;
subplot(2,1,1)
shadedErrBar([], AGF_running_average(mean(choice,2),2,2), AGF_running_average(se(choice',0.99),2,2),{'-o', 'color', cols(3,:), 'markerfacecolor',cols(3,:), 'markerSize', 1, 'lineWidth', 3}); hold;
plot(mean(Prob,2)/100,'LineWidth',2)
plot(AGF_running_average(mean(ChoiceProb,2),2,2),LineWidth=3);
xlim([1 length(AllI)])
set(gca,'xtick',[])
legend({'choice' 'GT' 'CP'},'FontSize',14);
set(0,'DefaultLegendAutoUpdate','off')
ylabel('Choice (avoid, chosen)','FontSize',14, 'FontWeight', 'bold')
gridxy(find([reversal(:,1)==1]))
gridxy([find(Stimno(:,1) == 2, 1 ) find(Stimno(:,1) == 3, 1 )],'linewidth',3,'Color','b');
plot(AGF_running_average(nanmean(lowCIChoice,2),2,2),'-.k',LineWidth=1);
plot(AGF_running_average(nanmean(highCIChoice,2),2,2),'-.k',LineWidth=1); 
set(gca,'FontSize',15);

subplot(2,1,2)
%shadedErrBar([], AGF_running_average(mean(RL_LR,2),2,2), AGF_running_average(se(RL_LR',0.99),2,2),{'-o', 'color', cols(4,:), 'markerfacecolor',cols(4,:), 'markerSize', 1, 'lineWidth', 3}); hold;
plot(AGF_running_average(mean(RL_LR(:,LowBias),2),2,2),LineWidth=3,Color=cols(3,:));hold on
plot(AGF_running_average(mean(RL_LR(:,HighBias),2),2,2),LineWidth=3,Color=cols(4,:));
xlim([1 length(AllI)])
set(gca,'xtick',[])
legend({'low bias' 'high bias'},'FontSize',14);
ylim([0.1 .25])
set(0,'DefaultLegendAutoUpdate','off')
ylabel('Learning rate','FontSize',14, 'FontWeight', 'bold')
gridxy(find([reversal(:,1)==1]))
gridxy([find(Stimno(:,1) == 2, 1 ) find(Stimno(:,1) == 3, 1 )],'linewidth',3,'Color','b');
set(gca,'FontSize',15);

h = gcf;
h.Position = [1         291        1094         710];
print2pdf(fullfile(pwd,'Figures',['ModelFitPostPreChecks.pdf']),h,300)
