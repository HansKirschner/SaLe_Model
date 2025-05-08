%% Script to generate figures in the Jang,Nassar et al paper
clear;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SET THE ROOT DIRECTORY HERE (directory of the 'publicCode' folder)
dir_root = pwd;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

addpath(genpath(dir_root));
dir_fmincon = fullfile(dir_root,'data/fminconResults');

%% Figure 1 or 7 (Task structure figures)
clearvars -except dir*;

%%%%%%%%%%%%%%%%%%%%%
% Adjust this for experiment 1 or 2
whichExp = 1;
%%%%%%%%%%%%%%%%%%%%%

if whichExp==1, modelSub=33;  % Subject we used for the figure
elseif whichExp==2, modelSub=35;
end
[datastruct_nd,datastruct_24] = initializeCpmScript(whichExp);
datastruct = datastruct_nd;
% for sub = goodSubs
for si = modelSub
    s = [datastruct.subno]==si;
    x = datastruct(s).trialNum;
    % Get choice and outcome after inverting for other category
    otherCat_i = floor(datastruct(s).imageNo/1000)==2;
    choice_corrected = datastruct(s).play_pass; choice_corrected(otherCat_i) = 1-choice_corrected(otherCat_i);
    outcome_corrected = datastruct(s).taskOutcome; outcome_corrected(otherCat_i) = 1-outcome_corrected(otherCat_i);
    % Add gap for plotting
    gapSize = 0.01;
    outcome_corrected(outcome_corrected==0) = outcome_corrected(outcome_corrected==0)-gapSize;
    outcome_corrected(outcome_corrected==1) = outcome_corrected(outcome_corrected==1)+gapSize;
    choice_corrected(choice_corrected==0) = choice_corrected(choice_corrected==0)-(gapSize*2);
    choice_corrected(choice_corrected==1) = choice_corrected(choice_corrected==1)+(gapSize*2);
    
    % Plot reward prob for category 1 and appropriate choice and outcomes
    plots = [];
    figure; hold on;
    if whichExp==1
        plots(1) = plot(x,datastruct(s).fixedRewardProb(:,1),'k-');
        plots(2) = plot(x(~otherCat_i),datastruct(s).taskOutcome(~otherCat_i)-gapSize,'k.','MarkerSize',10);
        plots(3) = plot(x(otherCat_i),datastruct(s).taskOutcome(otherCat_i)+gapSize,'r.','MarkerSize',10);
        plots(4) = plot(x,datastruct(s).expP_all(:,1));
        legend(plots,{'True reward probability (living category)','Outcome (this category)','Outcome (other category)','Model reward probability (living category)'});
        xlabel('Trials'); ylabel('P(reward) for living category');
    elseif whichExp==2
        darkGreen = [0,0.5,0]; orange = [1,0.5,0];
        plots(1) = plot(x,datastruct(s).fixedRewardProb(:,1),'--','Color',darkGreen,'LineWidth',1);
        plots(2) = plot(x,datastruct(s).fixedRewardProb(:,2),'--','Color',orange,'LineWidth',1);
        plots(3) = plot(x,datastruct(s).expP_all(:,1),'-','Color',darkGreen,'LineWidth',1);
        plots(4) = plot(x,datastruct(s).expP_all(:,2),'-','Color',orange,'LineWidth',1);
        legend(plots,{'True reward probability (living category)','reward probability (nonliving category)','Model reward probability (living category)','Model reward probability (nonliving category)'});
        xlabel('Trials'); ylabel('P(reward)');
        set(gca,'XLim',[1,160],'YLim',[0,1]);
    end
    
    % Plot CPP, RU, fit LR
    % Load fmincon of best model
    load(fullfile(dir_root,'data', 'fminconResults', 'fminconResult_1111110000000.mat'));
    figure;
    subplot(3,1,1); hold on; plot(x,datastruct(s).CPP,'k-');
    set(gca,'ylim',[0,0.3]);
    title('Surprise');
    subplot(3,1,2); hold on; plot(x,datastruct(s).RU,'k-');
    set(gca,'ylim',[3,6]);
    title('Uncertainty');
    plots = [];
    subplot(3,1,3); hold on;
    plots(1) = plot(x,datastruct(s).LR,'k-');
    plots(2) = plot(x,fminconResult.LR(fminconResult.subno==1000+si,:),'r-');
    set(gca,'ylim',[0,0.7]);
    title('Learning rate');
    legend(plots,{'Ideal observer','Behavioral fit'});
    
    % Plot the task structure reward prob & fixed reward prob
    figure; hold on;
    plot(x,datastruct(s).rewardProb(:,1),'k-');
    plot(x,datastruct(s).play_pass(:,1),'r.','MarkerSize',10);
    xlabel('Trial'); ylabel('P(reward)');
    set(gca,'YLim',[0,1]);
    title('True reward probability');
    
    % Plot trial-by-trial FB prediction error
    figure; hold on;
    plot(x,datastruct(s).predError(:,1),'k-');
    set(gca,'XLim',[0,160],'YLim',[-110,110],'YTick',-100:50:100);
    xlabel('Trials'); ylabel('fbRPE');

    % Plot trial-by-trial IMAGE prediction error
    figure; hold on;
    plot(x,datastruct(s).imRPE(:,1),'k-');
    set(gca,'XLim',[0,160],'YLim',[-60,60],'YTick',-100:50:100);
    xlabel('Trials'); ylabel('imRPE');
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Figure 2
clearvars -except dir*;
[datastruct_nd,datastruct_24] = initializeCpmScript(1);

% Plot proportion of play for different reward probability and value
datastruct = cat(2,datastruct_nd,datastruct_24); % Combine both conditions
allValues = unique(datastruct(1).trialValue)';
nbins = 4; allBins = [0:1/(nbins):1-1/(nbins);1/(nbins):1/(nbins):1];
playProp_value = nan(length(datastruct),length(allValues));
playProp_prob = nan(length(datastruct),nbins);
playProp_vp = nan(length(datastruct),nbins,length(allValues));
for s = 1:length(datastruct)
    for v = 1:length(allValues)
        playProp_value(s,v) = mean(datastruct(s).play_pass(datastruct(s).trialValue==allValues(v)));
    end
    for p = 1:size(allBins,2)
        playProp_prob(s,p) = mean(datastruct(s).play_pass(datastruct(s).expP>=allBins(1,p) & datastruct(s).expP<allBins(2,p)));
        % For combined value & probability plot
        for v = 1:length(allValues)
            playProp_vp(s,p,v) = mean(datastruct(s).play_pass(datastruct(s).trialValue==allValues(v) & datastruct(s).expP>=allBins(1,p) & datastruct(s).expP<allBins(2,p)));
        end
    end
end
plots = [];
plotColors = linspecer(size(allBins,2));
figure; hold on;
for p = 1:size(allBins,2)
    toPlot = squeeze(playProp_vp(:,p,:));
    plots(p) = plot(1:size(toPlot,2),nanmean(toPlot,1),'.-k','Color',plotColors(p,:),'MarkerSize',30);
    %     plots(p) = errorbar(1:size(toPlot,2),nanmean(toPlot,1),nanstd(toPlot,[],1)/sqrt(size(toPlot,1)),'.-','Color',plotColors(p,:),'MarkerSize',25,'CapSize',0);
    set(gca,'XTick',1:size(toPlot,2),'XLim',[0,size(toPlot,2)+1],'XTickLabel',allValues);
    xlabel('Trial value'); ylabel('Proportion of PLAY');
    pbaspect([1 0.8 1]);
end
for k = 1:size(allBins,2), legendStr{k} = sprintf('%.02f-%.02f',allBins(1,k),allBins(2,k)); end
legend(plots,legendStr);


% Plot subject choice vs model-predicted play
% 1. Use model expected value as a metric of model-predicted play
% 2. Use softmax function to get model's choice to play/pass
% Combine both groups
nbin = 8;
datastruct = cat(2,datastruct_nd,datastruct_24);
playProb_sub = nan(length(datastruct),nbin);
playProb_model = nan(length(datastruct),nbin);
modelRep = 20;  % Repeat for model
% Load fmincon of best model
load(fullfile(dir_root,'data', 'fminconResults', 'fminconResult_1111110000000.mat'));
% inverse temperature for softmax choice behavior (larger = more greedy)
invT_params = fminconResult.fitParams(:,strcmp(fminconResult.paramsStr,'invT'));
if length(invT_params) ~= length(datastruct), error('Number of params doesn''t match number of subjects!'); end
% Get the value of playing based on the model
playValue = fminconResult.playValue;

allBinMedians = [];
for s = 1:length(datastruct)
    % create 10 bins with equal number of elements for expected value
    binEdges = [-inf,quantile(datastruct(s).ev,nbin-1),inf];
    bins = discretize(datastruct(s).ev,binEdges);
    
    % Model's choice using softmax
    model_play = nan(length(datastruct(s).ev),modelRep);
    for rep = 1:modelRep
        for t = 1:size(playValue,2)
            [pChoice]=softMax([playValue(s,t) 0], invT_params(s));
            model_play(t,rep)=rand<pChoice(1);
        end
    end
    model_play = mean(model_play,2);
    % Bin data
    subBinMedians = [];
    for b = unique(bins)'
        playProb_sub(s,b) = mean(datastruct(s).play_pass(bins==b));  % Subject's actual choice
        playProb_model(s,b) = mean(model_play(bins==b));  % Model's choice
        subBinMedians(b) = median(datastruct(s).ev(bins==b));  % Bin medians
    end
    allBinMedians = cat(1,allBinMedians,subBinMedians);
end
xLocs = mean(allBinMedians,1);

% Colors
darkGreen = [0,0.5,0];
orange = [1,0.5,0];

plots = [];
figureHandle = figure; hold on;
plots(1) = plot(xLocs-0.5,nanmean(playProb_sub,1),'.-','Color',darkGreen,'MarkerSize',30);
plots(2) = plot(xLocs+0.5,nanmean(playProb_model,1),'.-','Color',orange,'MarkerSize',30);
set(gca,'XLim',[-20,80]);
xlabel('Expected value (bin median)'); ylabel('Play probability');
legend(plots,{'Participants','Model'});
pbaspect([1,1,1]);




%% Rank BIC from fitted RL models
clearvars -except dir*;
[datastruct_nd,datastruct_24] = initializeCpmScript(1);

% load base model
baseModelFileName = 'fminconResult_1111000000000.mat';
load(fullfile(dir_fmincon,baseModelFileName));
fminconResult_base = fminconResult;

% Get list of all models
fminconFileNames = getDirFileNames(dir_fmincon);
fminconFileNames(strcmp(fminconFileNames,baseModelFileName)) = [];

allParams = {'invT','valExp','playBias','intcpt','CPP','RU','idealLR','choice','value','outcome','prevOutcome','imRPE','imRPE_play'};
% Show BIC of 5 models for paper
paramsToShow = {};
paramsToShow{1} = {'invT','valExp','playBias','intcpt','CPP','RU'};
paramsToShow{2} = {'invT','valExp','playBias','intcpt','idealLR'};
paramsToShow{3} = {'invT','valExp','playBias','intcpt','CPP'};
paramsToShow{4} = {'invT','valExp','playBias','intcpt','RU'};
paramsToShow{5} = {'invT','valExp','playBias','intcpt','choice'};

fminconStruct = struct;
fminconStruct.BIC_diff = {}; fminconStruct.BIC_mean = []; fminconStruct.BIC_se = [];
fminconStruct.paramsStr = {}; fminconStruct.paramsStrCell = [];
fminconStruct.showModel = zeros(length(fminconFileNames),1);
for ff = 1:length(fminconFileNames)
    load(fullfile(dir_fmincon,fminconFileNames{ff}));
    % get difference from base model BIC
    BIC_diff = fminconResult.BIC - fminconResult_base.BIC;
    
    fminconStruct.BIC_diff = cat(1,fminconStruct.BIC_diff,BIC_diff);
    fminconStruct.BIC_mean = cat(1,fminconStruct.BIC_mean,mean(BIC_diff));
    fminconStruct.BIC_se = cat(1,fminconStruct.BIC_se,std(BIC_diff)/sqrt(length(BIC_diff)));
    paramsStr = '';
    for ps = 1:length(fminconResult.paramsStr), paramsStr = cat(2,paramsStr,fminconResult.paramsStr{ps},' / '); end
    paramsStr(end) = [];
    fminconStruct.paramsStr = cat(1,fminconStruct.paramsStr,paramsStr);
    fminconStruct.paramsStrCell = cat(1,fminconStruct.paramsStrCell,{fminconResult.paramsStr});
    for ii = 1:length(paramsToShow)
        if isequal(paramsToShow{ii},fminconResult.paramsStr)
            fminconStruct.showModel(ff) = 1;
        end
    end
end


[bic_mean,I] = sort(fminconStruct.BIC_mean(fminconStruct.showModel==1));
bic_se = fminconStruct.BIC_se(fminconStruct.showModel==1); bic_se = bic_se(I);
bic_str = fminconStruct.paramsStr(fminconStruct.showModel==1); bic_str = bic_str(I);
figure; hold on;
bar(1:length(paramsToShow),bic_mean,'FaceColor','y');
errorbar(1:length(paramsToShow),bic_mean,bic_se,'k.','Marker','none');
set(gca,'XTick',1:length(paramsToShow),'XTickLabel',bic_str,'XTickLabelRotation',45);
ylabel('BIC difference from base model');


% Get parameter estimates for the best model - surprise & uncertainty
bestModelFileName = 'fminconResult_1111110000000.mat';
load(fullfile(dir_fmincon,bestModelFileName));
fminconResult_best = fminconResult;
paramsToPlot = fminconResult_best.fitParams(:,5:6);
maxParamThresh = 10;  % exclude excessive param values due to poor fit
paramsToPlot(abs(paramsToPlot) > maxParamThresh) = nan;
figure; hold on;
bar(nanmean(paramsToPlot,1));
errorbar(nanmean(paramsToPlot,1),nanstd(paramsToPlot,[],1)/sqrt(size(paramsToPlot,1)),'.k','MarkerSize',0.001);
set(gca,'XLim',[0,3],'XTick',[1,2],'XTickLabel',{'Surprise','Uncertainty'},'XTickLabelRotation',45);
ylabel('Effect on learning rate');

% Stats on the effect of surprise & uncertainty on LR
i_group = [ones(length(datastruct_nd),1);2*ones(length(datastruct_24),1)];
[mainEffect_surp2,~] = cpm_stats(paramsToPlot(:,1),i_group,1);
[mainEffect_RU,~] = cpm_stats(paramsToPlot(:,2),i_group,1);



% Add choice / imRPE to best behavioral model, see if it's improved
% Best model + imRPE + choice
paramsToUse = {'invT','valExp','playBias','intcpt','CPP','RU','imRPE_play','choice'};
paramsToFit = false(length(allParams),1)';
paramsToFit(ismember(allParams,paramsToUse)) = true;
paramStr = '';
for k = 1:length(paramsToFit), paramStr = cat(2,paramStr,num2str(double(paramsToFit(k)))); end
saveFileDir = fullfile(dir_fmincon,sprintf('fminconResult_%s.mat',paramStr));
load(saveFileDir);
[mainEffect_extended,~] = cpm_stats([fminconResult_best.BIC,fminconResult.BIC],i_group,2);

bicDiff_best = fminconResult_best.BIC-fminconResult_base.BIC;
bicDiff_ext = fminconResult.BIC-fminconResult_base.BIC;
[mean_bestmod,se_bestmod] = getMeanAndSE(bicDiff_best);
[mean_extmod,se_extmod] = getMeanAndSE(bicDiff_ext);


%% Find correlation between sensitivity to rprob & value vs. imRPE effect on memory

% We were interested in testing whether participants who were sensitive to reward
% value and probability also had a strong image RPE on memory effect. In other words,
% participants that better adjusted their behavior using information about the trial
% value and probability will be more likely to remember items associated with higher
% RPEs. To quantify sensitivity to reward value and probability, we fit a logistic
% regression model on play/pass behavior that included z-scored versions of the reward
% probability (derived from the ideal observer model) and reward value as predictors
% for each participant. To find the effect of image RPE on memory, we fit a linear
% regression model on mean-centered memory score that included the image RPE as
% the predictor. We then computed the Spearman correlation between the
% coefficients of the two regression models.

clearvars -except dir*;
[datastruct_nd,datastruct_24] = initializeCpmScript(1);
% Plot time course of task structure and latent variables
datastruct = cat(2,datastruct_nd,datastruct_24);

rprob_val_memory = nan(length(datastruct),1);
imRPE_memory = nan(length(datastruct),1);
rprob_memory = nan(length(datastruct),1);
for s = 1:length(datastruct)
    % Effect of reward prob & value on choice behavior
    y = datastruct(s).play_pass;
    X = zScoreX([datastruct(s).expP,datastruct(s).trialValue]);
    b = glmfit(X,y,'binomial');
    rprob_val_memory(s) = b(2);
    
    % Effect of imRPE on memory
    y = datastruct(s).memScore_oldImage-mean(datastruct(s).memScore_oldImage);
    X = zScoreX(datastruct(s).imRPE);
    b = glmfit(X,y,'normal');
    imRPE_memory(s) = b(2);
    
    % Effect of rprob on memory
    y = datastruct(s).memScore_oldImage-mean(datastruct(s).memScore_oldImage);
    X = zScoreX(datastruct(s).expP);
    b = glmfit(X,y,'normal');
    rprob_memory(s) = b(2);
end

[R_imRPE,P_imRPE,RL_imRPE,RU_imRPE] = corrcoef(rprob_val_memory,imRPE_memory);
[R_rprob,P_rprob,RL_rprob,RU_rprob] = corrcoef(rprob_val_memory,rprob_memory);


%% Plot proportion of old stimuli per memory score
clearvars -except dir*;
[datastruct_nd,datastruct_24] = initializeCpmScript(1);

% Plot settings
condColors = {'b','r'};

figure; hold on;
for task = 1:2
    if task==1, datastruct = datastruct_nd;
    else datastruct = datastruct_24;
    end
    memScore_all = unique([datastruct.memScore_oldImage])';
    oldImgProp_memsScore = [];
    for s = 1:length(datastruct)
        for memScore = memScore_all
            numOldImages = sum(datastruct(s).memScore_oldImage==memScore);
            numNewImages = sum(datastruct(s).memScore_newImage==memScore);
            oldImgProp_memsScore(s,memScore) = numOldImages / (numOldImages+numNewImages);
        end
    end
    plot(memScore_all,nanmean(oldImgProp_memsScore,1),'Color',condColors{task});
    errorbar(memScore_all,nanmean(oldImgProp_memsScore,1),nanstd(oldImgProp_memsScore,[],1)/sqrt(size(oldImgProp_memsScore,1)),'.','Color',condColors{task},'Marker','none');
    set(gca,'XTick',memScore_all,'XLim',[memScore_all(1)-1,memScore_all(end)+1],'YLim',[0,1]);
    pbaspect([1,1,1]);
    xlabel('Memory score'); ylabel('Proportion of chosen images that were OLD');
end




%% Plot signal-detection measures (d', ROC curve, AUC)
clearvars -except dir*;
whichExp = 2;
[datastruct_nd,datastruct_24] = initializeCpmScript(whichExp);

% Plot settings
condColors = {'b','r'};
condStr = {'no-delay','24hr-delay'};
condFields = {'cond_nd','cond_24'};
trlCondFields = {'all','play','pass'};

% Original or 24-hour delay version?
sigDet = struct;
statStruct = struct; statStruct.p = []; statStruct.t = []; statStruct.df = []; statStruct.d = [];
for whichTask = 1:2
    if whichTask == 1, datastruct = datastruct_nd;
    elseif whichTask == 2, datastruct = datastruct_24;
    end
    
    tempStruct = struct; tempStruct.all = []; tempStruct.play = []; tempStruct.pass = [];
    dPrime = tempStruct; auc = tempStruct; dPrime_hc = tempStruct; dPrime_binConf = tempStruct;
    roc = struct; roc.hit = tempStruct; roc.fa = tempStruct;
    
    for s = 1:length(datastruct)
        for trlCond = 1:length(trlCondFields)
            trlCondStr = trlCondFields{trlCond};
            if strcmp(trlCondStr,'all'), trlCond_i = true(length(datastruct(s).trialNum),1);
            elseif strcmp(trlCondStr,'play'), trlCond_i = datastruct(s).play_pass==1;
            elseif strcmp(trlCondStr,'pass'), trlCond_i = datastruct(s).play_pass==0;
            end
            
            % First, get regular d prime ignoring confidence measures
            hitRate = mean(datastruct(s).oldImageAcc(trlCond_i));
            faRate = 1-mean(datastruct(s).newImageAcc(trlCond_i));
            dPrime.(trlCondStr)(s,1) = dprime(hitRate,faRate);
            if isinf(dPrime.(trlCondStr)(s,1)), dPrime.(trlCondStr)(s,1) = nan; end  % Remove infinity dPrime
            
            % Get dPrime for each confidence condition
            for conf = 1:4
%                 hc_i = datastruct(s).oldConfidence >= conf & datastruct(s).newConfidence >= conf;
                hc_i = datastruct(s).oldConfidence + datastruct(s).newConfidence >= 2*conf;
                hitRate = mean(datastruct(s).oldImageAcc(trlCond_i & hc_i));
                faRate = 1-mean(datastruct(s).newImageAcc(trlCond_i & hc_i));
                dPrime_hc.(trlCondStr)(s,conf) = dprime(hitRate,faRate);
                if isinf(dPrime_hc.(trlCondStr)(s,conf)), dPrime_hc.(trlCondStr)(s,conf) = nan; end  % Remove infinity dPrime
            end
            
            % Get dPrime for binary high/low confidence
            for confCond = 1:2
                if confCond==1
                    hc_i = datastruct(s).oldConfidence + datastruct(s).newConfidence <= 4;
                else
                    hc_i = datastruct(s).oldConfidence + datastruct(s).newConfidence > 4;
                end
                hitRate = mean(datastruct(s).oldImageAcc(trlCond_i & hc_i));
                faRate = 1-mean(datastruct(s).newImageAcc(trlCond_i & hc_i));
                dPrime_binConf.(trlCondStr)(s,confCond) = dprime(hitRate,faRate);
                if isinf(dPrime_binConf.(trlCondStr)(s,confCond)), dPrime_binConf.(trlCondStr)(s,confCond) = nan; end  % Remove infinity dPrime
            end
            
            % ROC method: Andrew P. Yonelinas and Colleen M. Parks, 2007
            hit_roc = []; fa_roc = [];
            for memScore = 8:-1:1
                p_hit_roc = 0; p_fa_roc = 0;
                for mm = 8:-1:memScore
                    p_hit_roc = p_hit_roc + sum(datastruct(s).memScore_oldImage(trlCond_i)==mm)/sum(trlCond_i);
                    p_fa_roc = p_fa_roc + sum(datastruct(s).memScore_newImage(trlCond_i)==mm)/sum(trlCond_i);
                end
                hit_roc = cat(2,hit_roc,p_hit_roc);
                fa_roc = cat(2,fa_roc,p_fa_roc);
            end
            auc.(trlCondStr) = cat(1,auc.(trlCondStr),trapz([0,fa_roc],[0,hit_roc]));
            roc.hit.(trlCondStr) = cat(1,roc.hit.(trlCondStr),hit_roc);
            roc.fa.(trlCondStr) = cat(1,roc.fa.(trlCondStr),fa_roc);
        end
    end
    
    sigDet.dPrime.(condFields{whichTask}) = dPrime;
    sigDet.dPrime_hc.(condFields{whichTask}) = dPrime_hc;
    sigDet.dPrime_binConf.(condFields{whichTask}) = dPrime_binConf;
    sigDet.auc.(condFields{whichTask}) = auc;
    sigDet.roc.(condFields{whichTask}) = roc;
end



% Plot overall dPrime
figure; hold on;
for cond = 1:2
    [dPrime_mean,dPrime_se] = getMeanAndSE(sigDet.dPrime.(condFields{cond}).all(:,1));
    bar(cond,dPrime_mean,'FaceColor',condColors{cond});
    errorbar(cond,dPrime_mean,dPrime_se,'k.','Marker','none');
    ylabel('D prime');
    set(gca,'YLim',[0,1],'XTick',1,'XTickLabel',condStr{cond});
end

% Get d prime versus zero
mainEffect_dp_nd = struct;
mainEffect_dp_nd.mean = nanmean(sigDet.dPrime.cond_nd.all);
[~,mainEffect_dp_nd.p,mainEffect_dp_nd.ci,stats] = ttest(sigDet.dPrime.cond_nd.all);
mainEffect_dp_nd.t = stats.tstat;
mainEffect_dp_nd.df = stats.df;
mainEffect_dp_nd.d = nanmean(sigDet.dPrime.cond_nd.all)/nanstd(sigDet.dPrime.cond_nd.all);

mainEffect_dp_24 = struct;
mainEffect_dp_24.mean = nanmean(sigDet.dPrime.cond_24.all);
[~,mainEffect_dp_24.p,mainEffect_dp_24.ci,stats] = ttest(sigDet.dPrime.cond_24.all);
mainEffect_dp_24.t = stats.tstat;
mainEffect_dp_24.df = stats.df;
mainEffect_dp_24.d = nanmean(sigDet.dPrime.cond_nd.all)/nanstd(sigDet.dPrime.cond_nd.all);

% Compare d prime for high vs low confidence
i_group = ones(size(sigDet.dPrime_binConf.cond_nd.all,1),1);
[mainEffect_dp_conf_nd,~] = cpm_stats([sigDet.dPrime_binConf.cond_nd.all(:,2),sigDet.dPrime_binConf.cond_nd.all(:,1)],i_group,2);
i_group = ones(size(sigDet.dPrime_binConf.cond_24.all,1),1);
[mainEffect_dp_conf_24,~] = cpm_stats([sigDet.dPrime_binConf.cond_24.all(:,2),sigDet.dPrime_binConf.cond_24.all(:,1)],i_group,2);


% Plot ROC curves
figure; hold on;
for cond = 1:2
    plot(([0, nanmean(sigDet.roc.(condFields{cond}).fa.play, 1)]),  ([0, nanmean(sigDet.roc.(condFields{cond}).hit.play, 1)]), 'Color',condColors{cond});
    plot(([0, nanmean(sigDet.roc.(condFields{cond}).fa.pass, 1)]),  ([0, nanmean(sigDet.roc.(condFields{cond}).hit.pass, 1)]), ':','Color',condColors{cond});    
end
plot([0,1],[0,1],'k:');
set(gca,'yLim',[0,1],'XTick',0:0.2:1,'YTick',0:0.2:1);
ylabel('Hit rate'); xlabel('False-alarm rate'); pbaspect([1,1,1]);

% Plot AUC
stats_auc = statStruct;
figure;
for cond = 1:length(condFields)
    for trlCond = 2:length(trlCondFields)
        trlCondStr = trlCondFields{trlCond};
        figureHandle2 = subplot(1,2,cond); hold on;
        auc = sigDet.auc.(condFields{cond}).(trlCondStr);
        b = bar(trlCond-1,nanmean(auc));
        set(b,'facecolor',condColors{cond});
        errorbar(trlCond-1,nanmean(auc),nanstd(auc)/sqrt(sum(~isnan(auc))),'.k');
    end
    % Stats (play vs pass)
    [h,p,ci,stats] = ttest(sigDet.auc.(condFields{cond}).play,sigDet.auc.(condFields{cond}).pass);
    stats_auc.p(cond) = p;
    stats_auc.t(cond) = stats.tstat;
    stats_auc.df(cond) = stats.df;
    stats_auc.d(cond) = computeCohen_d(sigDet.auc.(condFields{cond}).play,sigDet.auc.(condFields{cond}).pass,'paired');
    set(gca,'xTick',1:length(trlCondFields(2:3)),'xTickLabel',trlCondFields(2:3),'xTickLabelRotation',45,'yLim',[0.6,0.72]); ylabel('AUC');
end

% Stats for AUC
auc_play = [sigDet.auc.cond_nd.play;sigDet.auc.cond_24.play];
auc_pass = [sigDet.auc.cond_nd.pass;sigDet.auc.cond_24.pass];
[~,p_aucPP,ci_aucPP,stats_aucPP] = ttest(auc_play,auc_pass);
d_aucPP = computeCohen_d(auc_play,auc_pass,'paired');

% Test for group difference
cond1effect = sigDet.auc.cond_nd.play-sigDet.auc.cond_nd.pass;
cond2effect = sigDet.auc.cond_24.play-sigDet.auc.cond_24.pass;
[~,p_aucPP_gd,ci_aucPP_gd,stats_aucPP_gd] = ttest2(cond1effect,cond2effect);
d_aucPP_gd = computeCohen_d(cond1effect,cond2effect,'independent');





%% Plot play/pass effect
clearvars -except dir*;

whichExp = 1;
[datastruct_nd,datastruct_24] = initializeCpmScript(whichExp);

% Plot settings
condColors = {'b','r'};
condStr = {'no-delay','24hr-delay'};
condFields = {'cond_nd','cond_24'};
trlCondFields = {'all','play','pass'};

statStruct = struct; statStruct.p = []; statStruct.t = []; statStruct.df = [];
playPassEffect = struct;
for whichTask = 1:2
    if whichTask == 1, datastruct = datastruct_nd;
    elseif whichTask == 2, datastruct = datastruct_24;
    end
    
    for s = 1:length(datastruct)
        memScore_mean = mean(datastruct(s).memScore_oldImage);
        for cond = {'play','pass'}
            if strcmp(char(cond),'play'), pp_i = datastruct(s).play_pass==1;
            else, pp_i = datastruct(s).play_pass==0;
            end
            playPassEffect.memScore.(condFields{whichTask}).(char(cond))(s,1) = mean(datastruct(s).memScore_oldImage(pp_i));
            playPassEffect.memScoreNorm.(condFields{whichTask}).(char(cond))(s,1) = mean(datastruct(s).memScore_oldImage(pp_i)) - memScore_mean;
            playPassEffect.memScoreDiff.(condFields{whichTask}).(char(cond))(s,1) = mean(datastruct(s).memScore_oldImage(pp_i)-datastruct(s).memScore_newImage(pp_i));
            playPassEffect.memScoreSum.(condFields{whichTask}).(char(cond))(s,1) = mean(datastruct(s).memScore_oldImage(pp_i)+datastruct(s).memScore_newImage(pp_i));
        end
    end
end


% Play vs plot memory score scatterplot
figure; hold on;
for whichTask = 1:2
    plot(playPassEffect.memScore.(condFields{whichTask}).play,playPassEffect.memScore.(condFields{whichTask}).pass,'.','Color',condColors{whichTask},'MarkerSize',15);
    plot([1,8], [1,8], '--k');
    set(gca,'XLim',[1,8],'YLim',[1,8])
    title('memScore for old img'); xlabel('play'); ylabel('pass');
    pbaspect([1,1,1]);
end

% To to stats on combined condition
playMemScoreDiff = []; passMemScoreDiff = [];
stats_memScoreDiff = statStruct;
for whichTask = 1:2 
    thisPlay = playPassEffect.memScoreDiff.(condFields{whichTask}).play;
    thisPass = playPassEffect.memScoreDiff.(condFields{whichTask}).pass;
    playMemScoreDiff = cat(1,playMemScoreDiff,[thisPlay,repmat(whichTask,length(thisPlay),1)]);
    passMemScoreDiff = cat(1,passMemScoreDiff,[thisPass,repmat(whichTask,length(thisPlay),1)]);
end

% Stats
i_group = [ones(length(datastruct_nd),1);2*ones(length(datastruct_24),1)];
[mainEffect_memScoreDiffPP,groupDiff_memScoreDiffPP] = cpm_stats([playMemScoreDiff(:,1),passMemScoreDiff(:,1)],i_group,2);


% Plot just the memScoreDiff together for paper
if whichExp==1, yLimit = [0.6,1.8];
elseif whichExp==2, yLimit = [0.6,1.9];
end
figureHandle = figure; hold on;
for whichTask = 1:2
    thisPlay = playPassEffect.memScoreDiff.(condFields{whichTask}).play;
    thisPass = playPassEffect.memScoreDiff.(condFields{whichTask}).pass;
    bar([2*whichTask-1,whichTask*2],[mean(thisPlay),mean(thisPass)],'FaceColor',condColors{whichTask});
    errorbar([2*whichTask-1,whichTask*2],[mean(thisPlay),mean(thisPass)],[std(thisPlay)/sqrt(length(thisPlay)),std(thisPass)/sqrt(length(thisPass))],'k.','MarkerSize',0.1);
    ylabel('Memory score difference');
    set(gca,'XTick',1:4,'XTickLabel',{'Play','Pass','Play','Pass'},'YLim',yLimit);
end




%% Plot the relationship between different task variables and memory
% Variables of interest: imRPE, fbRPE, value, Surprise, Uncertainty
clearvars -except dir*;
[datastruct_nd,datastruct_24] = initializeCpmScript(1);

% Options for plotting
plotoptions = struct;
plotoptions.savefig = 0;
plotoptions.figSaveDir = '/Users/anthony_jang/Dropbox (Personal)/cpm/Manuscript/NHB/revision final/figures/figs_matlab';

% Memory metric
% 1: memScoreOld
% 2: memScoreDiff (old - new)
% 3: memScoreSum (old + new)
% 4: corrected recognition score
plotoptions.memoryMetric = 1;
% 1: use separate play pass bins, 2: use single binning combining all trials
plotoptions.binningOption = 2;

% load data
plotoptions.exp1_datapath = fullfile(dir_root,'data','allData1.mat');
plotoptions.exp2_datapath = fullfile(dir_root,'data','allData3.mat');
plotoptions.expToLoad = 1;


% Figure 4: Play vs pass analysis on imRPE, rewProb, value
effects_out = plotMainMemEffects(plotoptions);

% Compute stats on slope
i_group = [ones(length(datastruct_nd),1);2*ones(length(datastruct_24),1)];
% imRPE
varslope = effects_out.imRPE.delay_3.slope.play;
[mainEffect_imRPE,groupDiff_imRPE] = cpm_stats(varslope,i_group,1);
% rewProb
varslope = effects_out.expP.delay_3.slope.play;
[mainEffect_expP,groupDiff_expP] = cpm_stats(varslope,i_group,1);
% value
varslope = effects_out.trialValue.delay_3.slope.play;
[mainEffect_value,groupDiff_value] = cpm_stats(varslope,i_group,1);


% Figure 5: current vs previous trial analysis on play trials for surprise,
% uncertainty, feedback RPE
effects_out2 = plotMainMemEffects2(plotoptions);

% Compute stats on slope
i_group = [ones(length(datastruct_nd),1);2*ones(length(datastruct_24),1)];

% Current trial
% imRPE
varslope = effects_out2.CPP.cur.delay_3.slope.play;
[mainEffect_CPP_cur,groupDiff_CPP_cur] = cpm_stats(varslope,i_group,1);
% rewProb
varslope = effects_out2.RU.cur.delay_3.slope.play;
[mainEffect_RU_cur,groupDiff_RU_cur] = cpm_stats(varslope,i_group,1);
% value
varslope = effects_out2.predError.cur.delay_3.slope.play;
[mainEffect_fbRPE_cur,groupDiff_fbRPE_cur] = cpm_stats(varslope,i_group,1);

% Previous trial
% Surprise
varslope = effects_out2.CPP.prev.delay_3.slope_bin.play;
[mainEffect_CPP_prev,groupDiff_CPP_prev] = cpm_stats(varslope,i_group,1);
% Uncertainty
varslope = effects_out2.RU.prev.delay_3.slope_bin.play;
[mainEffect_RU_prev,groupDiff_RU_prev] = cpm_stats(varslope,i_group,1);
% fbRPE
varslope = effects_out2.predError.prev.delay_3.slope_bin.play;
[mainEffect_fbRPE_prev,groupDiff_fbRPE_prev] = cpm_stats(varslope,i_group,1);






%% RPE crossover effect for Experiment 2
clearvars -except dir*;
[datastruct_nd,datastruct_24] = initializeCpmScript(2);
datastruct = cat(2,datastruct_nd,datastruct_24); taskStr = 'combined';
i_group = [ones(length(datastruct_nd),1);ones(length(datastruct_24),1)*2];

% Plot settings
condColors = {'b','r'};
condStr = {'no-delay','24hr-delay'};
condFields = {'cond_nd','cond_24'};
trlCondFields = {'all','play','pass'};

% Number of bins
nbin = 4;

for initializeDataVars = 1
    tempStruct = struct;
    tempStruct.all = []; tempStruct.play = []; tempStruct.pass = [];
    acc_trueRewProb                   = tempStruct;
    acc_trueRewProb_otherCat          = tempStruct;
    acc_expP_relBin                   = tempStruct;
    acc_expP_otherCat_relBin          = tempStruct;
    acc_imRPE_relBin                 = tempStruct;
    binMedian_thisCat = nan(length(datastruct),nbin);
    binMedian_otherCat = nan(length(datastruct),nbin);
    trueRewardProbs = [0.2,0.8];
end

allB = struct; allB.thisCat = []; allB.otherCat = [];
allB_imRPE = nan(length(datastruct),1);
allTrialVals = unique([datastruct.trialValue])';
for s = 1:length(datastruct)
    memScore = datastruct(s).memScore_oldImage;
    yLabelStr = 'Memory score';
    
    sub_average = nanmean(memScore);
    
    %%%%%%%% Reward probability %%%%%%%%%%%%%%%
    % Bins with equal number of elements
    binEdges = [-inf,quantile(datastruct(s).expP,nbin-1),inf];
    bins = discretize(datastruct(s).expP,binEdges);
    for b = unique(bins)'
        acc_expP_relBin.all(s,b) = nanmean(memScore(bins==b)) - sub_average;
        acc_expP_relBin.play(s,b) = nanmean(memScore(bins==b & datastruct(s).play_pass==1)) - sub_average;
        acc_expP_relBin.pass(s,b) = nanmean(memScore(bins==b & datastruct(s).play_pass==0)) - sub_average;
        acc_expP_relBin.binMedian(s,b) = median(datastruct(s).expP(bins==b & datastruct(s).play_pass==1));
        binMedian_thisCat(s,b) = median(datastruct(s).expP(bins==b));
    end
    
    % Reward prob of other category
    binEdges = [-inf,quantile(datastruct(s).expP_otherCat,nbin-1),inf];
    bins = discretize(datastruct(s).expP_otherCat,binEdges);
    for b = unique(bins)'
        acc_expP_otherCat_relBin.all(s,b) = nanmean(memScore(bins==b)) - sub_average;
        acc_expP_otherCat_relBin.play(s,b) = nanmean(memScore(bins==b & datastruct(s).play_pass==1)) - sub_average;
        acc_expP_otherCat_relBin.pass(s,b) = nanmean(memScore(bins==b & datastruct(s).play_pass==0)) - sub_average;
        acc_expP_otherCat_relBin.binMedian(s,b) = median(datastruct(s).expP_otherCat(bins==b & datastruct(s).play_pass==1));
        binMedian_otherCat(s,b) = median(datastruct(s).expP_otherCat(bins==b));
    end

    % Get the memory accuracy for each reward probability condition (0.8 or 0.2)
    for ip = 1:length(trueRewardProbs)
        acc_trueRewProb.play(s,ip) = mean(memScore(datastruct(s).rewardProb==trueRewardProbs(ip) & datastruct(s).play_pass==1)) - sub_average;
        acc_trueRewProb_otherCat.play(s,ip) = mean(memScore(datastruct(s).rewardProb_otherCat==trueRewardProbs(ip) & datastruct(s).play_pass==1)) - sub_average;
        
        acc_trueRewProb.pass(s,ip) = mean(memScore(datastruct(s).rewardProb==trueRewardProbs(ip) & datastruct(s).play_pass==0)) - sub_average;
        acc_trueRewProb_otherCat.pass(s,ip) = mean(memScore(datastruct(s).rewardProb_otherCat==trueRewardProbs(ip) & datastruct(s).play_pass==0)) - sub_average;
    end
    
    % Stats
    x_thisCat = datastruct(s).expP(datastruct(s).play_pass==1);
    x_otherCat = datastruct(s).expP_otherCat(datastruct(s).play_pass==1);
    y = memScore(datastruct(s).play_pass==1);
    b = regress(y,[ones(length(x_thisCat),1),x_thisCat]);
    allB.thisCat(s) = b(2);
    b = regress(y,[ones(length(x_otherCat),1),x_otherCat]);
    allB.otherCat(s) = b(2);
    
    % Get memory for image RPE
    binEdges = [-inf,quantile(datastruct(s).imRPE,nbin-1),inf];
    bins = discretize(datastruct(s).imRPE,binEdges);
    for b = unique(bins(~isnan(bins)))'
        acc_imRPE_relBin.all(s,b) = nanmean(memScore(bins==b)) - sub_average;
        acc_imRPE_relBin.play(s,b) = nanmean(memScore(bins==b & datastruct(s).play_pass==1)) - sub_average;
        acc_imRPE_relBin.pass(s,b) = nanmean(memScore(bins==b & datastruct(s).play_pass==0)) - sub_average;
        acc_imRPE_relBin.binMedian(s,b) = median(datastruct(s).imRPE(bins==b & datastruct(s).play_pass==1));
    end
    b = regress(acc_imRPE_relBin.play(s,:)',[ones(nbin,1),acc_imRPE_relBin.binMedian(s,:)']);
    allB_imRPE(s) = b(2);
end

% Stats
[mainEffect_thisCatSlope,groupDiff_thisCatSlope] = cpm_stats(allB.thisCat,i_group,1);
[mainEffect_otherCatSlope,groupDiff_otherCatSlope] = cpm_stats(allB.otherCat,i_group,1);
% slope of imRPE vs totest
[mainEffect_imRPE_mem,groupDiff_imRPE_mem] = cpm_stats(allB_imRPE,i_group,1);

% Paper figure plot - this category vs other category reward probability
% effect
xjitter = 0.01;
for delayCond = 1:2
    dataToUse = {acc_expP_relBin.play(i_group==delayCond,:),acc_expP_otherCat_relBin.play(i_group==delayCond,:);};
    xPts = mean([binMedian_thisCat;binMedian_otherCat],1);
    plots = [];
    figure; hold on;
    for c = 1:2
        toPlot = dataToUse{c};
        if c==1 xPoints = xPts-xjitter;
        else, xPoints = xPts+xjitter;
        end
        xLabelStr = 1:nbin;
        toPlot_se = nanstd(toPlot,[],1)/sqrt(size(toPlot,1));
        toPlot_mean = nanmean(toPlot,1);
        errorbar(xPoints,toPlot_mean,toPlot_se,'.','Color',condColors{delayCond},'MarkerSize',35);
        if c==1, plots(c) = plot(xPoints,toPlot_mean,'-','Color',condColors{delayCond});
        else, plots(c) = plot(xPoints,toPlot_mean,'--','Color',condColors{delayCond});
        end
        set(gca,'XLim',[0,1]);
        pbaspect([1,1,1]);
    end
    legend(plots,{'This category','Other category'});
    set(gca,'YLim',[-0.3,0.3]);
    xlabel('Reward probability');
    ylabel('Memory Score');
    plot(get(gca,'XLim'),[0,0],'k--');
end

% Plot thiscat ver otherCat effect
% Use true reward prob (0.2/0.8)
for delayCond = 1:2
    plots = [];
    dataToUse = {acc_trueRewProb.play(i_group==delayCond,:),acc_trueRewProb_otherCat.play(i_group==delayCond,:)};
    % dataToUse = {acc_trueRewProb.pass,acc_trueRewProb_otherCat.pass};
    figureHandle = figure; hold on;
    for c = 1:2
        [this_mean,this_se] = getMeanAndSE(dataToUse{c});
        if c==1
            xPoints = trueRewardProbs-xjitter;
            plots(c) = plot(xPoints,this_mean,'-','Color',condColors{delayCond});
        else
            xPoints = trueRewardProbs+xjitter;
            plots(c) = plot(xPoints,this_mean,'--','Color',condColors{delayCond});
        end
        errorbar(xPoints,this_mean,this_se,'.','Color',condColors{delayCond},'MarkerSize',35);
        set(gca,'XLim',[0,1]);
        pbaspect([1,1,1]);
    end
    plot(get(gca,'XLim'),[0,0],'k--');
    legend(plots,{'This category','Other category'});
    set(gca,'YLim',[-0.15,0.25]);
    xlabel('Reward probability');
    ylabel('Memory Score');
end

% Stats using ANOVA
y = cat(1,acc_trueRewProb.play(:,1),acc_trueRewProb.play(:,2),acc_trueRewProb_otherCat.play(:,1),acc_trueRewProb_otherCat.play(:,2));
prob = cat(1,repmat({'0.2'},length(datastruct),1),repmat({'0.8'},length(datastruct),1),repmat({'0.2'},length(datastruct),1),repmat({'0.8'},length(datastruct),1));
whichCat = cat(1,repmat({'this'},length(datastruct)*2,1),repmat({'other'},length(datastruct)*2,1));
[p,tbl] = anovan(y,{prob,whichCat},'model','interaction','varnames',{'prob','whichCat'},'display','off');

