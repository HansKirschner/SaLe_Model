clear;
rootDir = pwd;
addpath(genpath(rootDir));
set(0,'defaultAxesFontSize',20);

% load data
load(fullfile(rootDir,sprintf('data/all_behaviour_M6_SimulatedData_REFIT')));


%%
% Model 6 - full model + starting EV scaling + LR decay
AF.HyperPriors = [0 2;       0 1;     0 1;      0 1;     0 1;       0 1;       0 1; ];
AF.ParamNames  = {'beta'   'alpha1'   'alpha2' 'alpha3' 'alpha4'    'omega'    'eta' }; 
AF.Defaults    = [0.1        0.4      0.4       0.4      0.4        0.5        .2    ];   %default parameters if no parameter is present
AF.Order       = [1          2        3         4        5          6          7     ];   %maps onto model order of parameters
AF.Free        = [1          0        0         0        0          0          0     ];
AF.Cut         = 1e-4; % how estimated
AF.CutP        = 1e-7;  %Determines power of the priors - how estimated - also currently not applied in FitRL?
FitF           = @FitRL_REFIT_idealObserver;
AF.ModelName   = char(FitF);
AF.PlotOn      = 1; % if set, plots Fit information and parameter distribution
AF.DoPrior     = 0;
AF.nP          = sum(AF.Free);


AF.prior_distributions = [makedist('gamma', 1.4,0.1) ];
AF.prior_functions = @(x1) ([pdf(AF.prior_distributions(1),x1(1)) ]); 


oldOpts = optimset('fmincon');
options=optimset(oldOpts, 'maxFunEvals', 1000000, 'MaxIter', 1000000,'Display', 'off');
reps = 20;

clear FitPara nLL
for s = 1 : length(D)
    tic
    fprintf('Modelling subject %d / %d',s,length(D));

    minNegLL=inf;
    params_fit = [];
    i = 1;
    while i <= reps

        if i == 1
            fminconOutcome.fitParams  = fmincon(@(x)FitF(x,D(s),AF), AF.Defaults, [], [], [], [], AF.HyperPriors(:,1), AF.HyperPriors(:,2),[],options);
            fminconOutcome.negLogLike = FitF(fminconOutcome.fitParams,D(s),AF);
        else
            OK = 0; count = 0;
            while OK == 0
                try
                    AF.newDefaults = unifrnd(AF.HyperPriors(:,1),AF.HyperPriors(:,2))';
                    fminconOutcome.fitParams  = fmincon(@(x)FitF(x,D(s),AF), AF.newDefaults, [], [], [], [], AF.HyperPriors(:,1), AF.HyperPriors(:,2),[],options);
                    fminconOutcome.negLogLike = FitF(fminconOutcome.fitParams,D(s),AF);
                    OK = 1;
                catch
                    OK = 0;

                end
            end
        end

        if fminconOutcome.negLogLike<minNegLL
            bestOutput=fminconOutcome;
            minNegLL=bestOutput.negLogLike;
            FitPara(:,s) = bestOutput.fitParams';
        end

        i = i+1;
    end

    [nLL(s), mo] = FitF(FitPara(:,s),D(s),AF);
    % Add Model parameters to behavioral data
    D(s).PC_idealObs  = mo.PC;
    elapsedTime = toc;
    fprintf('(%.02fsec)\n',elapsedTime);
end



[SumAIC, SumBIC, SumLL, iBIC, bic, aic]   = PlotFit(AF, FitPara,  nLL,D);

save('data/individualBicAic_idealObs.mat','bic','aic')
save(['data/all_behaviour_M6_SimulatedData_REFIT.mat'],'D')
save('data/FitPara_idealObs','FitPara');
ert

ttestout(bic(IndexSchizo),bic(IndexControl),2)
ttestout(bic(IndexDepression),bic(IndexControl),1)
ttestout(bic(IndexDepression),bic(IndexSchizo),1)

ttestout(aic(IndexSchizo),aic(IndexControl),1)
ttestout(aic(IndexDepression),aic(IndexControl),1)


%%
% Get all valid stages
% load data
load('data/all_behavior_M6.mat')
load('data/FitPara_M6.mat')


% Build Array (VP) with Names and Filenames
for i = 1:length(files)
    VP(1,i) = {strtok(files(i).name,'.')};
    VP(2,i) = {strtok(files(i).name)};
end

D(14) = [];%this subject has a odd number of trials, which makes it diffictult to include in this figure...
VP(:,14)=[];

IndexDepression = contains(VP(1,:),'_D_');
IndexControl 	= contains(VP(1,:),'_K_');
IndexSchizo 	= contains(VP(1,:),'_S_');


n = size(D,2);
SymStage = [];
for c =  1: n
    US = D(c).SymStage'; If = 1; AllI = [];
    for c2 = 2 : length(US)
        if US(c2)<=US(c2-1) %end of stage
            if US(c2-1)>19
                AllI = [AllI If:c2-1];
            end
            If = c2;
        end
    end
    length(AllI);
    AllIAll(c) = {AllI};
    SymStage = [SymStage D(c).SymStage(AllI)' ];
end
SymStage


% Change all fieldnames into equally size arrays
FN = fieldnames(D); FN(1:2) = [];
for c = 1 : length(FN)
    eval([FN{c} ' = [];'])
    for c2 = 1 : n
        eval([FN{c} ' = [' FN{c} ' D(c2).(FN{c})(AllIAll{c2})''];'])
    end
end

%% Plot choice behavior and chocie predictions
set(0,'defaultAxesFontSize',20);
figure(3);clf;
subplot(3,1,1)
shadedErrBar([], AGF_running_average(mean(choice(:,IndexControl),2),2,2), AGF_running_average(se(choice(:,IndexControl)',0.99),2,2),'k'); hold;
plot(mean(Prob,2)/100,'-k',LineWidth=2); 
plot(AGF_running_average(mean(PC1(:,IndexControl),2),2,2),LineWidth=2); 
legend({'choice' 'GT' 'PC'})
ylim([-.05 1.05]);gridxy(find(reversal(:,1)==1));
ylabel('Choices & P-choice'); %xlabel('Tr Nr'); 
title('Control');
subplot(3,1,2)
shadedErrBar([], AGF_running_average(mean(choice(:,IndexDepression),2),2,2), AGF_running_average(se(choice(:,IndexDepression)',0.99),2,2),'r'); hold;
plot(mean(Prob,2)/100,'-k',LineWidth=2); 
plot(AGF_running_average(mean(PC1(:,IndexDepression),2),2,2),LineWidth=2); 
legend({'choice' 'GT' 'PC'})
ylim([-.05 1.05]);gridxy(find(reversal(:,1)==1));
ylabel('Choices & P-choice'); %xlabel('Tr Nr'); 
title('MDD');
subplot(3,1,3)
shadedErrBar([], AGF_running_average(mean(choice(:,IndexSchizo),2),2,2), AGF_running_average(se(choice(:,IndexSchizo)',0.99),2,2),'b'); hold;
plot(mean(Prob,2)/100,'-k',LineWidth=2); 
plot(AGF_running_average(mean(PC1(:,IndexSchizo),2),2,2),LineWidth=2); 
legend({'choice' 'GT' 'PC'})
ylim([-.05 1.05]);gridxy(find(reversal(:,1)==1));
ylabel('Choices & P-choice'); xlabel('Tr Nr'); 
title('SZ');
ert

% beta
ttestout(FitPara(1,IndexSchizo),FitPara(1,IndexControl),1);
ttestout(FitPara(1,IndexDepression),FitPara(1,IndexControl),1);

% alpha1 - factual win 
ttestout(FitPara(2,IndexSchizo),FitPara(2,IndexControl),1);
ttestout(FitPara(2,IndexDepression),FitPara(2,IndexControl),1);

% alpha1 - factual loss
ttestout(FitPara(3,IndexSchizo),FitPara(3,IndexControl),1);
ttestout(FitPara(3,IndexDepression),FitPara(3,IndexControl),1);

% alpha1 - counterfactual win
ttestout(FitPara(4,IndexSchizo),FitPara(4,IndexControl),1);
ttestout(FitPara(4,IndexDepression),FitPara(4,IndexControl),1);

% alpha1 - counterfactual loss
ttestout(FitPara(5,IndexSchizo),FitPara(5,IndexControl),1);
ttestout(FitPara(5,IndexDepression),FitPara(5,IndexControl),1);

%omega
ttestout(FitPara(6,IndexSchizo),FitPara(6,IndexControl),1);
ttestout(FitPara(6,IndexDepression),FitPara(6,IndexControl),1);

%eta
ttestout(FitPara(7,IndexSchizo),FitPara(7,IndexControl),1);
ttestout(FitPara(7,IndexDepression),FitPara(7,IndexControl),1);


%% Plot LR
%are there differences in the LR? 
Group                   = zeros(1,length(VP));
Group(IndexDepression)  = 1;
Group(IndexControl)     = 2;
Group(IndexSchizo)      = 3;


for i = 1:344
    p(i) = anova1(squeeze(lr(i,:)),Group','off');
end


figure(4);clf;
subplot(3,1,1)
shadedErrBar([], AGF_running_average(mean(lr(:,IndexControl),2),0,0), AGF_running_average(se(lr(:,IndexControl)',0.99),2,2),'k'); hold;
plot(mean(Prob,2)/100/2,'-k',LineWidth=.5); 
shade_the_back(p<.05, [183 183 183]./255, 1:length(lr));
%plot(AGF_running_average(mean(PC1(:,IndexControl),2),2,2),LineWidth=2); 
%legend({'LR' 'GT'})
ylim([0.05 .45]);gridxy(find(reversal(:,1)==1));xlim([0 344]);
gridxy(find(Stimno(:,1) == 2, 1 ),'linewidth',1);
ylabel('LR'); %xlabel('Tr Nr'); 
title(['LR Control (Min/Max = ' num2str(min(mean(lr(:,IndexControl),2))) '/' num2str(max(mean(lr(:,IndexControl),2))) ')']);
subplot(3,1,2)
shadedErrBar([], AGF_running_average(mean(lr(:,IndexDepression),2),0,0), AGF_running_average(se(lr(:,IndexDepression)',0.99),2,2),'r'); hold;
plot(mean(Prob,2)/100/2,'-k',LineWidth=.5); 
shade_the_back(p<.05, [183 183 183]./255, 1:length(lr));
%plot(AGF_running_average(mean(PC1(:,IndexDepression),2),2,2),LineWidth=2); 
%legend({'LR' 'GT'})
ylim([0.05 .45]);gridxy(find(reversal(:,1)==1));xlim([0 344]);
gridxy(find(Stimno(:,1) == 2, 1 ),'linewidth',2);
ylabel('LR'); %xlabel('Tr Nr'); 
title(['LR MDD (Min/Max = ' num2str(min(mean(lr(:,IndexDepression),2))) '/' num2str(max(mean(lr(:,IndexDepression),2))) ')']);
subplot(3,1,3)
shadedErrBar([], AGF_running_average(mean(lr(:,IndexSchizo),2),0,0), AGF_running_average(se(lr(:,IndexSchizo)',0.99),2,2),'b'); hold;
plot(mean(Prob,2)/100/2,'-k',LineWidth=.5); 
shade_the_back(p<.05, [183 183 183]./255, 1:length(lr));
%plot(AGF_running_average(mean(PC1(:,IndexSchizo),2),2,2),LineWidth=2); 
%legend({'LR' 'GT'})
ylim([0.05 .45]);gridxy(find(reversal(:,1)==1));xlim([0 344]);
gridxy(find(Stimno(:,1) == 2, 1 ),'linewidth',2);
ylabel('LR'); xlabel('Tr Nr'); 
title(['LR SZ (Min/Max = ' num2str(min(mean(lr(:,IndexSchizo),2))) '/' num2str(max(mean(lr(:,IndexSchizo),2))) ')']);


%% plot EV and RPE
for i = 1:344
    p_EV(i)  = anova1(squeeze(EV(i,:)),Group','off');
    p_RPE(i) = anova1(squeeze(RPE(i,:)),Group','off');
end

%% Plot LR
figure(5);clf;
subplot(2,1,1)
shadedErrBar([], AGF_running_average(mean(RPE(:,IndexControl),2),0,0), AGF_running_average(se(RPE(:,IndexControl)',0.99),2,2),'k'); hold on;
shadedErrBar([], AGF_running_average(mean(RPE(:,IndexDepression),2),0,0), AGF_running_average(se(RPE(:,IndexDepression)',0.99),2,2),'g'); hold on;
shadedErrBar([], AGF_running_average(mean(RPE(:,IndexSchizo),2),0,0), AGF_running_average(se(RPE(:,IndexSchizo)',0.99),2,2),'b'); hold on;
plot((mean(Prob,2)-50)/100,'-k',LineWidth=1); 
%plot(AGF_running_average(mean(PC1(:,IndexControl),2),2,2),LineWidth=2); 
%legend({'LR' 'GT'})
ylim([-1 1]);gridxy(find(reversal(:,1)==1));xlim([0 344]);
gridxy(find(Stimno(:,1) == 2, 1 ),'linewidth',1);
ylabel('RPE'); %xlabel('Tr Nr'); 
title(['RPE Control (Min/Max = ' num2str(min(mean(RPE(:,IndexControl),2))) '/' num2str(max(mean(RPE(:,IndexControl),2))) ')' ...
    ', RPE MDD (Min/Max = ' num2str(min(mean(RPE(:,IndexDepression),2))) '/' num2str(max(mean(RPE(:,IndexDepression),2))) ')' ...
    ', RPE SZ (Min/Max = ' num2str(min(mean(RPE(:,IndexSchizo),2))) '/' num2str(max(mean(RPE(:,IndexSchizo),2))) ')']);
shade_the_back(p_RPE<.07, [183 183 183]./255, 1:length(lr));

subplot(2,1,2)
shadedErrBar([], AGF_running_average(mean(EV(:,IndexControl),2),0,0), AGF_running_average(se(EV(:,IndexControl)',0.99),2,2),'k'); hold on;
shadedErrBar([], AGF_running_average(mean(EV(:,IndexDepression),2),0,0), AGF_running_average(se(EV(:,IndexDepression)',0.99),2,2),'g'); hold on;
shadedErrBar([], AGF_running_average(mean(EV(:,IndexSchizo),2),0,0), AGF_running_average(se(EV(:,IndexSchizo)',0.99),2,2),'b'); hold on;
plot(mean(Prob,2)/100,'-k',LineWidth=.5); 
%plot(AGF_running_average(mean(PC1(:,IndexControl),2),2,2),LineWidth=2); 
%legend({'LR' 'GT'})
ylim([0 1]);gridxy(find(reversal(:,1)==1));xlim([0 344]);
gridxy(find(Stimno(:,1) == 2, 1 ),'linewidth',1);
ylabel('EV'); %xlabel('Tr Nr'); 
title(['EV Control (Min/Max = ' num2str(min(mean(EV(:,IndexControl),2))) '/' num2str(max(mean(EV(:,IndexControl),2))) ')' ...
    ', EV MDD (Min/Max = ' num2str(min(mean(EV(:,IndexDepression),2))) '/' num2str(max(mean(EV(:,IndexDepression),2))) ')' ...
    ', EV SZ (Min/Max = ' num2str(min(mean(EV(:,IndexSchizo),2))) '/' num2str(max(mean(EV(:,IndexSchizo),2))) ')']);
shade_the_back(p_EV<.07, [183 183 183]./255, 1:length(lr));
h = gcf;
%print2pdf(fullfile(pwd,'Figures',['EVandRPE.pdf']),h,300)