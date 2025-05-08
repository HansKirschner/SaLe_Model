%% Script to process data - add computational variables using raw data 
% --> adobted from Jang et al., 2019

clear;
dir_root = pwd;  % Set the root directory with all the code
dir_processed = fullfile(dir_root,'data');
addpath(genpath(dir_root));

load(fullfile(dir_root,sprintf('data/all_behavior_SaLe')));
dstruct = D;

for i = 1:length(dstruct)

    Index1 = find(dstruct(i).Stimno ==1 & dstruct(i).SymStage ==1);
    Index2 = find(dstruct(i).Stimno ==2 & dstruct(i).reversal ==1);
    Index3 = find(dstruct(i).Stimno ==3 & dstruct(i).reversal ==1);
    
    IndexFirstShow_i = [Index2(1) Index3(1)];
    IndexFirstShow = zeros(dstruct(i).Nch,1)';
    IndexFirstShow(IndexFirstShow_i) = 1;

    dstruct(i).IndexFirstShow = IndexFirstShow;

    dstruct(i).reversal(Index1(1:7))=1;
    dstruct(i).reversal(Index2(1:7))=1;
    dstruct(i).reversal(Index3(1:7))=1;

    dstruct(i).subno = i;

end

for s = 1:length(dstruct)
    totalTrials = length(dstruct(s).reversal);
    
    % Get model behavior    
    H = sum(dstruct(s).reversal)/totalTrials;% Hazard Rate
    modelOutput = getOptimalLRs(struct('data',dstruct(s).outcome','H',H,'StimChange',dstruct(s).IndexFirstShow'));
    shannonH = -modelOutput.dataLL; %what is shannonH?

    % Calculate change-point probability based on data likelihood and true hazard rate:
    Q = (.5.*H)./( exp(modelOutput.dataLL).*(1-H));
    CPP = Q./(1+Q);  % Change-point probability; Surprise
    RU = modelOutput.begTrialEntropy;  % Relative uncertainty
    expP = modelOutput.expP;  % experienced reward probability

    % Ideal learing rate
    LR = modelOutput.LR;

    % Compile all computational variables
    % Expected subjective reward probability
    dstruct(s).expP = expP;
    
    % Ideal learning rate
    dstruct(s).LR = LR;

    % Trial uncertainty
    dstruct(s).RU = RU;
    
    % Surprise (CPP; Change point probability)
    dstruct(s).CPP = CPP;
end

save("data/SaleData_IdealObserver.mat","dstruct")

%% Lets plot

% Plot CPP, RU, optimal LR and fitted LR
figure(2);clf;
subplot(4,1,1); hold on;
x = 1:dstruct(s).Nch;
gapSize = 0.01;
set(0,'defaultAxesFontSize',15);
plots = [];
plots(1) = plot(x,dstruct(s).Prob/100,'k-',LineWidth=2);
plots(2) = plot(x,dstruct(s).outcome,'k.','MarkerSize',15);
plots(3) = plot(x,dstruct(s).expP(:,1),LineWidth=3);
gridxy(find(dstruct(s).reversal==1));
gridxy(find(dstruct(s).IndexFirstShow==1),'LineWidth',3);
legend(plots,{'True reward probability','Outcome','Model reward probability'});
xlabel('Trials'); ylabel('P(reward)');
set(gca,'XLim',[1,dstruct(s).Nch],'YLim',[-.05,1.05]);

subplot(4,1,2); hold on; plot(x,dstruct(s).CPP,'k-',LineWidth=2);
set(gca,'ylim',[0,0.3],'xlim',[1,dstruct(s).Nch]);
title('Surprise');
gridxy(find(dstruct(s).reversal==1));
gridxy(find(dstruct(s).IndexFirstShow==1),'LineWidth',3);

subplot(4,1,3); hold on; plot(x,dstruct(s).RU,'k-',LineWidth=2);
set(gca,'ylim',[3,6],'xlim',[1,dstruct(s).Nch]);
title('Uncertainty');
gridxy(find(dstruct(s).reversal==1));
gridxy(find(dstruct(s).IndexFirstShow==1),'LineWidth',3);

subplot(4,1,4); hold on;
plots(1) = plot(x,dstruct(s).LR,'k-',LineWidth=2);
%plots(2) = plot(x,D(s).lr(1:end),'r-',LineWidth=2);
set(gca,'ylim',[0,0.7],'xlim',[1,dstruct(s).Nch]);
title('Learning rate');
gridxy(find(dstruct(s).reversal==1));
gridxy(find(dstruct(s).IndexFirstShow==1),'LineWidth',3);

ert
%legend(plots,{'Ideal observer','Behavioral fit SZ'});
h=gcf;
h.Position = [-2154          52        1848         962];
print2pdf('Figures/model_predictions_Sale',h,300)


figure(3);clf;
h=subplot(2,1,1); hold on;
x = 1:dstruct(s).Nch;
gapSize = 0.01;
set(0,'defaultAxesFontSize',25);
plots = [];
plots(1) = plot(x,dstruct(s).Prob(x)/100,'k-',LineWidth=2);
plots(2) = plot(x,dstruct(s).outcome(x),'k.','MarkerSize',20);
plots(3) = plot(x,AGF_running_average(dstruct(s).expP((x),1),2,2),LineWidth=6);
gridxy(find(dstruct(s).reversal(x)==1));
gridxy(find(dstruct(s).IndexFirstShow(x)==1),'LineWidth',3);
legend(plots,{'True reward probability','Outcome','Model reward probability'});
xlabel('Trials'); ylabel('P_(_r_e_w_a_r_d_)');
set(gca,'XLim',[1,dstruct(s).Nch],'YLim',[-.05,1.05]);

h = subplot(2,1,2); hold on;
plots(1) = plot(x,AGF_running_average(dstruct(s).LR(x),2,2),'k-',LineWidth=6);
set(0,'defaultAxesFontSize',25);
set(gca,'ylim',[0,0.4],'xlim',[1,dstruct(s).Nch]);
gridxy(find(dstruct(s).reversal(x)==1));
gridxy(find(dstruct(s).reversal==1));
gridxy(find(dstruct(s).IndexFirstShow==1),'LineWidth',3);
xlabel('Trials'); ylabel('Learning rate');

h=gcf;
h.Position = [ -2037         376        1125         619];
print2pdf('Figures/DynamicLR_Illustration',h,300)

R_idealLR           = dstruct(s).LR;
R_RewardMagnitude   = dstruct(s).magnitude';
R_VisualSurprise    = dstruct(s).flash';

%just visualize the designmatrix
DM = [norm_and_scale(R_idealLR, 0, 1) R_RewardMagnitude R_VisualSurprise];

figure(4);clf;
imagesc(DM); colormap('gray'); title('Designmatrix'); ylabel('trials'); xlabel('regressors'); set(gca, 'XTick', [1 2 3 ], 'XTickLabel', {'ideal LR' 'Reward Magnitude' 'Visual Surprise'})

h=gcf;
h.Position = [584   331   461   670];
print2pdf('Figures/DM_LR_RLModel',h,300)


