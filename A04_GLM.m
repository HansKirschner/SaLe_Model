
clear 
load('data/fminconResult/fminconResult_wPrior_1110011100.mat')
load('data/all_EndPOints.mat')

% Subject exclusion
Criterion = 200;
Exclude = find(EndPOints <= Criterion)
fminconResult.fitParams(Exclude,:) = []; EndPOints(Exclude) = [];

invT                  =  nanzscore(fminconResult.fitParams(:,1));
PlayBias              =  nanzscore(fminconResult.fitParams(:,2));
LRItcpt               =  nanzscore(fminconResult.fitParams(:,3));
bLR_idealLR           =  nanzscore(fminconResult.fitParams(:,4));
bLR_RewardMagnitude   =  nanzscore(fminconResult.fitParams(:,5));
bLR_VisualSurprise    =  nanzscore(abs(fminconResult.fitParams(:,6)));

Points = normalise(EndPOints)';

% 1. model

M.DesignM = table(invT,PlayBias,bLR_idealLR,bLR_RewardMagnitude,bLR_VisualSurprise, Points);
% M.DesignM.RT  = normalise(M.DesignM.RT);

M.labels    = { {'low' 'high'} {'low' 'high'} {'low' 'high'} {'low' 'high'} {'low' 'high'} {'low' 'high'}};
M.CatVars   = [                                                                                         ];
M.modelspec = 'Points ~ invT + PlayBias + bLR_idealLR + bLR_RewardMagnitude + bLR_VisualSurprise';
LF          = 'normal';


RegModall   = fitglm(M.DesignM, M.modelspec, 'CategoricalVars', M.CatVars,'Distribution', LF,'Options',statset('MaxIter',1000));
CI          = coefCI(RegModall);

RegModall.Rsquared.Ordinary
RegModall.Rsquared.Adjusted

ID = [1:length(invT)]';
bigDataGLM = table(ID,invT,PlayBias,bLR_idealLR,bLR_RewardMagnitude,bLR_VisualSurprise, Points);
writetable(bigDataGLM, 'SaLePoints_GLM.csv');

% check correlation of IV's
figure(1);clf;
imagesc(corr(table2array(M.DesignM)))
colorbar
corr(table2array(M.DesignM))

%% plot 

color    = get(groot,'DefaultAxesColorOrder');

paramLabels = {'invT' 'PlayBias' 'bLR_i_d_e_a_l_L_R' 'bLR_R_e_w_a_r_d_M_a_g_n_i_t_u_d_e' '|bLR_V_i_s_u_a_l_S_u_r_p_r_i_s_e|'};
figure(2);clf;
set(gca, 'box', 'off')
nP=length(CI)-1;
hold on

bCoeff    = [-.001 .293 .184 .120 -.624 -.749]';
bCI       = [-.31 .31; -.124 .731; -.181 .552; -.272 .542; -1.008 -.220; -1.126 -.359];

CI = bCI;

[~, I]=sort(RegModall.Coefficients.Estimate(2:end));

%sortWeights = RegModall.Coefficients.Estimate(I+1);

sortWeights = bCoeff(I+1);

used=RegModall.Coefficients.Estimate(2:end);
Scale=max(max(abs(CI(2:end,:))));
xline(0, '--k',LineWidth=1)
plot( CI(I+1,:)', [1:nP; 1:nP],'-k',LineWidth=2)
plot(sortWeights, 1:nP,  'o', 'markerFaceColor', color(1,:),'markerEdgeColor', 'k', 'lineWidth', 1, 'markerSize', 15)
ylim([0, nP+1])
xlabel('Predictor weight')

set(gca, 'ytick', 1:nP, 'yticklabels', paramLabels(I), 'box', 'off')
fig = gcf;
fig.PaperUnits = 'centimeters';
fig.PaperPosition = [0 0 12 24];
set(gca, 'FontSize', 20)
ytickangle(30)

h = gcf;
h.Position = [-1681         501         560         420];
print2pdf(fullfile(pwd,'Figures',['GLM_overallPoints_Bambi.pdf']),h,300)


