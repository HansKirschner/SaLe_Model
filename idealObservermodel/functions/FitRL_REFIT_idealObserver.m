function [nLL, mo] = FitRL_REFIT_idealObserver(IN,D,AF)
%keyboard

AF.nP          = sum(AF.Free); 

%% inital output
EV      = D.norm_expP;

%defaults 
Beta = AF.Defaults(1);


%%
Beta        = IN(AF.Order(1));

OParams = Beta;


%% calculate priors
if AF.DoPrior
    Prior               = AF.prior_functions(OParams);
    Prior(Prior<AF.Cut) = AF.Cut;
    Prior               = -log(Prior(logical(AF.Free))); %has been change to only incluce priors for free parameters
else
    Prior = 0;
end


%% Model

%this was done seperately


%%
%bring EVs in borders
EV(EV>1) = 1;
EV(EV<0) = 0;

%get choice probabilty and choice 
PC                  = 1 ./ (1 + exp((-EV+0.5) ./ Beta)); %softmax of expected value is choice probability - this beta to one ad define the other stuff above

mo.PC   = PC;

nLL               = -sum(log(1-abs(PC'-make_long([D.choice])))) + sum(Prior);


mo.OParams        = OParams;
mo.CallParms      = IN;
mo.Priors         = Prior;


return