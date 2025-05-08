%% Script to fit a reinforcement leaning model to behavior of the SaLe Task
% using a constrained search algorithm (fmincon)
clear;
rootDir = pwd;
addpath(genpath(rootDir));
set(0,'defaultAxesFontSize',20);
fminconDir = fullfile(rootDir,'data/fminconResult');
savepath = fullfile(rootDir,'data');

%% Run fmincon

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% How many iterations of fmincon should we run?
reps    = 15;
maxReps = 30;

% All parameters that can potentially influence decision behavior
allParams = {'invT','playBias','intcpt','CPP','RU','idealLR','RewardMagnitude','VisualSurprise','FeedConfrim','FeedReality'};

% Choose parameters to test
paramset1 = {'invT','playBias','intcpt'};
paramset2 = {'invT','playBias','intcpt','idealLR'};
paramset6 = {'invT','playBias','intcpt','idealLR','RewardMagnitude'};
paramset8 = {'invT','playBias','intcpt','idealLR','RewardMagnitude','VisualSurprise'};

%allParamSets = {paramset1,paramset2,paramset3,paramset4,paramset5,paramset6,paramset7,paramset8,paramset9};
allParamSets = {paramset8};% this is the model in the paper

for tt = 1:length(allParamSets)
    
    paramsToUse = allParamSets{tt};
    % Load data
    load('data/SaleData_IdealObserver.mat');

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    paramsToFit = false(length(allParams),1)';
    paramsToFit(ismember(allParams,paramsToUse)) = true;
    paramCount = sum(paramsToFit);
    allParamsToFit = paramsToFit;
    allParamsToFit = logical(allParamsToFit);
    
    
    subCount = 1;
    for paramSet = 1:size(allParamsToFit,1)
        % Check if this parameter set has been tested
        thisParamsBin = allParamsToFit(paramSet,:);
        paramStr = '';
        for k = 1:length(thisParamsBin), paramStr = cat(2,paramStr,num2str(double(thisParamsBin(k)))); end
        saveFileDir = fullfile(fminconDir,sprintf('fminconResult_wPrior_%s.mat',paramStr));
        saveFileDirData = fullfile(savepath,sprintf('Data_wPrior_%s.mat',paramStr));

        saveTempFileDir     = fullfile(fminconDir,sprintf('fminconResult_wPrior_%s_incomplete.mat',paramStr));
        saveTempFileDirData = fullfile(savepath,sprintf('Data_wPrior_%s_incomplete.mat',paramStr));
        runThisSub = false;
        if exist(saveTempFileDir,'file')
            runThisSub = true;
            load(saveTempFileDir);
            load(saveTempFileDirData);
            s_start = length(fminconResult.subno) + 1;
        elseif ~exist(saveFileDir,'file')
            runThisSub = true;
            s_start = 1;
            % Initialize variables to save parameters and BIC
            fminconResult = struct;
            fminconResult.subno = [];
            fminconResult.paramsStr = allParams(thisParamsBin);
            fminconResult.paramsBin = thisParamsBin;
            fminconResult.fitParams = [];
            fminconResult.BIC = [];
            fminconResult.negLogLike = [];
           
      
        end
        if runThisSub
            for s = s_start:length(dstruct)
                fminconResult.subno = cat(1,fminconResult.subno,dstruct(s).subno);
                % Get the input structure for fmincon
                tic;
                fprintf('Param %d / %d: subno %d',tt,length(allParamSets),s);
                input = getBehavModelInput_Sale(dstruct(s),allParams(thisParamsBin),nan);
                
                minNegLL=inf;
                params_fit = [];
                reps_thisSub = 0;   % So it doesn't exceed maxReps
                i = 1;
                while i <= reps
                    reps_thisSub = reps_thisSub + 1;
                    try
                        [fminconOutput]= fitLearningFunc_Sale(input);
                    catch
                        fprintf('round %d of 15 timeout error...\n',i);
                    end
                    if fminconOutput.negLogLike<minNegLL
                        bestOutput=fminconOutput;
                        minNegLL=bestOutput.negLogLike;
                        params_fit = bestOutput.params;
                    end

                    % set the new start point for new fmincon
                    newInput = getBehavModelInput_Sale(dstruct(s),allParams(thisParamsBin),nan);
                    input.startPoint = newInput.newStartPoint;
                    
                    % Do more reps if fit values look sketchy
                    if any(abs(params_fit(2:end)) > 4)
                        if reps_thisSub < maxReps
                            i = 1;
                        else
                            i = reps+1;
                        end
                    else
                        i = i + 1;
                    end
                end
                
                subBIC = computeBIC(bestOutput.logLike_woPrior, paramCount, length(dstruct(s).TrNr));
                
                % store the fit values for each param
                fminconResult.fitParams = cat(1,fminconResult.fitParams,params_fit);
                % store the BIC
                fminconResult.BIC = cat(1,fminconResult.BIC,subBIC);
                fminconResult.negLogLike = cat(1,fminconResult.negLogLike,minNegLL);
               
                dstruct(s).EV = bestOutput.expP(1:end-1);
                dstruct(s).RL_LR = bestOutput.allLR;
                dstruct(s).RPE = bestOutput.RPE;
                dstruct(s).ChoiceProb = bestOutput.ChoiceProb';

                % Save progress in temporary structure in case I need to
                % stop the code midway
                save(saveTempFileDir,'fminconResult');
                save(saveTempFileDirData,'dstruct');
                
                subCount = subCount + 1;
                elapsedTime = toc;
                fprintf('(%.02fsec)\n',elapsedTime);
            end
            save(saveFileDir,'fminconResult');
            save(saveFileDirData,'dstruct');

            % delete temp data
            delete(saveTempFileDir);
            delete(saveTempFileDirData);
        end
    end
end


