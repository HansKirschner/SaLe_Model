function effects_out = plotMainMemEffects2(plotoptions)
%Function to plot how different variables affect memory
% Input plotoptions fields:
% style:
% only play, plot both delay in same plot, include previous trial effect

% load data
if plotoptions.expToLoad==1 || plotoptions.expToLoad==2
    load(plotoptions.(sprintf('exp%d_datapath',plotoptions.expToLoad)));
    taskstr = sprintf('Experiment %d',plotoptions.expToLoad);
elseif plotoptions.expToLoad==2
    [datastruct_nd,datastruct_24] = combineExperiments(plotoptions.exp1_datapath,plotoptions.exp2_datapath);
    taskstr = 'Experiment 1 & 2';
end

% plotting style, and which variables to use for each
varstotest = {'Surprise','Uncertainty','fbRPE'};
vars_field = {'CPP','RU','predError'};
vars_nbin = [3,3,3];
xAxisLimit = {[0.01,0.08],[3.2,4.5],[-45,45]};
yAxisLimit = {[-0.2,0.301],[-0.15,0.301],[-0.2,0.4]};


effects_out = struct;
plotmarkersize = 45;
delayColors = {'b','r','g'};
trialTypeStr = {'Current trial','Previous trial'};
for v = 1:length(vars_field)
    var_mem = struct;
    
    for whichTrial = 1:2  % current vs previous trial
        for delayCond = 1:3
            % Plotting parameters
            if delayCond == 1
                datastruct = datastruct_nd;
            elseif delayCond == 2
                datastruct = datastruct_24;
            elseif delayCond==3
                datastruct = cat(2,datastruct_nd,datastruct_24);
            end
            nbin = vars_nbin(v);
            
            var_mem_thisDelay = struct;
            var_mem_thisDelay.bindata.play = nan(length(datastruct),nbin);
            var_mem_thisDelay.bindata.binmedian.play = nan(length(datastruct),nbin);
            var_mem_thisDelay.slope.play = nan(length(datastruct),1);
            var_mem_thisDelay.slope_bin.play = nan(length(datastruct),1);
            
            if plotoptions.memoryMetric==1, ystr = 'Memory score (old)';
            elseif plotoptions.memoryMetric==2, ystr = 'Memory score difference';
            elseif plotoptions.memoryMetric==3, ystr = 'Memory score sum';
            elseif plotoptions.memoryMetric==4, ystr = 'Corrected recognition score';
            end
            
            for s = 1:length(datastruct)
                if plotoptions.memoryMetric==1
                    mem = datastruct(s).memScore_oldImage;
                elseif plotoptions.memoryMetric==2
                    mem = datastruct(s).memScore_oldImage-datastruct(s).memScore_newImage;
                elseif plotoptions.memoryMetric==3
                    mem = datastruct(s).memScore_oldImage+datastruct(s).memScore_newImage;
                elseif plotoptions.memoryMetric==4
                    mem = datastruct(s).oldImageAcc - (1-datastruct(s).newImageAcc);
                end
                
                % Use bins
                var_all = datastruct(s).(vars_field{v});
                if whichTrial==1
                    % current trial was play
                    i_play = find(datastruct(s).play_pass==1);
                    var_pp = var_all(i_play);
                    mem_pp = mem(i_play);
                elseif whichTrial==2
                    % previous trial was play
                    i_play = find(datastruct(s).play_pass==1);
                    i_play_prev = i_play(i_play < length(datastruct(s).play_pass)) + 1;
                    var_pp = var_all(i_play_prev);
                    mem_pp = mem(i_play_prev);
                    % Remove zero values for fbRPE
                    if v==3
                        i_remove = var_pp==0;
                        var_pp(i_remove) = [];
                        mem_pp(i_remove) = [];
                    end
                end
                bins = discretize(var_pp,[-inf,quantile(var_pp,nbin-1),inf]);
                for b = 1:nbin
                    var_mem_thisDelay.bindata.play(s,b) = mean(mem_pp(bins==b)) - mean(mem);
                    var_mem_thisDelay.bindata.binmedian.play(s,b) = mean(var_pp(bins==b));
                end
                
                % non-binned regression
                betas = regress(mem_pp,[ones(length(var_pp),1),var_pp]);
                var_mem_thisDelay.slope.play(s,1) = betas(2);
                
                % binned regression
                betas = regress(var_mem_thisDelay.bindata.play(s,:)',[ones(nbin,1),var_mem_thisDelay.bindata.binmedian.play(s,:)']);
                var_mem_thisDelay.slope_bin.play(s) = betas(2);
            end
            
            if whichTrial==1
                var_mem.cur.(sprintf('delay_%d',delayCond)) = var_mem_thisDelay;
            elseif whichTrial==2
                var_mem.prev.(sprintf('delay_%d',delayCond)) = var_mem_thisDelay;
            end
        end
        
    end
    
    fh = figure('units','normalized','outerposition',[1,1,1,0.6]);
    for whichTrial = 1:2  % current vs previous
        subplot(1,2,whichTrial); hold on;
        for delayCond = 1:2
            if whichTrial==1, var_mem_thisDelay = var_mem.cur.(sprintf('delay_%d',delayCond));
            elseif whichTrial==2, var_mem_thisDelay = var_mem.prev.(sprintf('delay_%d',delayCond));
            end
            nbin = vars_nbin(v);
            pp_line = '.-';
            if strcmp(vars_field{v},'trialValue'), xvals = 1:5; set(gca,'xlim',[0,xvals(end)+1]);
            else, xvals = nanmean(var_mem_thisDelay.bindata.binmedian.play,1);
            end
            [mean_pp,se_pp] = getMeanAndSE(var_mem_thisDelay.bindata.play);
            plots(delayCond) = errorbar(xvals,mean_pp,se_pp,pp_line,'Color',delayColors{delayCond},'markersize',plotmarkersize);
        end
        pbaspect([1,1,1]);
        set(gca,'xlim',xAxisLimit{v},'ylim',yAxisLimit{v});
        plot(get(gca,'xlim'),[0,0],'k--');
        title(sprintf('%s (%s)',taskstr,trialTypeStr{whichTrial})); xlabel(varstotest{v}); ylabel(ystr);
        legend(plots,{'No delay','24 Hr delay'});
    end
    if plotoptions.savefig == 1
        print(fh,fullfile(plotoptions.figSaveDir,sprintf('%s_mem_exp(%d)_delay(%d).svg',varstotest{v},plotoptions.expToLoad,delayCond)),'-dsvg','-r300');
    end
    
    % Compile processed data
    effects_out.(vars_field{v}) = var_mem;
    
end



