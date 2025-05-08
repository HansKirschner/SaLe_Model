function effects_out = plotMainMemEffects(plotoptions)
%Function to plot how different variables affect memory
% Input plotoptions fields:
% style:
% 1: play/pass for each delay condition)
% 2: only play, plot both delay in same plot, include previous trial effect

% load data
if plotoptions.expToLoad==1
    load(plotoptions.(sprintf('exp%d_datapath',plotoptions.expToLoad)));
    taskstr = sprintf('Experiment %d',plotoptions.expToLoad);
elseif plotoptions.expToLoad==2
    [datastruct_nd,datastruct_24] = combineExperiments(plotoptions.exp1_datapath,plotoptions.exp2_datapath);
    taskstr = 'Experiment 1 & 2';
end

% plotting style, and which variables to use for each
varstotest = {'Value','P(rew)','imRPE'};
vars_field = {'trialValue','expP','imRPE'};
vars_nbin = [5,4,4];
xAxisLimit = {[0,6],[0,1],[-25,25]};
yAxisLimit = {[-0.35,0.35],[-0.35,0.35],[-0.35,0.35]};


effects_out = struct;
plotmarkersize = 45;
delayColors = {'b','r','g'};
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
            var_mem_thisDelay.bindata.pass = nan(length(datastruct),nbin);
            var_mem_thisDelay.bindata.binmedian.all = nan(length(datastruct),nbin);
            var_mem_thisDelay.bindata.binmedian.play = nan(length(datastruct),nbin);
            var_mem_thisDelay.bindata.binmedian.pass = nan(length(datastruct),nbin);
            var_mem_thisDelay.slope.play = nan(length(datastruct),1); var_mem_thisDelay.slope.pass = nan(length(datastruct),1);
            var_mem_thisDelay.slope_bin.play = nan(length(datastruct),1); var_mem_thisDelay.slope_bin.pass = nan(length(datastruct),1);
            
            ppfields = {'all','play','pass'};
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
                for pp = 1:3
                    if pp==1, i_pp = true(length(datastruct(s).play_pass),1);
                    elseif pp==2, i_pp = datastruct(s).play_pass==1;
                    elseif pp==3, i_pp = datastruct(s).play_pass==0;
                    end
                    var_pp = datastruct(s).(vars_field{v})(i_pp);
                    mem_pp = mem(i_pp);
                    var_all = datastruct(s).(vars_field{v});
                    bins = discretize(var_all,[-inf,quantile(var_all,nbin-1),inf]);
                    for b = 1:nbin
                        var_mem_thisDelay.bindata.(ppfields{pp})(s,b) = mean(mem(i_pp & bins==b)) - mean(mem);
                        var_mem_thisDelay.bindata.binmedian.(ppfields{pp})(s,b) = mean(var_all(i_pp & bins==b));
                    end
                    
                    % non-binned regression
                    betas = regress(mem_pp,[ones(sum(i_pp),1),var_pp]);
                    var_mem_thisDelay.slope.(ppfields{pp})(s,1) = betas(2);
                    
                    % binned regression
                    betas = regress(var_mem_thisDelay.bindata.(ppfields{pp})(s,:)',[ones(nbin,1),var_mem_thisDelay.bindata.binmedian.(ppfields{pp})(s,:)']);
                    var_mem_thisDelay.slope_bin.(ppfields{pp})(s) = betas(2);
                end
            end
            
            var_mem.(sprintf('delay_%d',delayCond)) = var_mem_thisDelay;
        end
        
    end
    
    fh = figure('units','normalized','outerposition',[1,1,1,0.6]);
    % Plot
    for delayCond = 1:2
        var_mem_thisDelay = var_mem.(sprintf('delay_%d',delayCond));
        % Plotting parameters
        if delayCond == 1
            delayStr = 'No delay';
        elseif delayCond == 2
            delayStr = '24 Hr delay';
        elseif delayCond==3
            delayStr = 'Combined';
        end
        nbin = vars_nbin(v);
        
        plots = []; i_pl = 1;
        subplot(1,2,delayCond); hold on;
        for pp = 2:3
            if pp==2, pp_line = '.-';
            elseif pp==3, pp_line = '.--';
            end
            if strcmp(vars_field{v},'trialValue'), xvals = 1:5; set(gca,'xlim',[0,xvals(end)+1]);
            else, xvals = nanmean(var_mem_thisDelay.bindata.binmedian.(ppfields{pp}),1);
            end
            [mean_pp,se_pp] = getMeanAndSE(var_mem_thisDelay.bindata.(ppfields{pp}));
            plots(i_pl) = errorbar(xvals,mean_pp,se_pp,pp_line,'Color',delayColors{delayCond},'markersize',plotmarkersize);
            i_pl = i_pl+1;
        end
        pbaspect([1,1,1]);
        set(gca,'xlim',xAxisLimit{v},'ylim',yAxisLimit{v});
        plot(get(gca,'xlim'),[0,0],'k--');
        title(sprintf('%s (%s)',delayStr,taskstr)); xlabel(varstotest{v}); ylabel(ystr);
        legend(plots,{'Play','Pass'});
    end
    if plotoptions.savefig == 1
        print(fh,fullfile(plotoptions.figSaveDir,sprintf('%s_mem_exp(%d)_delay(%d).svg',varstotest{v},plotoptions.expToLoad,delayCond)),'-dsvg','-r300');
    end
    
    
    % Compile processed data
    effects_out.(vars_field{v}) = var_mem;
    
end



