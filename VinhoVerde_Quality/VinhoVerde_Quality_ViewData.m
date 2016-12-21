%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% $Author: Lumbrer $    $Date: 2016/12/16 $    $Revision: 1.2 $
% Copyright: Francisco Lumbreras
%
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% File Name: VinhoVerde_Quality_ViewData.m
% Description: Script to visualise the source data used in this study
% from Cortez et al. 2009 'Modeling wine preferences by data mining from
% physicochemical properties'.
%
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% ---------------- CODE HERE ----------------------------------------------

clc;
close all;
clear all;
set(0,'DefaultTextInterpreter','none') 
%Add path to files required
addpath(genpath(pwd));
%Define a plot flag to prevent the creation of figures if set to 0.
plotflag=true;

%struct to store figures
FiguresView=struct;

%% Import data and add label

Data=readtable('winedata.csv');
Data.QCLabel=categorical(Data.QCLabel);
Labels=unique(Data.QCLabel);
LabelsCell=categories(Labels);

%% Visualise our data with focus on Fixed Acidity & Density
if plotflag
FiguresView.fig1=findobj('Type','Figure','Number',1);
if ~isempty(FiguresView.fig1)
    close(FiguresView.fig1);
end
FiguresView.fig1=figure(1);
gscatter(Data.Density,Data.FixedAcidity,Data.QCLabel);
title('QCLabel as function of Density and Fixed Acidity');
xlabel('Density'); ylabel('Fixed Acidity');


% Extract mean and standard deviation and plot by class
[mFA,sFA]=grpstats(Data.FixedAcidity,Data.QCLabel,{@mean,@std});
FiguresView.fig2=findobj('Type','Figure','Number',2);
if ~isempty(FiguresView.fig2)
    close(FiguresView.fig2);
end
FiguresView.fig2=figure(2);
hold all;
for uu=1:length(mFA)
normFA=normpdf((-3*sFA(uu)+mFA(uu):0.1:3*sFA(uu)+mFA(uu)),mFA(uu),sFA(uu));
plot((-3*sFA(uu)+mFA(uu):0.1:3*sFA(uu)+mFA(uu)),normFA,'DisplayName',['QC Label ',char(Labels(uu))]);
legend('-DynamicLegend')
end
grid on;
hold off;
title('Fixed Acidity Normal PDF Fit')
xlabel('Fixed Acidity');ylabel('P(FixedAcidity|Category)')

end

%% Clean data from NaN and normalise to zero mean and unitary std

winedata=normTable(Data); %The function normTable will also clean all NaN and inf values.
winedataArr=winedata{:,1:end-1};

winedata01range=normTable(Data,{'Range'});
winedata01rangeArr=winedata01range{:,1:end-1};
%% Extract data
measnames = winedata.Properties.VariableNames(1:end-1);

%%Initial look to how features are correlated to labels. 

if plotflag
    
    FiguresView.fig3=findobj('Type','Figure','Number',3);
if ~isempty(FiguresView.fig3)
    close(FiguresView.fig3);
end
FiguresView.fig3=figure(3);

for kk = 1:11
    subplot(3,4,kk);
    boxplot(winedataArr(:,kk),winedata.QCLabel,'plotstyle','compact')
    title(measnames{kk})
end
 

%% Another view of quality-variable correlation
%  Calculate the median and CI of each variable, grouped by quality score
[medbyqc,rngbyqc] = grpstats(winedataArr,winedata.QCLabel,{@median,'predci'});
ncats=length(categories(Labels));

%  Plot each variable against quality
FiguresView.fig4=findobj('Type','Figure','Number',4);
if ~isempty(FiguresView.fig4)
    close(FiguresView.fig4);
end
FiguresView.fig4=figure(4);
for kk = 1:11
    ax = subplot(3,4,kk);
    % Plot median
    plot(1:ncats,medbyqc(:,kk),'bo')
    % Add CI
    hold on
    plot(1:ncats,rngbyqc(:,kk,1),'+-')
    plot(1:ncats,rngbyqc(:,kk,2),'+-')
    % Clean up axes
    ylabel(measnames{kk})
    ax.XTick = 1:ncats;
    ax.XTickLabel = categories(Labels);
end
legend('Median [ o ]','Confidence Interval -95% [-+-]','Confidence Interval +95% [-+-]')
legend('boxoff')

%%
%Now let us plot histograms for each feature separated by label but using
%the data that has been normalised to a [0,1] range. 


FiguresView.fig5=findobj('Type','Figure','Number',5);
if ~isempty(FiguresView.fig5)
    close(FiguresView.fig5);
end
FiguresView.fig5=figure(5);

for kk=1:11
    subplot(3,4,kk)
    hold on
    for i=ncats:-1:1
        index=(winedata.QCLabel==LabelsCell{i});
        histogram(winedata01rangeArr(index,kk),'Normalization','probability','BinWidth',0.05)
    end
    xlabel(measnames{kk})
    ylabel('QC Label')

    
end



%%
% Let us now perform data visualisation in a reduced dimension space.
%
%% Select a subset of the data
rng(1234)
idx = randsample(size(winedataArr,1),2000);

% transform the data into 3 dimensions using pairwise distance
distances=pdist(winedataArr(idx,:));

%reconstruct the coordinates using the minimum number of dimensions to
%represent the pairwise distance

[recwinedata,eigs]=cmdscale(distances);

%check the amount of information that the principal axes can help maintain
%by looking at the eigenvalues of x*x'

FiguresView.fig6=findobj('Type','Figure','Number',6);
if ~isempty(FiguresView.fig6)
    close(FiguresView.fig6);
end
FiguresView.fig6=figure(6);
pareto(eigs)

%Now we can visualise the data on the new converted space both all
%together and by quality label
FiguresView.fig7=findobj('Type','Figure','Number',7);
if ~isempty(FiguresView.fig7)
    close(FiguresView.fig7);
end
FiguresView.fig7=figure(7);
scatter3(recwinedata(:,1),recwinedata(:,2),recwinedata(:,3),[],'.')
view(-105,60)

FiguresView.fig8=findobj('Type','Figure','Number',8);
if ~isempty(FiguresView.fig8)
    close(FiguresView.fig8);
end
FiguresView.fig8=figure(8);
scatter3(recwinedata(:,1),recwinedata(:,2),recwinedata(:,3),[],winedata.QCLabel(idx),'.')
view(-105,60)

disp(' ')
    disp('Finished Data Display ')
    
    user_option=input('Would you like to close the figures from this chapter? Y/N [N]:','s');
    if isempty(user_option)
        user_option='n';
    end
    if strcmpi('y',user_option)
        
        for vv=1:8
            figname=strcat('fig',num2str(vv));
            if ishghandle(FiguresView.(figname))&&strcmp(get(FiguresView.(figname),'BeingDeleted'),'off')
                close(FiguresView.(figname));
            end
        end
    end


end

%--------------------------------------------------------------------------
%--------------------------------------------------------------------------



