%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% $Author: Lumbrer $    $Date: 2016/12/16 $    $Revision: 2.1 $
% Copyright: Francisco Lumbreras
%
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% File Name: VinhoVerde_Quality_UnsupervisedLearning.m
% Description: Script to perform unsupervised learning via a number of
% different algorithms on the data from Cortez et al. 2009 'Modeling wine
% preferences by data mining from physicochemical properties'.
%
% This file has been generated as script code but could easily be
% integrated in a user interactive GUI. In the first section of the script,
% a number of switches are defined to activate or deactivate the different
% chapters of the script. These can all run independently from each other. 
%
% In order to introduce unsupervised learning to new Matlab users the
% following chapters have been created:
%
% A) A study of the data natural clustering via self organising maps (SOM) 
% including multiple visualisations of the latter. 
%
% B) A study of K-means clustering on the data and the parameterisation of
% the algorithm. In this chapter, simple parallelisation is used to study
% the influence of varying parameters and different criteria are explored
% to extract an optimal number of clusters. Additionally, data is
% visualised giving preference to the clusters showing greatest sparsity
% accross the data. 
%
% C) A brief tutorial on the tools in Matlab used to perform Hierarchical
% clustering and evaluate the suitability of such analysis to the sample
% data
%
% D) A study on EM Mixture of Gaussian clustering including 2D
% visualisation of the data to better understand the assumptions behind
% this approach to clustering. 
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
% Create a struct to store figures
FiguresUnsup=struct;


%% Define switches to activate each of the chapters of the script
% Switches for chapters A, B, C & D
SOMswitch=boolean(1);        % A - Activate self organising map generation
Kmeansswitch=boolean(0);     % B - Activate kmeans clustering
Hierarchswitch=boolean(1);   % C - Activate hierarchical clustering
GMMswitch=boolean(1);        % D - Activate Mixture of Gaussians Model

% Switch to identify if multiple parallel workers are available locally
Parallelswitch=true;
% Define the number of cores in the parallel pool available locally
n_cores=4;
% Set random seed for reproducibility
rng(1234) 

%% Section 1
% Import data and add label to classify 

Data=readtable('winedata.csv');
Data.QCLabel=categorical(Data.QCLabel);
QCLabel=Data.QCLabel; %Includes NaN data
Labels=unique(Data.QCLabel);
LabelsCell=categories(Labels);

% Load the data label distinguishing between Red and White wines
load winetype

% Remove all invalid data as NaN or inf. See the function file for more
% details
DataClean=cleanTable(Data);

% Standardise the data in the table for distance and gradient computation
winedata=normTable(Data);

% Extract a clean version of the labels
WineClasses=DataClean.QCLabel;


%% Section 2 
% Extract the features in a numeric matrix

Features=DataClean{:,1:end-1};
Features_Norm=winedata{:,1:end-1};

% Extract number of features and their names
n_features=size(Features,2);
Features_Names=winedata.Properties.VariableNames(1:end-1);


if SOMswitch % {CHAPTER A}
    
    disp(' ')
    disp(['--------------------------------------------------------------------------------------------';...
        '------------------------------------  CHAPTER A  -------------------------------------------';...
        '--------------------------------------------------------------------------------------------'])
    
    %% Section 3-a
    % Define a Self Organising Map (SOM):  Competitive neural network that 
    % aims to represent high dimensional data in a reduced space using 
    % nodes that can be seen as reference points optimised to depict 
    % distances accross all training samples. 
    
    % Let us define a SOM of 8x8 nodes, train on half the data and test on 
    % the other half
    
    % For Neural Network training the data is expected in a format where 
    % each training sample is a column rather than a row
    randind=rand(size(Features_Norm,1),1)>0.5;
    
    TrainsetSOM=Features_Norm(randind,:)';
    TestsetSOM=Features_Norm(~randind,:)';
    
    % Define size of SOM
    SOMsize=[8 8];
    
    % Create the map, define training epochs and deactivate GUI window
    % display
    Map=selforgmap(SOMsize);
    Map.trainParam.epochs=100;
    Map.trainParam.showWindow=false;
    
    % Train map, notice the use of a method in a class implemented using 
    % pass by value (this is the standard way to call the train method).
    Map=train(Map,TrainsetSOM);
    
    %% Section 3-b
    % Visualize the resulting map
    % Sample hits
    FiguresUnsup.f1=findobj('type','figure','Name','SOM Sample Hits (plotsomhits)');
    if ishghandle(FiguresUnsup.f1)
        close(FiguresUnsup.f1)
    end
    figure
    plotsomhits(Map,TrainsetSOM)
    % Neighbour distances
    FiguresUnsup.f2=findobj('type','figure','Name','SOM Neighbor Distances (plotsomnd)');
    if ishghandle(FiguresUnsup.f2)
        close(FiguresUnsup.f2)
    end
    FiguresUnsup.somnd=figure;
    plotsomnd(Map)
    %% Section 3-c
    % Test network on testset
    
    % We can predict the assignment of each train sample to a neuron. As we
    % have m*n neurons and the output is a matrix of m*n rows and same 
    % number of columns (samples) as the test data. The row with a 1 value 
    % on it indicates pertenence to that row number neuron for the sample 
    % corresponding to the column.
    predsSOM=Map(TestsetSOM);
    % We can use vec2ind to convert the prediction to a more readable
    % arrange were each value will indicate the node number to which a
    % sample has been assigned. 
    predindSOM=vec2ind(predsSOM);
    % Now use the function plotsomhitscolored to show an averaged colourmap 
    % over the classes and how these are arranged in the SOM.
    FiguresUnsup.somhitcol=findobj('type','figure','Name',...
        'SOM Sample Hits (plotsomhits)');
    if ishghandle(FiguresUnsup.somhitcol)
        close(FiguresUnsup.somhitcol)
    end
    FiguresUnsup.somhitcol=figure;
    plotsomhitscolored(Map,TestsetSOM,WineClasses(~randind))
    
    %% Section 3-d
    % Create smaller SOM and train with all data
    
    Map2=selforgmap([5 5]);
    Map2.trainParam.epochs=200;
    Map2.trainParam.showWindow=false;
    Map2.trainParam.showCommandLine=true;
    Map2=train(Map2,Features_Norm');
    
    % Plot weights if dimension of SOM is small compared to data
    FiguresUnsup.somwd=findobj('type','figure','Name',...
        'SOM Pies - Weight Values');
    if ishghandle(FiguresUnsup.somwd)
        close(FiguresUnsup.somwd)
    end
    FiguresUnsup.somwd=figure('Name','SOM Pies - Weight Values');
    plotsomweightdist(Map2,Features_Norm');
    
    disp(' ')
    disp('Finished Chapter A ')
    
    user_option=input('Would you like to close the figures from this chapter? Y/N [N]:','s');
    if isempty(user_option)
        user_option='n';
    end
    if strcmpi('y',user_option)
              
         if ishghandle(FiguresUnsup.somhitcol)&&strcmp(get(FiguresUnsup.somhitcol,'BeingDeleted'),'off')
            close(FiguresUnsup.somhitcol);
        end

        
        if ishghandle(FiguresUnsup.somnd)&&strcmp(get(FiguresUnsup.somnd,'BeingDeleted'),'off')
            close(FiguresUnsup.somnd);
        end
        
        if ishghandle(FiguresUnsup.somwd)&&strcmp(get(FiguresUnsup.somwd,'BeingDeleted'),'off')
            close(FiguresUnsup.somwd);
        end
        
    end
    
    disp(' ')
    disp(['--------------------------------------------------------------------------------------------';...
        '------------------------------------     END     -------------------------------------------';...
        '--------------------------------------------------------------------------------------------'])
    fprintf('\nProgram paused. Press enter to continue.\n\n');
    pause;

    
end %SOMswitch

if Kmeansswitch % {Chapter B}
    
    disp(' ')
    disp(['--------------------------------------------------------------------------------------------';...
        '------------------------------------  CHAPTER B  -------------------------------------------';...
        '--------------------------------------------------------------------------------------------'])
    
%% Section 4-a 
% Now let us perform unsupervised clustering on the normalised features.
% In this section we will use K-means, an algorithm that attempts to reveal
% a number of hidden clusters or groups in a dataset by means of minimising
% an objective function over the total number of clusters. 

% We know there are 5 quality labels on the data so to begin with let us
% calculate cultering based on that. Notice how k-means is likely to
% converge to a local minimum as it is heuristic approach, however we will
% address this isssue later.

% In this first analysis, we will analyse the influence of initialising the
% clusters using a random sample from the training set vs a preliminary 
% cluster phase on a random 10% subsample using sample initialisation. 
% Notice we use the default squared euclidean distance and train on the 
% complete dataset

% We could use kmedoids instead if we needed to force the cluster centres 
% to be coincident with a training sample.
    
    % Define the number of clusters
    n_groups=5;
    % Kmeans will produce a classification label, a matrix containing the
    % centroids and the total distance per centroid to all the samples
    % assigned to the latter. We will not request distances per sample but
    % that would be a fourth output.
    
    % Start by samplig 
    [KmeansGrp_s,Cen_s,sumd_s]=kmeans(Features_Norm,n_groups,'Start','sample',...
        'Display','final','Distance','sqeuclidean');
    % Start by subsample clustering
    [KmeansGrp_c,Cen_c,sumd_c]=kmeans(Features_Norm,n_groups,'Start','cluster',...
        'Display','final','Distance','sqeuclidean');
    
    
    %% Section 4-b
    % Plot the results from clustering using a single iteration. Let us 
    % choose 3 variables to plot centroids in a 3d space. In this case
    % pH, alcohol and density. Choose the best local optimum result from 
    % the two initialisation options above
    
    if sum(sumd_c)<sum(sumd_s)
        BestGrp=KmeansGrp_c;
        BestCen=Cen_c;
        Bestsum=sumd_c;
    else
        BestGrp=KmeansGrp_s;
        BestCen=Cen_s;
        Bestsum=sumd_s;
    end
    
    
    PlotData=winedata{:,{'Density','pH','Alcohol'}};
    % Prepare a vector with size inversely proportional to total distance
    % to centroid per cluster to plot bigger centroids the closer the
    % assigned datapoints are to them
    areas=1./(Bestsum./sum(Bestsum))*200;
    
    % Extract feature indexes as we know their names.
    ids=[find(not(cellfun('isempty',strfind(winedata.Properties.VariableNames,'Density')))),...
        find(not(cellfun('isempty',strfind(winedata.Properties.VariableNames,'pH')))),...
        find(not(cellfun('isempty',strfind(winedata.Properties.VariableNames,'Alcohol'))))];
    
    % Visualise the data
    FiguresUnsup.f3=findobj('type','figure','Name','Quality Groups');
    if ishghandle(FiguresUnsup.f3)
        close(FiguresUnsup.f3)
    end
    FiguresUnsup.f3=figure('Name','Quality Groups');
    
    scatter3(PlotData(:,1),PlotData(:,2),PlotData(:,3),[],winedata.QCLabel,'.')
    title('Data grouped by Quality Label')
    xlabel('Density');ylabel('pH');zlabel('Alcohol');
    
    
    FiguresUnsup.f4=findobj('type','figure','Name','K-means 1 Iteration');
    if ishghandle(FiguresUnsup.f4)
        close(FiguresUnsup.f4)
    end
    FiguresUnsup.f4=figure('Name','K-means 1 Iteration');
    scatter3(PlotData(:,1),PlotData(:,2),PlotData(:,3),[],BestGrp,'o')
    hold on;
    scatter3(BestCen(:,ids(1)),BestCen(:,ids(2)),BestCen(:,ids(3)),areas,...
        (1:n_groups)','d','filled','LineWidth',2,'MarkerEdgeColor','k')
    hold off
    drawnow;
    title('Data grouped by K-Means Clustering & Centroids (Bigger the smaller sum of distances)')
    xlabel('Density');ylabel('pH');zlabel('Alcohol');
    
    %% Section 4-c 
    % Knowing the local minima problem with K-means, we will now repeat the
    % clustering but using repetitions. This should improve performance For
    % this analysis we will also change the distance matrix to cityblock.
    
    if Parallelswitch
        % Start the parallel pool to ensure more predictable results when
        % distributing the models.
        poolobj=parpool('local',n_cores);
        
        % Train K-means using both initialisation strategies
        [KmeansGrpIte_s,CenIte_s,sumdIte_s]=kmeans(Features_Norm,n_groups,'Start','sample',...
            'Display','final','Distance','cityblock','Replicates',50,'Options',statset('UseParallel',1));
        
        [KmeansGrpIte_c,CenIte_c,sumdIte_c]=kmeans(Features_Norm,n_groups,'Start','cluster',...
            'Display','final','Distance','cityblock','Replicates',50,'Options',statset('UseParallel',1));
        
    else
         % Train K-means using both initialisation strategies - no
         % parallelisation
        [KmeansGrpIte_s,CenIte_s,sumdIte_s]=kmeans(Features_Norm,n_groups,'Start','sample',...
            'Display','final','Distance','cityblock','Replicates',20);
         
        [KmeansGrpIte_c,CenIte_c,sumdIte_c]=kmeans(Features_Norm,n_groups,'Start','cluster',...
            'Display','final','Distance','cityblock','Replicates',20);
    end
    
    %% Section 4-d 
    % Extract the best result again
    
    if sum(sumdIte_c)<sum(sumdIte_s)
        BestGrpIte=KmeansGrpIte_c;
        BestCenIte=CenIte_c;
        BestsumIte=sumdIte_c;
    else
        BestGrpIte=KmeansGrpIte_s;
        BestCenIte=CenIte_s;
        BestsumIte=sumdIte_s;
    end
    
    % Now look for the combination of 3 features that produce the maximum 
    % separation of the centroids in a 3D plot
    
    allcomb=nchoosek(1:n_features,3);
    ncomb=size(allcomb,1);
    
    % Extract distances
    dis=zeros(ncomb,1);
    for jj=1:ncomb
        dis(jj)=sum(pdist(BestCenIte(:,allcomb(jj,:))));
    end
    
    % Find maximum distance combination
    [~,maxj]=max(dis);
    bestcomb=allcomb(maxj,:);
    bestfeat=winedata.Properties.VariableNames(bestcomb);
    
    % Display the selected 3 features
    disp('The three features with highest 3D centroid separation are:')
    disp(bestfeat)
    
    % Extract the data from such features and generate an area vector 
    % representing the quality of the cluster (smaller distance bigger area
    % in the plot)
    PlotDataIte=winedata{:,bestfeat};
    areasIte=1./(BestsumIte./sum(BestsumIte))*200;
    
    % Plot the groups using the selected features and the centroids as 
    % diamonds with area proportional to the quality of their clusters
    FiguresUnsup.f5=findobj('type','figure','Name','K-means Multiple Iterations');
    if ishghandle(FiguresUnsup.f5)
        close(FiguresUnsup.f5)
    end
    FiguresUnsup.f5=figure('Name','K-means Multiple Iterations');
    scatter3(PlotDataIte(:,1),PlotDataIte(:,2),PlotDataIte(:,3),[],BestGrpIte,'o')
    hold on;
    scatter3(BestCenIte(:,bestcomb(1)),BestCenIte(:,bestcomb(2)),...
        BestCenIte(:,bestcomb(3)),areasIte,(1:n_groups)','d','filled',...
        'LineWidth',2,'MarkerEdgeColor','k')
    hold off
    drawnow;
    title('Data grouped by K-Means Clustering & Centroids (Bigger the smaller sum of distances)')
    xlabel(bestfeat{1});ylabel(bestfeat{2});zlabel(bestfeat{3});
    
    
    %% Section 4-e 
    % Interpretation of results for best iterative clustering
    
    % Plot how each feature characterises the 5 groups extracted with 
    % k-means as well as the centroids. In order to expand the
    % visualisation beyond mean value, we will use the 0.5 quantile option
    
    FiguresUnsup.f6=findobj('type','figure','Name','Parallelcoords by group');
    if ishghandle(FiguresUnsup.f6)
        close(FiguresUnsup.f6)
    end
    FiguresUnsup.f6=figure('Name','Parallelcoords by group');
    parallelcoords(Features_Norm,'Group',BestGrpIte,'Quantile',0.5)
    grid on;
    ax = gca;
    ax.XTickLabel=winedata.Properties.VariableNames(1,1:end-1);
    
    FiguresUnsup.f7=findobj('type','figure','Name','Parallelcoords by centroids');
    if ishghandle(FiguresUnsup.f7)
        close(FiguresUnsup.f7)
    end
    FiguresUnsup.f7=figure('Name','Parallelcoords by centroids');
    parallelcoords(BestCenIte,'Group',1:5)
    grid on;
    ax = gca;
    ax.XTickLabel=winedata.Properties.VariableNames(1,1:end-1);
    
    % In order to perform a pie plot of the features we have to normalise 
    % their values into the interval [0,1]. Note how this scaling should 
    % have been applied to the original features but for simplicity we will
    % apply it on top of the already standardised features.
    
    % Display pie plots of each feature per centroid as well as the stars
    % corresponding to each centroid and the feature values that
    % characterise it. 
    
    BestCenIte01=zeros(size(BestCenIte));
    for jj=1:size(BestCenIte01,2)
        rangecol=max(BestCenIte(:,jj))-min(BestCenIte(:,jj));
        BestCenIte01(:,jj)=(BestCenIte(:,jj)-min(BestCenIte(:,jj)))./rangecol;
    end
    % Add small value to avoid zeros (Pie plot would be upset)
    BestCenIte01(BestCenIte01==0)=1e-4;
    
    
    FiguresUnsup.f8=findobj('type','figure','Name','Centroids as Pie');
    if ishghandle(FiguresUnsup.f8)
        close(FiguresUnsup.f8)
    end
    FiguresUnsup.f8=figure('Name','Centroids as Pie');
    for jj=1:n_groups
        subplot(2,3,jj)
        pie(BestCenIte01(jj,:),winedata.Properties.VariableNames(1,1:end-1));
    end
    % Add a glyphplot to show the characteristics of each centroid 
    subplot(2,3,jj+1)
    glyphplot(BestCenIte01,'standardize','off','VarLabels',...
        winedata.Properties.VariableNames(1,1:end-1))
    
    
    %% Section 4-f
    % Now let us look at how well the k means classification matches the QC
    % Labels in the source data
    
    % We can use the crosstab function for a quick overview
    [counts_kmeans,~,~,labels_kmeans]=crosstab(BestGrpIte,winedata.QCLabel);
    
    FiguresUnsup.f8=findobj('type','figure','Name','Clustering Accuracy');
    if ishghandle(FiguresUnsup.f8)
        close(FiguresUnsup.f8)
    end
    FiguresUnsup.f8=figure('Name','Clustering Accuracy');
    bar3(counts_kmeans)
    set(gca,'XTick',1:5,'XTickLabel',labels_kmeans(:,2))
    
    
    %% Section 4-g
    % Now we will evaluate the single iteration clustering against the 
    % iterative by means of silhouette. This is a graphical representation
    % of each training sample by means of a normalised value between -1 and
    % 1 measuring how close the point is to the rest in the same cluster.
    % High values indicate good clustering performance. 
    
    % Silhouette may require excessive time to process hence it is better 
    % to downsample the data
    
    [Xdowns,idd]=datasample(Features_Norm,1000);
    
    BestGrpSub=BestGrp(idd);
    BestGrpIteSub=BestGrpIte(idd);
    
    FiguresUnsup.f9=findobj('type','figure','Name','Silhouette');
    if ishghandle(FiguresUnsup.f9)
        close(FiguresUnsup.f9)
    end
    FiguresUnsup.f9=figure('Name','Silhouette');
    
    subplot(2,1,1)
    silhouette(Xdowns,BestGrpSub);
    title('Clusters on single iteration using sqeuclidean distance')
    subplot(2,1,2)
    silhouette(Xdowns,BestGrpIteSub);
    title('Clusters on multiple iterations using sqeuclidean distance')
    
    %% Section 4-g
    % So far we have tried to classify all the wine samples based on its
    % quality label but we have ignored the second dimension of the
    % problem. We have data on the wine type (red, white). Let us now find
    % an optimal cluster for only the white wine data (hence the title
    % Vinho verde!). The data for white wine will be rescaled to a [0,1]
    % interval before any processing.
    
    idwhite=wineinfo=='White';
    
    % Extract data from white wine, remove NaN and inf.
    WhiteData=DataClean(idwhite,1:end-1);
    % Rescale data to the interval [0,1]
    WhiteData_Norm=normTable(WhiteData,{'Range'});
    Features_w_Norm=WhiteData_Norm{:,:};
    
    QCLabel_w=WineClasses(idwhite);
    
    % Downsample the data to use silhouette as a performance indicator.
    [Feat_w_sub,idwd]=datasample(Features_w_Norm,2000);
    
    % Preallocate the cluster indication variable to be used in a loop to
    % check the results using 3 to 6 clusters
    ClusGrp=zeros(size(Features_w_Norm,1),4);
    
    FiguresUnsup.f10=findobj('type','figure','Name','Manual Best K Search - White Wine');
    if ishghandle(FiguresUnsup.f10)
        close(FiguresUnsup.f10)
    end
    FiguresUnsup.f10=figure('Name','Manual Best K Search - White Wine');
    for nclus=3:6
        if Parallelswitch
            
            ClusGrp(:,nclus-2)=kmeans(Features_w_Norm,nclus,'Distance','cityblock',...
                'Replicates',50,'Options',statset('UseParallel',1));
            
        else
            ClusGrp(:,nclus-2)=kmeans(Features_w_Norm,nclus,'Distance','cityblock',...
                'Replicates',20);
        end
        subplot(2,2,nclus-2)
        silhouette(Feat_w_sub,ClusGrp(idwd,nclus-2));
        title(['Number Clusters = ',num2str(nclus)]);
    end
    
    if Parallelswitch
        delete(gcp('nocreate'))
    end
    
    
    %% Section 4-h
    % Now we will find the optimal number of clusters by means of the gap
    % criterion. [Tibshirani, R., G. Walther, and T. Hastie. "Estimating 
    % the number of clusters in a data set via the gap statistic." Journal 
    % of the Royal Statistical Society: Series B. Vol. 63, Part 2, 2001, 
    % pp.411–423].
    
    % Create an evalclusters object. We will use the downsample data to
    % speed up computation
    
    % We will try two different search gap criteria for the optimal number 
    % of clusters as available in the default options (when using the gap
    % metric)
    
    % Global max:
    % |---------------------------|
    % | Gap(K)>=GAPMAX-SE(GAPMAX) |
    % |---------------------------|
    % where K is the number of clusters, Gap(K) is the gap value for the 
    % clustering solution with K clusters, GAPMAX is the largest gap value, 
    % and SE(GAPMAX) is the standard error corresponding to the largest gap
    % value.
    
    ClusEval_gap_global=evalclusters(Feat_w_sub,'kmeans','gap','KList',...
        (1:6),'SearchMethod','globalMaxSE','Distance','cityblock');
    
    % First max: Pick smallest number of clusters satisfying 
    % |--------------------------|
    % | Gap(K)>=Gap(K+1)-SE(K+1) |
    % |--------------------------|
    % where K is the number of clusters, Gap(K) is the gap value for the 
    % clustering solution with K clusters, and SE(K + 1) is the standard 
    % error of the clustering solution with K + 1 clusters. 
    
    ClusEval_gap_first=evalclusters(Feat_w_sub,'kmeans','gap','KList',...
       (1:6),'SearchMethod','firstMaxSE','Distance','cityblock');
    
    disp(['Optimal number of clusters for white wine samples using the ',...
        'gap criterion and global maximisation solution is:'])
    disp(ClusEval_gap_global.OptimalK)
    
    FiguresUnsup.f11=findobj('type','figure','Name','Evalclusters BestK Search White Wine');
    if ishghandle(FiguresUnsup.f11)
        close(FiguresUnsup.f11)
    end
    FiguresUnsup.f11=figure('Name','Evalclusters BestK Search White Wine');
    
    plot(ClusEval_gap_global)
    
    
    disp(['Optimal number of clusters for white wine samples using the ',...
        'gap criterion and first maximisation solution is:'])
    disp(ClusEval_gap_first.OptimalK)
    
    %% Section 4-i 
    % Now we will compute the Kbest clusters as specified by the previous
    % optimisation problem in section 4-h and plot these in a 3D base given
    % by the features that produce the greates separation of cluster
    % centroids
    
   [KmeansBestK,CenBestK,sumdBestK]=kmeans(Features_w_Norm,ClusEval_gap_global.OptimalK...
       ,'Start','cluster','Display','final','Distance','cityblock','Replicates',20);
   
    % Now look for the combination of 3 features that produce the max 
    % separation of the groups in a 3D plot
    
    allcomb=nchoosek(1:n_features,3);
    ncomb=size(allcomb,1);
    
    % Extract distances
    dis=zeros(ncomb,1);
    for jj=1:ncomb
        dis(jj)=sum(pdist(CenBestK(:,allcomb(jj,:))));
    end
    
    % Find maximum centroid distances
    [~,maxj]=max(dis);
    bestcomb=allcomb(maxj,:);
    bestfeat=winedata.Properties.VariableNames(bestcomb);
    
    % Display the selected 3 features
    disp('The three features with highest 3D centroid separation for White wine are:')
    disp(bestfeat)
    
    % Extract the data from the selected features and generate an area 
    % vector representing the quality of the cluster (smaller distance 
    % bigger area in the plot)
    PlotDataBestK=winedata{idwhite,bestfeat};
    areasBestK=1./(sumdBestK./sum(sumdBestK))*200;
    
    % Plot the groups using the selected features and the centroids as 
    % diamonds with area proportional to the quality of their clusters
    FiguresUnsup.f12=findobj('type','figure','Name','K-means Best K - White Wine');
    if ishghandle(FiguresUnsup.f12)
        close(FiguresUnsup.f12)
    end
    FiguresUnsup.f12=figure('Name','K-means Best K - White Wine');
    scatter3(PlotDataBestK(:,1),PlotDataBestK(:,2),PlotDataBestK(:,3),[],KmeansBestK,'o')
    hold on;
    scatter3(CenBestK(:,bestcomb(1)),CenBestK(:,bestcomb(2)),...
        CenBestK(:,bestcomb(3)),areasBestK,(1:ClusEval_gap_global.OptimalK)',...
        'd','filled','LineWidth',2,'MarkerEdgeColor','k')
    hold off
    drawnow;
    title('WHite Wine Data grouped by Best K & Centroids (Bigger the smaller sum of distances)')
    xlabel(bestfeat{1});ylabel(bestfeat{2});zlabel(bestfeat{3});
    ylim([-4 4])
    xlim([-2 3]) 
    
    disp(' ')
    disp('Finished Chapter B')
    
    user_option=input('Would you like to close the figures from this chapter? Y/N [N]:','s');
    if isempty(user_option)
        user_option='n';
    end
    if strcmpi('y',user_option)
        
        for vv=4:12
            figname=strcat('f',num2str(vv));
            if ishghandle(FiguresUnsup.(figname))&&strcmp(get(FiguresUnsup.(figname),'BeingDeleted'),'off')
                close(FiguresUnsup.(figname));
            end
        end
   
    end
    
    
    disp(' ')
    disp(['--------------------------------------------------------------------------------------------';...
        '------------------------------------     END     -------------------------------------------';...
        '--------------------------------------------------------------------------------------------'])
    
    fprintf('\nProgram paused. Press enter to continue.\n\n');
    pause;

    
end %if Kmeansswitch

if Hierarchswitch % {Chapter C}
    
    disp(' ')
    disp(['--------------------------------------------------------------------------------------------';...
        '------------------------------------  CHAPTER C  -------------------------------------------';...
        '--------------------------------------------------------------------------------------------'])
    
    %% Section 5-a 
    % We will study now hierarchical clustering on the wine dataset.
    % Hierarchical clustering will try to cluster the data in two steps.
    % Firstly, it will identify the hierarchical structure of the data by
    % grouping points that are close in the m dimensional space (m=number 
    % of features). Secondly, it will cluster such structure to minimise 
    % a specific distance objective function.
    
    % In this section we will use again the data from all wine samples
    % standardised to 0 mean and unitary std
    
    % For easier visualisaton reasons, ee will use a downsized sample of 
    % the data.
    
    [Features_Hier,idhier]=datasample(Features_Norm,500);
    
    % Find the hierarchical structure, using Ward's inner squared distance
    % (minimum variance) and Euclidean distance
    HierMat=linkage(Features_Hier,'ward');
    
    % Visualise the hierarchical structure using dendrogram
    FiguresUnsup.f13=findobj('type','figure','Name','Hierarchical Structure');
    if ishghandle(FiguresUnsup.f13)
        close(FiguresUnsup.f13)
    end
    FiguresUnsup.f13=figure('Name','Hierarchical Structure');
    dendrogram(HierMat);
    xlabel('Wine Data Observations')
    ylabel('Euclidean Distance (Ward)')
    
    
    FiguresUnsup.f14=findobj('type','figure','Name','Hierarchical Structure Colour');
    if ishghandle(FiguresUnsup.f14)
        close(FiguresUnsup.f14)
    end
    FiguresUnsup.f14=figure('Name','Hierarchical Structure Colour');
    % Plot the tree again and change colour for any cluster whose linkage
    % distance is less that the average + 4*std in this case
    dendrogram(HierMat,'ColorThreshold',mean(HierMat(:,3))+4*std(HierMat(:,3)))
    xlabel('Wine Data Observations')
    ylabel('Euclidean Distance (Ward)')
    
    %% Section 5-b
    % Now let us look at how well the 2 groups classification matches the 
    % type of wine in the source data.
    % Let us try to cluster the data in red/white wine by limiting the
    % number of clusters to 2. Notice how it is possible to impose
    % constraints to the number of elements at each tree leaf or the
    % inconsistency between different branches in the tree, but we will not
    % use these 
    HierClus=cluster(HierMat,'maxclust',2);
    
    % Compare results to true label and use a histogram to show results. 
    [counts_rw,~,~,labels_rw]=crosstab(HierClus,wineinfo(idhier));
    
    FiguresUnsup.f15=findobj('type','figure','Name','Red / White Hierarchical Clustering');
    if ishghandle(FiguresUnsup.f15)
        close(FiguresUnsup.f15)
    end
    FiguresUnsup.f15=figure('Name','Red / White Hierarchical Clustering');
    bar(counts_rw,'stacked')
    set(gca,'XTick',1:5,'XTickLabel',labels_rw(:,2))
    legend(categories(wineinfo),'Location','EastOutside')
    xlabel('Cluster Number')
    
    %% Section 5-c
    % We can use the Cophenetic Correlation Coefficient to assess the
    % quality of this hierarchical tree (how the tree represents
    % the dissimilarities between observations). A value close to 1
    % indicates high quality hierarchical structure. 
    % To compute the CCC we need to use the pairwise distances between
    % observations
    
    Pdist_Hier=pdist(Features_Hier);
    
    ccc=cophenet(HierMat,Pdist_Hier);
    
    disp(['The Cophenetic Correlation Coefficient for the Hierarchical tree',...
        ' using a downsized sample is:']);
    disp(num2str(ccc))
    
    %% Section 5-d 
    % Plot the results to understand how red vs white wine is characterised
    % in terms of the classification performed by the hierarchical
    % structure
    
    FiguresUnsup.f16=findobj('type','figure','Name','Red / White ParallelCoords');
    if ishghandle(FiguresUnsup.f16)
        close(FiguresUnsup.f16)
    end
    FiguresUnsup.f16=figure('Name','Red / White ParallelCoords');
    
    % Plot the coordinates of the 2 clusters using the 0.25 quantiles
    parallelcoords(Features_Hier,'Group',HierClus,'Quantile',0.25)
    legend({'Red','','','White','',''})
    
    FiguresUnsup.f17=findobj('type','figure','Name','Red / White Group Stats - Truth vs Classified');
    ax = gca;
    ax.XTickLabel=winedata.Properties.VariableNames(1,1:end-1);
    if ishghandle(FiguresUnsup.f17)
        close(FiguresUnsup.f17)
    end
    FiguresUnsup.f17=figure('Name','Red / White Group Stats - Truth vs Classified');
    % Plot the mean values of the features for red and white wines
    bar(grpstats(Features_Hier,HierClus))
    ax = gca;
    ax.XTickLabel=categories(wineinfo);
    legend(winedata.Properties.VariableNames(1,1:end-1),'Location','eastoutside')
    hold all;
    bar(grpstats(Features_Hier,wineinfo(idhier)),'FaceAlpha',0.3,...
        'EdgeColor','r')
    text(min(ax.XTick),(ax.YTick(end-1)+ax.YTick(end))/2,...
        {'Filled - Classification Result'; 'Trans. - True Class'});
    
    disp(' ')
    disp('Finished Chapter C ')
    
    user_option=input('Would you like to close the figures from this chapter? Y/N [N]:','s');
    if isempty(user_option)
        user_option='n';
    end
    if strcmpi('y',user_option)
        
        for vv=13:17
            figname=strcat('f',num2str(vv));
            if ishghandle(FiguresUnsup.(figname))&&strcmp(get(FiguresUnsup.(figname),'BeingDeleted'),'off')
                close(FiguresUnsup.(figname));
            end
        end
       
    end
    
    disp(' ')
    disp(['--------------------------------------------------------------------------------------------';...
        '------------------------------------     END     -------------------------------------------';...
        '--------------------------------------------------------------------------------------------'])
    
    fprintf('\nProgram paused. Press enter to continue.\n\n');
    pause;

    
end %if Hierarchswitch

if GMMswitch % {Chapter D}
    
    disp(' ')
    disp(['--------------------------------------------------------------------------------------------';...
        '------------------------------------  CHAPTER D  -------------------------------------------';...
        '--------------------------------------------------------------------------------------------'])
    
    %% Section 6-a 
    % In this section we will cluster the data using a mixture of
    % Gaussians. The idea behind is to model each cluster as a multivariate
    % Gaussian distribution and assign each point to a class based on the
    % probability of pertenence to each distribution. Hence, the clustering
    % is probabilistic rather than binary
    
    % First we will perform a simple analysis on 2D in order to visualise
    % GMM. First choose 2 random quality categories
    qua_2D=[];
    while(isempty(qua_2D)||qua_2D(1)==qua_2D(2))
        qua_2D=diag(randi(length(LabelsCell),2,'int8'))';
    end
    
    id2D=(wineinfo=='White')&((WineClasses==LabelsCell{qua_2D(1)})|...
        (WineClasses==LabelsCell{qua_2D(2)}));
    
    % Pick 2 random features
    col_2D=[];
    while(isempty(col_2D)||col_2D(1)==col_2D(2))
        col_2D=diag(randi(n_features,2,'int8'))';
    end
    % Extract random features from original unmodified features
    Feat_2D=Features(id2D,col_2D);
    % Standardised the selected features and extract their names
    Feat_2D=zscore(Feat_2D);
    Feat_2D_Names=DataClean.Properties.VariableNames(col_2D);
    % Extract quality data for 2D features and remove inexistent
    % categories. This removal is needed because the categorical array
    % where these points are coming from was created considering five
    % labels. 
    Feat_2D_qc=WineClasses(id2D);
    Feat_2D_qc=removecats(Feat_2D_qc); 
    
    % Fit a mixture of 2 Gaussians to the 2D data
    GM_2D=fitgmdist(Feat_2D,2);
    
    % Create a grid to plot Gaussians and classify points. Notice we
    % normalised features to have 0 mean and 1 std hence the interval 
    % chosen
    [F1,F2]=meshgrid(linspace(-3,3));
    
    % Notice the mesh points are vectorised to evaluate the Gaussians
    Pdf_mesh=pdf(GM_2D,[F1(:),F2(:)]);
    
    %Plot the 2D GMM
    FiguresUnsup.f18=findobj('type','figure','Name','2D GMM');
    if ishghandle(FiguresUnsup.f18)
        close(FiguresUnsup.f18)
    end
    FiguresUnsup.f18=figure('Name','2D GMM');
    
    gscatter(Feat_2D(:,1),Feat_2D(:,2),Feat_2D_qc,[1 128/255 0;.3765 .3765 .3765],'.')
    hold on;
    % Reshape mesh points to plot contour
    contour(F1,F2,reshape(Pdf_mesh(:,1),size(F1)),25)
    hold off;
    xlabel(Feat_2D_Names{1}); ylabel(Feat_2D_Names{2});
    
    
    %% Section 6-b
    % Cluster the 2D mesh data and check the quality of the clustering.
    % Notice again mesh data is vectorised to evaluate the clustering. 
    GM_2D_clus=cluster(GM_2D,[F1(:) F2(:)]);
    
     FiguresUnsup.f19=findobj('type','figure','Name','2D GMM Mesh Clustering');
    if ishghandle(FiguresUnsup.f19)
        close(FiguresUnsup.f19)
    end
    FiguresUnsup.f19=figure('Name','2D GMM Mesh Clustering');
    
    gscatter(F1(:),F2(:),GM_2D_clus,[1 128/255 0;.3765 .3765 .3765],'.')
    hold on;
    % Now plot points showing ground truth. Notice in gscatter the order of
    % inputs is: XData,YData,Labels,Colours,Symbols and Sizes.
    gscatter(Feat_2D(:,1),Feat_2D(:,2),Feat_2D_qc,...
        [1 128/255 0;.3765 .3765 .3765],'x',[10 10])
    % Reshape mesh points to plot contour
    contour(F1,F2,reshape(Pdf_mesh(:,1),size(F1)),25)
    hold off;
    xlabel(Feat_2D_Names{1}); ylabel(Feat_2D_Names{2});
    xlim([-3 3]); ylim([-3 3]);
    
    % Let us split our data in a hold-out partition with 20% test data 
    % points
    Part=cvpartition(size(Feat_2D,1),'HoldOut',0.2);
    
    % Train the model on the training set
    GM_2D_train=fitgmdist(Feat_2D(Part.training,:),2);
    % Extract pdf on mesh for the new model extracted using only the 
    % training set
    Pdf_mesh_train=pdf(GM_2D_train,[F1(:),F2(:)]);
    % Test the performance of the model on the test set
    GM_2D_clus_test=cluster(GM_2D_train,Feat_2D(Part.test,:));
    % Plot results
    FiguresUnsup.f20=findobj('type','figure','Name','2D GMM Features Clustering');
    if ishghandle(FiguresUnsup.f20)
        close(FiguresUnsup.f20)
    end
    FiguresUnsup.f20=figure('Name','2D GMM Features Clustering');
    
    gscatter(Feat_2D(Part.test,1),Feat_2D(Part.test,2),GM_2D_clus_test,[1 128/255 0;.3765 .3765 .3765],'.')
    hold on;
    % Reshape mesh points to plot contour
    contour(F1,F2,reshape(Pdf_mesh(:,1),size(F1)),25)
    hold off;
    xlabel(Feat_2D_Names{1}); ylabel(Feat_2D_Names{2});
    xlim([-3 3]); ylim([-3 3]);
    
    %% Section 6-c
    % We will use GMM to classify the wine in red/white using an
    % evalclusters object. In the latter the criterion used to select an
    % optimal number of clusters is the silhouette metric in a range of K
    % from 2 to 5. 
    
    % Downsample data
    [Xdowns,idd]=datasample(Features_Norm,2000);
    
    GMM_clus_obj_rw=evalclusters(Xdowns,'gmdistribution','silhouette','KList',2:5);
    FiguresUnsup.f21=findobj('type','figure','Name','GMM Evalclusters White/Red');
    if ishghandle(FiguresUnsup.f21)
        close(FiguresUnsup.f21)
    end
    FiguresUnsup.f21=figure('Name','GMM Evalclusters White/Red');
    plot(GMM_clus_obj_rw);
    
    % Let us now assume we wanted to discriminate white from red wine by
    % means of using a mixture of Gaussians including all the features in
    % the normalised data. 
    
    % First we would fit the distribution
    GMM_all=fitgmdist(Features_Norm,GMM_clus_obj_rw.OptimalK);
    % Cluster the data according to the optimal K (2 for red + white wine
    % data)
    GMM_label=cluster(GMM_all,Features_Norm);
    
    % Get the precision and recall measurements. Here true is assumed to be
    % a wine labelled as white and false a wine sample labelled as red. We
    % use crosstab to extract number of true and false positives. Do the 
    % calculation only for 2 clusters. 
    [GMM_all_cross,~,~,GMM_all_labels] = crosstab(GMM_label,wineinfo);
    
    % Identify which label corresponds to white wine
    if GMM_clus_obj_rw.OptimalK==2
    if strcmp(GMM_all_labels{2,1},'White')
        Whitecol=1;
        Redcol=2;
    else
        Redcol=1;
        Whitecol=2;
    end
    
    [~,Whiterow]=max(GMM_all_cross(:,Whitecol));
    [~,Redrow]=max(GMM_all_cross(:,Redcol));
    
    if Redrow~=Whiterow
    % Calculate Precision [P=n true pos/ (n true pos + n false positive)]
    % Here label white corresponds to the first group in the GMM
    % clustering. 
  
    Precision=GMM_all_cross(Whiterow,Whitecol)/(GMM_all_cross(Whiterow,Whitecol)+GMM_all_cross(Whiterow,Redcol));
    disp(['White Wine Classification Precision P = ',num2str(Precision)]);
    
    % Calculate Recall [R=n true pos/ (n true pos + n false negative)]
    Recall=GMM_all_cross(Whiterow,Whitecol)/(GMM_all_cross(Whiterow,Whitecol)+GMM_all_cross(Redrow,Whitecol));
    disp(['White Wine Classification Recall R = ',num2str(Recall)]);
    
    % Calculate F Score [F=2*P*R/(P+R)]. When the Fscore is close to the
    % unit it means the classifier shows excellent performance for the data
    % used to evaluate P and R. Notice here training and testing sets are
    % the same and equal to the total available dataset
    F_score=2*Precision*Recall/(Precision+Recall);
    disp(['White Wine Classification F_Score F = ',num2str(F_score)]);
    end
    end
    
    
    %% Section 6-d
    
    % Finally we can use the white wine data to fit a multivariate Gaussian
    % for each of the quality labels
    
    idwhite=wineinfo=='White';
    
    % Extract data from white wine
    Features_w_Norm=Features_Norm(idwhite,:);
    
    QCLabel_w=WineClasses(idwhite);
    
    % Downsample the data to use silhouette as performance measurement.
    [Feat_w_sub,idwd]=datasample(Features_w_Norm,2000);
    
    % For predictable results
    rng(1234);
    
    % In case the data is poorly conditioned to fit multiple Gaussians it 
    % is possible to avoid ill conditioned or singular covariances matrices
    % by means of specifying that the covariance matrix should be diagonal.
    % In order to use such setting in combination with evalclusters we need
    % to define a clustering function handle to pass to evalclusters. We 
    % can also specify a number of replicates when fitting the mixture of
    % Gaussian model as we now the EM algorithm is prone to local minima 
    
    clusfunc=@(Data,k) cluster(fitgmdist(Data,k,'CovType','diagonal',...
        'Replicates',10,'Options',statset('MaxIter',200)),Data);
    
    % Create evalclusters object for white wine samples according to QC
    % label
    GMM_clus_obj_w_qc=evalclusters(Feat_w_sub,clusfunc,'silhouette',...
        'KList',2:7);
    
    FiguresUnsup.f22=findobj('type','figure','Name','GMM Evalclusters White-QC');
    if ishghandle(FiguresUnsup.f22)
        close(FiguresUnsup.f22)
    end
    FiguresUnsup.f22=figure('Name','GMM Evalclusters White-QC');
    plot(GMM_clus_obj_w_qc);
    
    % Now we can cluster using a mixture of Gaussians with the optimal K
    % value (number of clusters IN WHITE DATA based on the Silhouette
    % metric)
    
    GMM_w=fitgmdist(Features_w_Norm,GMM_clus_obj_w_qc.OptimalK,...
        'CovType','diagonal','Replicates',10);
    
    [GMM_w_Label,~,GMM_w_pr]=cluster(GMM_w,Features_w_Norm);
    
    % View average feature values for each of the Gaussian clusters.
    % (Mean of clusters)
    GMM_w_means=grpstats(Features_w_Norm,GMM_w_Label);
    
    FiguresUnsup.f23=findobj('type','figure','Name','GMM White Wine Cluster Means');
    if ishghandle(FiguresUnsup.f23)
        close(FiguresUnsup.f23)
    end
    FiguresUnsup.f23=figure('Name','GMM White Wine Cluster Means');
    bar(GMM_w_means)
    legend(Features_Names,'Location','eastoutside');
    
    % Find 3 features with biggest difference across groups
    [~,sortedInd]=sort(range(GMM_w_means)./min(GMM_w_means),'descend');
    
    % Let us plot the points with a colour scale based on the probability 
    % of pertenence to each cluster.
    % Create a colormap for each of the observations according to the
    % probability of being assigned to each of the Gaussians
    
    % To populate both colour maps we create as many absolute colour maps
    % as Gaussians we have to plot and scale each of this with the
    % probability of pertenence of each point to each of the Gaussians
    c_map=GMM_w_pr*summer(size(GMM_w_pr,2));
    GMM_pointsizes=10*GMM_w_Label;
   
    
    FiguresUnsup.f24=findobj('type','figure','Name','GMM White Wine Coloured by Probability');
    if ishghandle(FiguresUnsup.f24)
        close(FiguresUnsup.f24)
    end
    FiguresUnsup.f24=figure('Name','GMM White Wine Coloured by Probability');
    scatter3(Features_w_Norm(:,sortedInd(1)),Features_w_Norm(:,sortedInd(2)),...
        Features_w_Norm(:,sortedInd(3)),GMM_pointsizes,c_map,'o','filled');
    xlabel(Features_Names(:,sortedInd(1)))
    ylabel(Features_Names(:,sortedInd(2)))
    zlabel(Features_Names(:,sortedInd(3))) 
    
    % Zoom in to 5% and 95% quantiles 
    % (We can remove outliers)
    quantiles = quantile(Features_w_Norm(:,sortedInd(1:3)),[0.05 0.95]);
    quantiles = quantiles(:); % Vectorise values
    axis(quantiles)
    
    disp(' ')
    disp('Finished Chapter D')
    
    user_option=input('Would you like to close the figures from this chapter? Y/N [N]:','s');
    if isempty(user_option)
        user_option='n';
    end
    if strcmpi('y',user_option)
        
        for vv=18:24
            figname=strcat('f',num2str(vv));
            if ishghandle(FiguresUnsup.(figname))&&strcmp(get(FiguresUnsup.(figname),'BeingDeleted'),'off')
                close(FiguresUnsup.(figname));
            end
        end
        
        if ishghandle(FiguresUnsup.somnd)&&strcmp(get(FiguresUnsup.somnd,'BeingDeleted'),'off')
            close(FiguresUnsup.somnd);
        end
        
    end

    disp(' ')
    disp(['--------------------------------------------------------------------------------------------';...
        '------------------------------------     END     -------------------------------------------';...
        '--------------------------------------------------------------------------------------------'])
    
    
end %if GMMswitch


%--------------------------------------------------------------------------
%--------------------------------------------------------------------------


































































