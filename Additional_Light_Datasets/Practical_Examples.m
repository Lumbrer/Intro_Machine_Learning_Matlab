%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% $Author: Lumbrer $    $Date: 2017/01/11 $    $Revision: 0.1$
% Copyright: Francisco Lumbreras
%
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% File Name: Practical_Examples.m
% Description: Script to apply Machine Learning algorithms to simple
% datasets using the ML Toolbox capabilities.
%
% In the first section of the script a number of switches are defined to
% activate or deactivate the different chapters of the script. These can
% all run independently from each other. Each chapter will refer to a
% differente data source. Notice all the data sources used for the
% following work are copyright of MathWorks Inc. and are used for
% illustration purposes here.
%
% The following chapters are defined:
%
% A) Inspect corporate bonds data in terms of Price, Yield To Mature,
% Coupon Rate and Current Yield. The data includes some non rated bonds as
% well as outliers that need to be removed for the sake of clean data
% visualisation. Then use the numeric data to cluster the Bonds and find an
% optimal number of cluster using the Silhouette metric.
%
% B) In this chapter we will cluster inertial data measured using a
% smartphone while performing different physical activities. In this
% section we will use the Error Correcting Output Codes approach to train a
% multiclass SVM classifier.
%
% C) In chapter three we will use the data from the open repository on
% satellite image based terrain classification as available in the UCI
% website - [https://archive.ics.uci.edu/ml/datasets/Urban+Land+Cover]
%
% D) This chapter uses data from a marketing campaign performed by a bank
% in an attempt to predict the profile of the customer that accepted the
% product introduced and the circumstances that led to such decision. The
% presence of categorical features in this sections will be used when
% training models like a Naive Bayes classifier.
%
% E) The last chapter focuses on predicting cre3dit ratings based on a
% number of finantial features. Here we will use ensemble learning to
% improve generalisation error and study how noise on the measurements
% affects the robustness of the algorithm.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% ---------------- CODE HERE ----------------------------------------------
clc;
close all force;
clear all;
set(0,'DefaultTextInterpreter','none')
% Add path to files required
addpath(genpath(pwd));
% Create a struct to store figures
FiguresExe=struct;


%% Define switches to activate each of the chapters of the script
% Switches for chapters A, B, C, D & E
CBswitch=boolean(1);         % A - Activate Coorporate Bonds
ADswitch=boolean(1);         % B - Activate Human Activity
SIswitch=boolean(1);         % C - Activate Satellite Image
BMswitch=boolean(1);         % D - Activate Bank Marketing
CRswitch=boolean(1);         % E - Activate Credit Ratings

% Switch to identify if multiple parallel workers are available locally
Parallelswitch=true;
% Define the number of cores in the parallel pool available locally
n_cores=4;
% Set random seed for reproducibility
rng(1234)


if CBswitch % {CHAPTER A}
    
    
    disp(' ')
    disp(['--------------------------------------------------------------------------------------------';...
        '------------------------------------  CHAPTER A  -------------------------------------------';...
        '--------------------------------------------------------------------------------------------'])
    %% Section 1-a
    % We will import and display coorporate bonds from a sample of data but
    % keeping only those entries with positive Yield To Maturity values and
    % removing outliers as well as the data without rating.
    
    % Read the data from all Bonds (includes corporate and municipal)
    B_Data=readtable('BondData.xls');
    
    % Convert to categorical the Type and Rating
    B_Data.Type=categorical(B_Data.Type);
    B_Data.Rating=categorical(B_Data.Rating);
    
    % Remove any NaN, Inf, categorical undefined and 3 std outliers from
    % the raw data.
    B_Data=cleanTable(B_Data,{'NaN','Inf','Undefined','Outlier'});
    
    % Extract the corporate Bonds and remove Not Rated category from
    % ratings
    CB_Data=B_Data(B_Data.Type=='Corp'&B_Data.Rating~='Not Rated',:);
    
    % Keep only positive YTM
    CB_Data_YTM_Pos=CB_Data(CB_Data.YTM>0,:);
    
    % Explore mean and max values for YTM using a boxplot
    FiguresExe.f1=findobj('type','figure','Name','Corporate Bonds Boxplots');
    if ishghandle(FiguresExe.f1)
        close(FiguresExe.f1)
    end
    FiguresExe.f1=figure('Name','Corporate Bonds Boxplots');
    subplot(2,2,1)
    boxplot(CB_Data_YTM_Pos{:,'YTM'},CB_Data_YTM_Pos.Rating,'plotstyle',...
        'compact','colors',[.3765 .3765 .3765])
    ylabel('YTM')
    subplot(2,2,2)
    boxplot(CB_Data_YTM_Pos{:,'Coupon'},CB_Data_YTM_Pos.Rating,'plotstyle',...
        'compact','colors',[.3765 .3765 .3765])
    ylabel('Coupon Rate')
    subplot(2,2,3)
    boxplot(CB_Data_YTM_Pos{:,'Price'},CB_Data_YTM_Pos.Rating,'plotstyle',...
        'compact','colors',[.3765 .3765 .3765])
    ylabel('Bond Price')
    subplot(2,2,4)
    boxplot(CB_Data_YTM_Pos{:,'CurrentYield'},CB_Data_YTM_Pos.Rating,'plotstyle',...
        'compact','colors',[.3765 .3765 .3765])
    ylabel('Current Yield')
    
    
    
    % Group Scatter the Coupon Rate and YTM based on the Rating. For this
    % purpose we will create a transition between 2 colours to plot the
    % data.
    n_Cat=length(categories(CB_Data_YTM_Pos.Rating));
    linVec=linspace(0,1,n_Cat);
    ColMat=[linVec' linVec(end:-1:1)']*[1 128/255 0;.3765 .3765 .3765];
    FiguresExe.f2=findobj('type','figure','Name','Corporate Bonds Group Scatter');
    if ishghandle(FiguresExe.f2)
        close(FiguresExe.f2)
    end
    FiguresExe.f2=figure('Name','Corporate Bonds Group Scatter');
    gscatter(CB_Data_YTM_Pos.Coupon,CB_Data_YTM_Pos.YTM,CB_Data_YTM_Pos.Rating,ColMat)
    grid on;
    xlabel('Coupon Rate')
    ylabel('Yield To Maturity')
    
    
    %% Section 1-b
    % We will import the data in a different format as available and use
    % part of the numeric features to find a number of optimal clusters.
    
    % Load the data from corporate Bonds only.
    load corpBondData
    % Specify the numeric features used for the clustering
    numFeat={'Coupon','YTM','CurrentYield','RatingNum'};
    
    % Extract subset of numeric features we will use for clustering
    CB_NumData=corp{:,numFeat};
    
    % Specify a distance metric
    dist_metric='cosine';
    
    % Define a kmeans template function to be used in the call to
    % evalclusters and include multiple replicates to avoid local minima
    my_kmeans= @(X,K)(kmeans(X, K, 'emptyaction','singleton',...
        'Distance',dist_metric,'replicate',15));
    % Generate the evalclusters object using the template clustering
    % function
    ClusEval_Kmeans_CB=evalclusters(CB_NumData,my_kmeans,'silhouette','KList',...
        (2:6));
    % Cluster the data using the best K value obtained above. Notice how if
    % we use the metric cosine the cluster centre points we get represent
    % the mean of the points in that cluster, after normalizing those
    % points to unit Euclidean length so we need to undo such normalisation
    % if we intend to plot centroids together with data points.
    [KmeansGrp_CB,Cen_CB,sumd_CB]=kmeans(CB_NumData,...
        ClusEval_Kmeans_CB.OptimalK,'Start','cluster','replicate',15,...
        'Display','final','Distance',dist_metric);
    
    % Plot Silhouette metric for the clustering performed. Notice how we
    % take individual silhouette values as an output to show their mean as
    % part of the title of the figure.
    FiguresExe.f3=findobj('type','figure','Name','Silhouette Plot Corporate Bonds');
    if ishghandle(FiguresExe.f3)
        close(FiguresExe.f3)
    end
    FiguresExe.f3=figure('Name','Silhouette Plot Corporate Bonds');
    [sil_values,~] = silhouette(CB_NumData,KmeansGrp_CB,dist_metric);
    title(['Mean silhouette value for Corp. Bonds: ',num2str(mean(sil_values))])
    ylabel('Optimal Clusters based on Silhouette Value')
    
    % Plot the groups in a 3D scatter by choosing exclusively the three
    % first features in the data.
    FiguresExe.f4=findobj('type','figure','Name','Scatter Group Plot Corporate Bonds');
    if ishghandle(FiguresExe.f4)
        close(FiguresExe.f4)
    end
    FiguresExe.f4=figure('Name','Scatter Group Plot Corporate Bonds');
    scatter3(CB_NumData(:,1),CB_NumData(:,2),CB_NumData(:,3),[],KmeansGrp_CB,'o')
    
    title(['Corp. Bonds Data grouped by Best K = ',...
        num2str(ClusEval_Kmeans_CB.OptimalK)])
    xlabel(numFeat{1});ylabel(numFeat{2});zlabel(numFeat{3});
    
    
    disp(' ')
    disp('Finished Chapter A ')
    
    user_option=input('Would you like to close the figures from this chapter? Y/N [N]:','s');
    if isempty(user_option)
        user_option='n';
    end
    if strcmpi('y',user_option)
        
        for vv=1:4
            figname=strcat('f',num2str(vv));
            if ishghandle(FiguresExe.(figname))&&strcmp(get(FiguresExe.(figname),'BeingDeleted'),'off')
                close(FiguresExe.(figname));
            end
        end
    end
    
    
    disp(' ')
    disp(['--------------------------------------------------------------------------------------------';...
        '------------------------------------     END     -------------------------------------------';...
        '--------------------------------------------------------------------------------------------'])
    
    fprintf('\nProgram paused. Press enter to continue.\n\n');
    pause;
    
    
end % CBswitch


if ADswitch % {CHAPTER B}
    
    disp(' ')
    disp(['--------------------------------------------------------------------------------------------';...
        '------------------------------------  CHAPTER B  -------------------------------------------';...
        '--------------------------------------------------------------------------------------------'])
    
    %% Section 2-a
    % In this section we will import the data from different physical
    % activities as measure using the internal sensors of a smartphone.
    
    % Read data and convert activity into categorical
    A_Data=readtable('ActivityData.txt');
    A_Data.Activity=categorical(A_Data.Activity);
    
    % Check the mean and std value of the gravitational Z acceleration for
    % the different activities. We expect higher data separation in the Z
    % axis. After obtaining the data show it to the user.
    [mean_Ag_z,std_Ag_z]=grpstats(A_Data.GravAccMeanZ,A_Data.Activity,{@mean,@std});
    disp('Mean and std values for average gravitational acceleration on Z axis')
    disp(categories(A_Data.Activity))
    
    disp(mean_Ag_z)
    disp(std_Ag_z)
    
    % Plot the different activities in a 2D map with Z axis energy and Z
    % axis gravitational mean acceleration as the coordinates of the map.
    FiguresExe.f5=findobj('type','figure','Name','Activity Data on Z axis by Group');
    if ishghandle(FiguresExe.f5)
        close(FiguresExe.f5)
    end
    FiguresExe.f5=figure('Name','Activity Data on Z axis by Group');
    
    gscatter(A_Data.AccEnergyZ,A_Data.GravAccEnergyZ,A_Data.Activity)
    grid on;
    xlabel('AccEnergyZ')
    ylabel('GravAccEnergyZ')
    
    %% Section 2-b
    % We will import the complete data, remove the descriptive features and
    % the true class and extract the number of optimal clusters based on
    % the silhouette metric to understand if it is close to the number of
    % physical activities recorded.
    
    % Read the data and extract the numeric features.
    load HAData
    
    A_NumData=data{:,3:end};
    
    % We will change the distance metric in this occasion to cityblock
    dist_metric='cityblock';
    
    % Define a kmeans template function to be used in the call to
    % evalclusters and include multiple replicates to avoid local minima
    my_kmeans= @(X,K)(kmeans(X, K, 'emptyaction','singleton',...
        'Distance',dist_metric,'replicate',15));
    % Generate the evalclusters object and limit number of clusters to be
    % considered to the actual number of activities
    ClusEval_Kmeans_A=evalclusters(A_NumData,my_kmeans,'silhouette','KList',...
        (2:length(categories(A_Data.Activity))));
    
    % Plot the average Silhouette value for each of the number of clusters
    % considered
    FiguresExe.f6=findobj('type','figure','Name','Activity Data Silhouette Metric - Kmeans SqEuclidean');
    if ishghandle(FiguresExe.f6)
        close(FiguresExe.f6)
    end
    FiguresExe.f6=figure('Name','Activity Data Silhouette Metric - Kmeans SqEuclidean');
    plot(ClusEval_Kmeans_A)
    
    % Now define a new Kmeans problem where the algorithm is forced to
    % produce the same number of clusters as different activities are
    % recorded in the data
    [KmeansGrp_A,Cen_A,sumd_A]=kmeans(A_NumData,...
        length(categories(A_Data.Activity)),'Start','cluster','replicate',15,...
        'Display','final','Distance',dist_metric);
    
    % Cross tabulate the results against the true label of the data
    cross_Act=crosstab(KmeansGrp_A,A_Data.Activity);
    
    % Plot the cross tabulation using a 3D bar plot.
    FiguresExe.f7=findobj('type','figure','Name','Activity Data Clusters vs True Groups - Kmeans SqEuclidean');
    if ishghandle(FiguresExe.f7)
        close(FiguresExe.f7)
    end
    FiguresExe.f7=figure('Name','Activity Data Clusters vs True Groups - Kmeans SqEuclidean');
    
    bar3(cross_Act)
    ax = gca;
    ax.XTick = 1:length(categories(A_Data.Activity));
    ax.XTickLabel = categories(A_Data.Activity);
    
    ylabel('Kmeans Clusters')
    zlabel('Number of belonging Data Samples')
    
    %% Section 2-c
    % Now that we have clustered the data and understood the effect of
    % using a distance metric on the data we will train a classification
    % tree on the data and use 30% hold-out cross validation.
    
    % Create the partition on the activity data
    AD_pt=cvpartition(data.Activity,'holdout',0.3);
    
    AD_train=A_NumData(AD_pt.training,:);
    TrainLabel_AD=data.Activity(AD_pt.training);
    AD_test=A_NumData(AD_pt.test,:);
    TestLabel_AD=data.Activity(AD_pt.test);
    
    % Train the tree and set minimum number of leaf observations to three
    % (default enables single observation)
    CTmodel_A=fitctree(AD_train,TrainLabel_AD,'MinLeafSize',3);
    view(CTmodel_A,'Mode','text')
    
    % Extract prediction and use this to assess train and test errors
    AD_Pred=CTmodel_A.predict(AD_test);
    TrainErr_AD=100*resubLoss(CTmodel_A);
    TestErr_AD=100*CTmodel_A.loss(AD_test,TestLabel_AD);
    
    disp(['Physical Activity Data Classification Error using Classification Tree',...
        ' in Training set - ',num2str(TrainErr_AD,'%.4g'), '% - and in Validation set - ',...
        num2str(TestErr_AD,'%.4g'),'% -']);
    
    % Extract the confusion matrix for the validation set and plot the
    % latter using an illustrated image map
    [conf_mtrx_AD,conf_mtrx_AD_label] = confusionmat(TestLabel_AD,AD_Pred);
    
    FiguresExe.f8=findobj('type','figure','Name','Activity Data Class. Tree Confusion Matrix');
    if ishghandle(FiguresExe.f8)
        close(FiguresExe.f8)
    end
    FiguresExe.f8=figure('Name','Activity Data Class. Tree Confusion Matrix');
    imagesc(conf_mtrx_AD)
    % Display colour bar legend
    colorbar
    % Change colour map
    colormap parula
    
    ylabel('True Class')
    xlabel('Predicted Class')
    ax=gca;
    ax.XTick=1:length(categories(A_Data.Activity)); ax.XTickLabel=categories(conf_mtrx_AD_label);
    ax.YTick=1:length(categories(A_Data.Activity)); ax.YTickLabel=categories(conf_mtrx_AD_label);
    
    % Add labels at the center of the image
    [xpos,ypos] = meshgrid(1:size(conf_mtrx_AD,1),1:size(conf_mtrx_AD,2));
    text(xpos(:),ypos(:),num2str(conf_mtrx_AD(:),'%d'),'HorizontalAlignment','Center')
    
    
    disp(' ')
    disp('Finished Chapter B')
    
    user_option=input('Would you like to close the figures from this chapter? Y/N [N]:','s');
    if isempty(user_option)
        user_option='n';
    end
    if strcmpi('y',user_option)
        
        for vv=5:8
            figname=strcat('f',num2str(vv));
            if ishghandle(FiguresExe.(figname))&&strcmp(get(FiguresExe.(figname),'BeingDeleted'),'off')
                close(FiguresExe.(figname));
            end
        end
    end
    
    disp(' ')
    disp(['--------------------------------------------------------------------------------------------';...
        '------------------------------------     END     -------------------------------------------';...
        '--------------------------------------------------------------------------------------------'])
    fprintf('\nProgram paused. Press enter to continue.\n\n');
    pause;
    
    
end % ADswitch



if SIswitch % {CHAPTER C}
    
    disp(' ')
    disp(['--------------------------------------------------------------------------------------------';...
        '------------------------------------  CHAPTER C  -------------------------------------------';...
        '--------------------------------------------------------------------------------------------'])
    
    %% Section 3-a
    % In this section we will load the data from different terrain
    % satellite images and use it to train an SVM classifier by means of
    % the error correcting output codes approach.
    
    % Import the data and standardise the values of each feature
    load satdata
    
    SI_Data=zscore(satData);
    
    % Train SVM classifier on all the data using 10 folds cross validation
    % as accuracy assessment. Notice how as we have more than 2 classes we
    % need to use the ECOC approach and we have fixed the Kernel to be
    % polynomial of order 2 and box constraint equal to 0.1
    
    % Define number of CV folds and box constraint
    numFolds=10;
    BCons=0.1;
        
    % We will create an SVM template using the desired settings. Notice
    % this is done inside the for loop as we need to update the box
    % constraint value
    
    Template_Learner=templateSVM('KernelFunction','polynomial',...
        'PolynomialOrder',2,'BoxConstraint',BCons,'Prior','empirical',...
        'Solver','SMO');
    
    if Parallelswitch
        
        % Start the parallel pool to ensure more predictable results
        delete(gcp('nocreate'));
        poolobj=parpool('local',n_cores);
        
        % Set options to use parallelisation
        ecoc_options = statset('UseParallel',1);      
        
        % Train always 1vs1 models.   
        ECOC_1vs1_model_SI=fitcecoc(SI_Data,satClass,'Learners',...
            Template_Learner,'Coding','onevsone','KFold',numFolds,...
            'Options',ecoc_options);
        
        SI_Pred=kfoldPredict(ECOC_1vs1_model_SI,'Options',ecoc_options);
           
        % Close parallel pool
        delete(gcp('nocreate'));       
        
    else
        % Train models without using parallelisation
        ECOC_1vs1_model_SI=fitcecoc(SI_Data,satClass,'Learners',...
            Template_Learner,'KFold',numFolds,'Coding','onevsone');
        
        SI_Pred=kfoldPredict(ECOC_1vs1_model_SI);
        
    end 
    
    disp(['SVM ECOC 1vs1',num2str(numFolds),'-Fold loss using BCons=',num2str(BCons),...
        ' equals ',num2str(100* ECOC_1vs1_model_SI.kfoldLoss,'%.4g'),'%'])
    
    % Extract the confusion matrix for all data after K-fold prediction and
    % plot the latter using an illustrated image map
    [conf_mtrx_SI,conf_mtrx_SI_label] = confusionmat(satClass,SI_Pred);
    
    FiguresExe.f9=findobj('type','figure','Name','Satellite Terrain Image SVM ECOC Confusion Matrix');
    if ishghandle(FiguresExe.f9)
        close(FiguresExe.f9)
    end
    FiguresExe.f9=figure('Name','Satellite Terrain Image SVM ECOC Confusion Matrix');
    imagesc(conf_mtrx_SI)
    % Display colour bar legend
    colorbar
    % Change colour map
    colormap autumn
    
    ylabel('True Class')
    xlabel('Predicted Class')
    ax=gca;
    ax.XTick=1:length(categories(satClass)); ax.XTickLabel=categories(conf_mtrx_SI_label);
    ax.YTick=1:length(categories(satClass)); ax.YTickLabel=categories(conf_mtrx_SI_label);
    
    % Add labels at the center of the image
    [xpos,ypos] = meshgrid(1:size(conf_mtrx_SI,1),1:size(conf_mtrx_SI,2));
    text(xpos(:),ypos(:),num2str(conf_mtrx_SI(:),'%d'),'HorizontalAlignment','Center')
       
    disp(' ')
    disp('Finished Chapter C')
    
    user_option=input('Would you like to close the figures from this chapter? Y/N [N]:','s');
    if isempty(user_option)
        user_option='n';
    end
    if strcmpi('y',user_option)
        
        for vv=9:9
            figname=strcat('f',num2str(vv));
            if ishghandle(FiguresExe.(figname))&&strcmp(get(FiguresExe.(figname),'BeingDeleted'),'off')
                close(FiguresExe.(figname));
            end
        end
    end
    
    disp(' ')
    disp(['--------------------------------------------------------------------------------------------';...
        '------------------------------------     END     -------------------------------------------';...
        '--------------------------------------------------------------------------------------------'])
    fprintf('\nProgram paused. Press enter to continue.\n\n');
    pause;
    
    
end % SIswitch

if BMswitch % {CHAPTER D}
    
    disp(' ')
    disp(['--------------------------------------------------------------------------------------------';...
        '------------------------------------  CHAPTER D  -------------------------------------------';...
        '--------------------------------------------------------------------------------------------'])
    
    %% Section 4-a
    % In this section we will look at data from a bank attempting to
    % predict the profile of a customer that is likely to purchase a
    % product.
    % Many of the features are categorical and describe the social status
    % of the costumers. Hence we will first convert these from categorical
    % to numeric features and then train both a discriminant and a naive
    % bayes model in an attempt to predict the decision of the customer.
    
    % First we will read the data and extract numeric features and target
    % response.
    SourceData=load('bankData.mat');
    
    bank=SourceData.bank;
    
    NumBankData=varfun(@double,bank(:,1:end-1));
    BM_Data=NumBankData{:,:};
    Cust_Resp=categorical(bank.y);
    
    % We will do a simple 2D plot to visualise the points.
    
    FiguresExe.f10=findobj('type','figure','Name','Bank Marketing Customer Response');
    if ishghandle(FiguresExe.f10)
        close(FiguresExe.f10)
    end
    FiguresExe.f10=figure('Name','Bank Marketing Customer Response');
    gscatter(bank.balance,bank.duration,bank.y)
    grid on
    xlabel('Bank Account Balance')
    ylabel('Last Call Duration')
    title('Customer Decision against Balance and Time on the Phone')
    
    % Examine response data
    disp('Counts of Yes & No responses from Customers')
    tabulate(Cust_Resp)
    
    FiguresExe.f11=findobj('type','figure','Name','Customer Responses');
    if ishghandle(FiguresExe.f11)
        close(FiguresExe.f11)
    end
    FiguresExe.f11=figure('Name','Customer Responses');
    hist(Cust_Resp)
    ylabel('Number Customers'); title('Marketing Campaign Result');
    
    % Create a CV partition of the data 30/70
    BM_part = cvpartition(height(bank),'holdout',0.30);
    
    
    BM_train = BM_Data(BM_part.training,:);
    Resp_train = Cust_Resp(BM_part.training,:);
    BM_test = BM_Data(BM_part.test,:);
    Resp_test = Cust_Resp(BM_part.test,:);
    
    % Train the discriminant model and the naive bayes models. In the first
    % approach for NB modelling we will assume normal distributions for all
    % independent features with respect to the response but for a second
    % approach we will specify which features are categorical and hence can
    % be modelled as multivariate multinomial.
    
    GD_model_BM = fitcdiscr(BM_train,Resp_train,'discrimType','quadratic','Prior',...
        'empirical');
    
    NBnormal_model_BM=fitcnb(BM_train,Resp_train,'DistributionNames',...
        'normal','Prior','empirical');
    
    % Specify the categorical distributions as mvmn. The rest will be
    % treated as normal.
    mvpredictors={'normal','mvmn','mvmn','mvmn','mvmn','normal','mvmn','mvmn',...
        'mvmn','normal','mvmn','normal','normal','normal','normal','mvmn'};
    
    NBmvmn_model_BM=fitcnb(BM_train,Resp_train,'DistributionNames',...
        mvpredictors,'Prior','empirical','CategoricalPredictors',[2 3 4 5 7 8 9 11 16]);
    
    % Extract and display the classification error on the validation set.
    GD_CustPred = predict(GD_model_BM,BM_test);
    GD_Class_Err = 100*nnz(GD_CustPred ~= Resp_test)/length(Resp_test);
    
    NBnormal_CustPred = predict(NBnormal_model_BM,BM_test);
    NBnormal_Class_Err = 100*nnz(NBnormal_CustPred ~= Resp_test)/length(Resp_test);
    
    NBmvmn_CustPred = predict(NBmvmn_model_BM,BM_test);
    NBmvmn_Class_Err = 100*nnz(NBmvmn_CustPred ~= Resp_test)/length(Resp_test);
    
    disp(['Misclassification error over test set for Gaussian Discriminant Quadratic - ',...
        num2str(GD_Class_Err,'%.4g'),' % - for Gaussian Naive Bayes - ',...
        num2str(NBnormal_Class_Err,'%.4g'),' % - and for Multinomial Naive Bayes - ',...
        num2str(NBmvmn_Class_Err,'%.4g'),' % -']);
    
    
    disp(' ')
    disp('Finished Chapter D')
    
    user_option=input('Would you like to close the figures from this chapter? Y/N [N]:','s');
    if isempty(user_option)
        user_option='n';
    end
    if strcmpi('y',user_option)
        
        for vv=10:11
            figname=strcat('f',num2str(vv));
            if ishghandle(FiguresExe.(figname))&&strcmp(get(FiguresExe.(figname),'BeingDeleted'),'off')
                close(FiguresExe.(figname));
            end
        end
    end
    
    disp(' ')
    disp(['--------------------------------------------------------------------------------------------';...
        '------------------------------------     END     -------------------------------------------';...
        '--------------------------------------------------------------------------------------------'])
    fprintf('\nProgram paused. Press enter to continue.\n\n');
    pause;
    
    
end % BMswitch

if CRswitch % {CHAPTER E}
    
    disp(' ')
    disp(['--------------------------------------------------------------------------------------------';...
        '------------------------------------  CHAPTER E  -------------------------------------------';...
        '--------------------------------------------------------------------------------------------'])
    
    %% Section 5-a
    % The last excercise will focus on attempting to predict credit rating
    % based on different finaltial features.
    
    % We will first load the data, extract three significant numeric
    % features and the response (credit rating in this case)
    load CreditData
    
    CR_Data=[CreditRatings.RE_TA,CreditRatings.MVE_BVTD,CreditRatings.Industry];
    Ratings = categorical(CreditRatings.Rating);
    
    % Create the partition
    CR_part=cvpartition(Ratings,'holdout',0.3);
    CR_train=CR_Data(CR_part.training,:);
    TrainRating=Ratings(CR_part.training);
    CR_test=CR_Data(CR_part.test,:);
    TestRating=Ratings(CR_part.test);
    
    
    % Train a classification tree
    CTmodel_CR=fitctree(CR_train,TrainRating);
    view(CTmodel_CR,'Mode','text')
    
    % Extract prediction and use this to assess train and test errors
    CR_Pred=CTmodel_CR.predict(CR_test);
    TrainErr_CR=100*resubLoss(CTmodel_CR);
    TestErr_CR=100*CTmodel_CR.loss(CR_test,TestRating);
    
    disp(['Credit Ratings 30% Hold-out Classification Error using Classification Tree',...
        ' in Training set - ',num2str(TrainErr_CR,'%.4g'), '% - and in Validation set - ',...
        num2str(TestErr_CR,'%.4g'),'% -']);
    
    % Extract the confusion matrix for the validation set and plot the
    % latter using an illustrated image map
    [conf_mtrx_CR,conf_mtrx_CR_label] = confusionmat(TestRating,CR_Pred);
    
    FiguresExe.f12=findobj('type','figure','Name','Credit Ratings Decision Tree Confusion Matrix');
    if ishghandle(FiguresExe.f12)
        close(FiguresExe.f12)
    end
    FiguresExe.f12=figure('Name','Credit Ratings Decision Tree Confusion Matrix');
    imagesc(conf_mtrx_CR)
    % Display colour bar legend
    colorbar
    % Change colour map
    colormap spring
    
    ylabel('True Class')
    xlabel('Predicted Class')
    title('Credit Ratings - Single Decision Tree Conf Matrix')
    ax=gca;
    ax.XTick=1:length(categories(Ratings)); ax.XTickLabel=categories(conf_mtrx_CR_label);
    ax.YTick=1:length(categories(Ratings)); ax.YTickLabel=categories(conf_mtrx_CR_label);
    
    % Add labels at the center of the image
    [xpos,ypos] = meshgrid(1:size(conf_mtrx_CR,1),1:size(conf_mtrx_CR,2));
    text(xpos(:),ypos(:),num2str(conf_mtrx_CR(:),'%d'),'HorizontalAlignment','Center')
    
    %% Section 5-b
    % We will now fit an ensemble learner to improve the accuracy of
    % classification. We will use a bag of trees with the same CV partition
    % as above. We will call the model ClassificationForest_model_CR and
    % add 100 trees on it.
    
    % Use parallelisation if available
    if Parallelswitch
        
        % Set options to use parallelisation
        Bagger_options = statset('UseParallel',1);
        % Start the parallel pool to ensure more predictable results
        delete(gcp('nocreate'));
        poolobj=parpool('local',n_cores);
        
        % Train the bag of trees classifier and enable the out of the bag
        % error in order to plot it afterwards.
        CF_model_CR=TreeBagger(100,CR_train,TrainRating,'OOBPred',...
            'on','Options',Bagger_options);
        
        % Close parallel pool
        delete(gcp('nocreate'));
        
    else
        % Train the bag of trees classifier and enable the out of the bag
        % error in order to plot it afterwards.
        CF_model_CR=TreeBagger(100,CR_train,TrainRating,'OOBPred','on');
    end
    % Predict using the bag model. Note how the method predict for a bagged
    % tree learner produces a cell array and hence we need to convert it to
    % a categorical array.
    
    CF_Pred=categorical(predict(CF_model_CR,CR_test));
    
    % Extract and display the error from this learner. We cannot use the
    % method loss for a TreeBagger object
    CF_validErr=100*nnz(CF_Pred ~= TestRating)/length(TestRating);
    
    disp(['Hold-out Cross Validation Error for Forest of 100 Trees in Credit Ratings Equals ',...
        num2str(CF_validErr,'%.4g'),'%']);
    
    % And compute the confusion matrix for this ensemble classifier
    [conf_mtrx_CF,conf_mtrx_CF_label]=confusionmat(TestRating,CF_Pred);
    
    % Plot the confusion matrix as a heat map image
    FiguresExe.f13=findobj('type','figure','Name','Credit Ratings Decision Tree Forest Confusion Matrix');
    if ishghandle(FiguresExe.f13)
        close(FiguresExe.f13)
    end
    FiguresExe.f13=figure('Name','Credit Ratings Decision Tree Forest Confusion Matrix');
    imagesc(conf_mtrx_CF)
    % Display colour bar legend
    colorbar
    % Change colour map
    colormap spring
    
    ylabel('True Class')
    xlabel('Predicted Class')
    title('Credit Ratings - Bag of Trees Conf Matrix')
    ax=gca;
    ax.XTick=1:length(categories(Ratings)); ax.XTickLabel=categories(conf_mtrx_CF_label);
    ax.YTick=1:length(categories(Ratings)); ax.YTickLabel=categories(conf_mtrx_CF_label);
    
    % Add labels at the center of the image
    [xpos,ypos] = meshgrid(1:size(conf_mtrx_CF,1),1:size(conf_mtrx_CF,2));
    text(xpos(:),ypos(:),num2str(conf_mtrx_CF(:),'%d'),'HorizontalAlignment','Center')
    
    % Display the out of the bag error for the bag of 100 trees
    FiguresExe.f14=findobj('type','figure','Name','Credit Ratings Out-of-bag Error for Forest of Decision Trees');
    if ishghandle(FiguresExe.f14)
        close(FiguresExe.f14)
    end
    FiguresExe.f14=figure('Name','Credit Ratings Out-of-bag Error for Forest of Decision Trees');
    plot(100*oobError(CF_model_CR))
    grid on;
    xlabel('Number of Trees in Bag (Forest Size)')
    ylabel('Out-of-bag Error [%]')
    
    %% Section 5-c
    % We will now add gaussian noise from 0 to 5% std to all the colums of
    % the TEST data only in order to study the sensitivity of both a single
    % decision tree versus a forest of multiple elements to data
    % variations.
    
    % Generate the noise and add it to the TEST set
    noise_to_add=bsxfun(@times,randn(size(CR_test)),0.05*std(CR_test));
    CR_test_noise=CR_test+noise_to_add;
    
    % Predict the label using the single tree as trained beforehand.
    CR_Pred_noise=CTmodel_CR.predict(CR_test_noise);
    % Extract the test error on the noisy test set.
    TestErr_noise_CR=100*CTmodel_CR.loss(CR_test_noise,TestRating);
    
    % Predict label and extract test error on the noisy data for the forest
    % of 100 trees.
    CF_Pred_noise=categorical(predict(CF_model_CR,CR_test_noise));
    CF_validErr_noise=100*nnz(CF_Pred_noise ~= TestRating)/length(TestRating);
    
    % Show results to user.0
    disp(['Classification Error on Test Set after up to 5% STD Noise Addition',...
        ' In Single tree - ',num2str(TestErr_noise_CR,'%.4g'),'% - and in  Forest of Trees - ',...
        num2str(CF_validErr_noise,'%.4g'),'% -']);
    
    % Compute the confusion matrix for the noisy test set on the single
    % tree learner and plot the result as a heat map image.
    [conf_mtrx_CR_noise,conf_mtrx_CR_noise_label] = confusionmat(TestRating,CR_Pred_noise);
    
    FiguresExe.f15=findobj('type','figure','Name','Credit Ratings Decision Tree Confusion Matrix with up to 5% std Noise');
    if ishghandle(FiguresExe.f15)
        close(FiguresExe.f15)
    end
    FiguresExe.f15=figure('Name','Credit Ratings Decision Tree Confusion Matrix with up to 5% std Noise');
    imagesc(conf_mtrx_CR_noise)
    % Display colour bar legend
    colorbar
    % Change colour map
    colormap spring
    
    ylabel('True Class')
    xlabel('Predicted Class')
    title('Credit Ratings - Single Decision Tree Conf Matrix -  Noisy Test Set')
    ax=gca;
    ax.XTick=1:length(categories(Ratings)); ax.XTickLabel=categories(conf_mtrx_CR_noise_label);
    ax.YTick=1:length(categories(Ratings)); ax.YTickLabel=categories(conf_mtrx_CR_noise_label);
    
    % Add labels at the center of the image
    [xpos,ypos] = meshgrid(1:size(conf_mtrx_CR_noise,1),1:size(conf_mtrx_CR_noise,2));
    text(xpos(:),ypos(:),num2str(conf_mtrx_CR_noise(:),'%d'),'HorizontalAlignment','Center')
    
    % Compute the confusion matrix for the noisy test set on the forest of
    % tree learner and plot the result as a heat map image.
    [conf_mtrx_CF_noise,conf_mtrx_CF_noise_label]=confusionmat(TestRating,CF_Pred_noise);
    
    FiguresExe.f16=findobj('type','figure','Name','Credit Ratings Decision Tree Forest Confusion Matrix with up to 5% std Noise');
    if ishghandle(FiguresExe.f16)
        close(FiguresExe.f16)
    end
    FiguresExe.f16=figure('Name','Credit Ratings Decision Tree Forest Confusion Matrix with up to 5% std Noise');
    imagesc(conf_mtrx_CF_noise)
    % Display colour bar legend
    colorbar
    % Change colour map
    colormap spring
    
    ylabel('True Class')
    xlabel('Predicted Class')
    title('Credit Ratings - Bag of Trees Conf Matrix - Noisy Test Set')
    ax=gca;
    ax.XTick=1:length(categories(Ratings)); ax.XTickLabel=categories(conf_mtrx_CF_noise_label);
    ax.YTick=1:length(categories(Ratings)); ax.YTickLabel=categories(conf_mtrx_CF_noise_label);
    
    % Add labels at the center of the image
    [xpos,ypos] = meshgrid(1:size(conf_mtrx_CF_noise,1),1:size(conf_mtrx_CF_noise,2));
    text(xpos(:),ypos(:),num2str(conf_mtrx_CF_noise(:),'%d'),'HorizontalAlignment','Center')
    
    disp(' ')
    disp('Finished Chapter E')
    
    user_option=input('Would you like to close the figures from this chapter? Y/N [N]:','s');
    if isempty(user_option)
        user_option='n';
    end
    if strcmpi('y',user_option)
        
        for vv=12:16
            figname=strcat('f',num2str(vv));
            if ishghandle(FiguresExe.(figname))&&strcmp(get(FiguresExe.(figname),'BeingDeleted'),'off')
                close(FiguresExe.(figname));
            end
        end
    end
    
    disp(' ')
    disp(['--------------------------------------------------------------------------------------------';...
        '------------------------------------     END     -------------------------------------------';...
        '--------------------------------------------------------------------------------------------'])
    fprintf('\nProgram paused. Press enter to continue.\n\n');
    pause;
    
    
end % CRswitch


%--------------------------------------------------------------------------
%--------------------------------------------------------------------------


































































