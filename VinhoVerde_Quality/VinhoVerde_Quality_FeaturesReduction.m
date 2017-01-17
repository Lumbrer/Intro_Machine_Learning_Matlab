%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% $Author: Lumbrer $    $Date: 2017/01/17 $    $Revision: 0.8$
% Copyright: Francisco Lumbreras
%
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% File Name: VinhoVerde_Quality_FeaturesReduction.m
% Description: Script to perform dimensionality reduction and feature
% selection on the data from Cortez et al. 2009 'Modeling wine
% preferences by data mining from physicochemical properties'.
%
% This file has been coceived as a script but can be easily converted into
% a parameterised GUI to select data and activate different types of
% algorithms on Matlab. In the first section of the script a number of
% switches are defined to activate or deactivate the different chapters of
% the script. These can all run independently from each other.
%
% In this script, we will analyse some of the built in functions that
% Matlab provides to perform features pre-processing before training a ML
% algorithm. We will cover methods for both feature dimensionality
% reduction and feature selection. The chapters here defined are:
%
% A) The first chapter covers Principal Components Analysis and how it can
% be used to transform the dimensionality of a feature set whilst
% maintaining variance information from the original data. This section
% covers different methods to extract Principal Components and the feature
% scores to be used in latter training of an algorithm as well as methods
% to intuitively visualise the Principal Components.
%
% B) The second chapter introduces Factor Analysis and the circumstances
% under which one might consider such algorithm to perform dimensionality
% reduction. In addition, the extraction of feature scores and the division
% of variance into individual, common and error is discussed. Methods to
% intuitively visualise factor loadings are included.
%
% C) This chapter covers the use of reduced scores from both Principal
% Components and Common Factors to train a classification model and compare
% the performance and execution time with the use of the original features
% by means of a practical example.
%
% D) The last chapter covers Feature Selection methods and in particular
% the built-in functions available in Matlab to perform wrapper selection.
% Given a practical example with fixed hyperparameters, the performance
% imporvement after sequential forwards and backwards elimination is
% studied. Furthermore, a brief example on randomised single search is
% provided to explain the limitations of sequential approaches.
%
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% ---------------- CODE HERE ----------------------------------------------
clc;
close all force;
clear all;
set(0,'DefaultTextInterpreter','none')
%Add path to files required
addpath(genpath(pwd));
% Create a struct to store figures
FiguresFeat=struct;


%% Define switches to activate each of the chapters of the script
% Switches for chapters A, B, C & D
PCAswitch=boolean(1);        % A - Activate PCA analysis
FAswitch=boolean(1);         % B - Activate Factor Analysis
DRswitch=boolean(1);         % C - Activate Supervised Learning using
% Dimensionality Reduction. Requires either
% PCAswitch or FAswith to be selected.
FSswitch=boolean(1);         % D - Activate Feature Selection


% Switch to identify if multiple parallel workers are available locally
Parallelswitch=true;
% Define the number of cores in the parallel pool available locally
n_cores=4;
% Set random seed for reproducibility
rng(1234)

%% Section 1
% Import data and add label. Split white & red wine data for
% classification problems on the quality label

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

% Extract a clean version of the labels
WineClasses=DataClean.QCLabel;

% Split white wine data & labels
idwhite=wineinfo=='White';
WhiteData=DataClean(idwhite,1:end-1);
QCLabel_w=WineClasses(idwhite);

% Split red wine data & labels
RedData=DataClean(~idwhite,1:end-1);
QCLabel_r=WineClasses(~idwhite);

%% Section 2
% Extract the features in a numeric matrix. We will use standardisation to
% zero mean and 1 std in this script to be in line with simplifications
% during Princiapl Components Analysis and Factor Analysis.

Features=DataClean{:,1:end-1};

% Extract number of features and their names
n_features=size(Features,2);
Feature_Names=DataClean.Properties.VariableNames(1:end-1);

% Standardise White & Red wine data to zero mean and unitary std and
% extract nummeric feature arrays
WhiteData_Norm=normTable(WhiteData,{'Centre'});
Features_w=WhiteData{:,:};
Features_w_Norm=WhiteData_Norm{:,:};
RedData_Norm=normTable(RedData,{'Centre'});
Features_r=RedData{:,:};
Features_r_Norm=RedData_Norm{:,:};


if PCAswitch % {CHAPTER A}
    
    
    disp(' ')
    disp(['--------------------------------------------------------------------------------------------';...
        '------------------------------------  CHAPTER A  -------------------------------------------';...
        '--------------------------------------------------------------------------------------------'])
    %% Section 3-a
    % In order to reduce the computational load of training a model and the
    % energy consumption linked to a complex machine learning problem, it
    % is frequently desired to reduce the dimension of the input to any
    % learning algorithm. This reduction comes with a subsequent loss of
    % information but can lead to notorious time efficiency improvement and
    % remove irrelevant information that constitutes noise as well as
    % reducing memory requirements to store the data required.
    
    % In this first chapter we will introduce Principal Components Analysis
    % and its formulation in Matlab. PCA produces a linear transformation
    % of the features from an original 'n' dimensional space to a different
    % 'n' dimensional space whose 'n' basis are defined as the principal
    % components of the data. These form an orthogonal basis so that data
    % redundancy can be avoided.
    
    % The reduction occurs when from these 'n' principal components, only
    % 'd' are chosen (d<n) following the criterion of maximising the
    % variance retained as a new basis on which features can be projected.
    % The level of variance retained by a subset of principal components
    % can be extracted by looking at the cumulative sum of the eigenvalues
    % of 1/m(X'X) (X is the matrix of numeric features where each row
    % corresponds to a data sample, m represents the total number of
    % samples available) or alternatively the singular values associated
    % to X/sqrt(m). The eigenvectors produced by either the decompostion
    % of 1/m(X'X) or the V matrix after performing singular value
    % decomposition on X/sqrt(m) constitute the principal components linked
    % to maintaining the cumulative variance represented by their
    % eigenvalues. Note that for eigenvectors calculation, all the constant
    % terms linked to 'm' are not needed.
    
    % A common error in PCA analysis is to mistake principal component
    % directions and loadings. Directions are given by the eigenvectors of
    % the covariance matrix on the data while loadings (representing how
    % important each feature is for the composition of a particular
    % principal component) are extracted when the directions are scaled by
    % one divided by the square root of the eigenvalue of the covariance
    % matrix linked to such direction. 
    
    % For more details on variance retained optimisation see:
    % [http://cs229.stanford.edu/notes/cs229-notes10.pdf]
    
    % We will perform PCA on the white wine features. In order to
    % facilitate the formulation of PCA via the assumption of zero mean is
    % good practice to standardise the data before extracting the principal
    % components. Hence we will use the standardised white wine features
    % even though Matlab can handle nonstandardised data with the built in
    % function 'pca'
    
    % Perform PCA decomposition. The three outputs from the function 'pca'
    % from Matlab are as follows:
    
    % 1. A matrix containing the principal components in descending order
    % of component variance where each row corresponds to the contribution
    % or loading per feature and each column represents a principal
    % component.
    % 2. The scores or transformed samples using principal components
    % 3. The latency or variance in descending order.
    % 4. The Hotelling's T-squared statistic for each observation in the
    % matrix of features.
    % 5. The percentage of the total variance explained by each principal
    % component (non cumulative just as a percentaje of the total).
    % 6. The estimated mean of each feature.
    
    % PCA allows the user to be critical with the data used in order to
    % handle missing points and NaN instances, to weight the coefficients
    % as the inverse of the variance of the samples to avoid the influence
    % of outliers or to specify the algorithm used to obtain the components
    % (singular values decomposition or eigenvalues extraction for example).
    
    % Let us extract the principal components. We will request the use of
    % svd, as although it is slower for matrices with greater number of
    % observations than features (as here) it is more accurate due to the
    % condition number of X'X being the square of the condition number of X.
    % We will also deactivate matrix centering as we know the features have
    % been standardised already.
    
    [Princ_comp_w,Score_w,Latency_w,~,Explained_w]=pca(Features_w_Norm,...
        'Algorithm','svd','Centered',false);
    
    % Once we have extracted the principal components we can visualise how
    % many are needed to retain 90% of the variance (or alternative
    % threshold value). The function pareto will display only up to 95% of
    % the cumulative value of the data it is presented. Hence we pass the
    % variance as an argument.
    
    FiguresFeat.f1=findobj('type','figure','Name','PCA Variance Retained up to 95% - White Wine');
    if ishghandle(FiguresFeat.f1)
        close(FiguresFeat.f1)
    end
    FiguresFeat.f1=figure('Name','PCA Variance Retained up to 95% - White Wine');
    
    pareto(Latency_w)
    xlabel('Principal Component Number - Columns Princ_comp_w')
    ylabel('PCA Variance')
    
    % Label both y-axes if your version allows for such feature
    if strcmp(version('-release'),'2016b')
        yyaxis right
        ylabel('Variance Retained [%]')
    end
    
    % An alternative way of plotting the cumulative variance retained.
    FiguresFeat.f2=findobj('type','figure','Name','PCA Variance Retained - White Wine');
    if ishghandle(FiguresFeat.f2)
        close(FiguresFeat.f2)
    end
    FiguresFeat.f2=figure('Name','PCA Variance Retained - White Wine');
    
    bar(cumsum(Explained_w),'FaceColor',[.3765 .3765 .3765],'EdgeColor',...
        [1 128/255 0]);
    xlabel('Principal Component Number - Columns Princ_comp_w')
    ylabel('Cumulative Variance Retained [%]')
    
    
    %% Section 3-b
    % Now that we know we need the seven first principal components to
    % explain about 90% of the variance we will specify we only need to extract
    % these when calling the function pca and the feature scores projected
    % into this new four dimensional basis. Notice how as we had zero mean
    % accross features we can express feature scores as:
    
    % |--------------------------------------------------------|
    % | Features_Norm_PCA_w=Features_Norm_w*Princ_comp_90_w    |
    % |--------------------------------------------------------|
    
    [Princ_comp_90_w,Features_Norm_PCA_w,Latency_90_w]=pca(Features_w_Norm,...
        'Algorithm','svd','Centered',false,'NumComponents',7);
    
    % We have extracted the Directions, the Scores on the new basis but we
    % need the loadings. We need to normalise each direction by dividing by
    % the square root of the corresponding eigenvalue or latency.
    
    Princ_comp_Loadings_90_w=Princ_comp_90_w./...
        (repmat(sqrt(Latency_90_w(1:7)'),n_features,1));
    
    % We can now visualise the PCA result using multiple display options.
    % Being limited to 3D displays only helps if with fewer than three
    % components we can represent 90-95% or more of the variance on the data.
    % However, the important idea is that the principal components are
    % directions built by linear combinations of the original features and
    % hence we can understand how much each original feature contributes to
    % the generation of these maximum variance directions.
    
    % It is helpful to produce an abbreviated version of the feature names
    % in order to label plots for space reasons.
    
    Feature_Names_Abr={};
    for ii=1:length(Feature_Names)
        Name=Feature_Names{ii};
        % Take location of capital letters for each name
        Cap_Indexes=regexp(Name,'[A-Z]');
        if ~isempty(Cap_Indexes)
            % Take capital part of name as a candidate
            Candidate_Abr=Name(Cap_Indexes);
            % if there is already an abbreviated name that is coincident,
            % dispose of the candidate and use the full name of the feature
            if ~isempty(Feature_Names_Abr)&&any(strcmp(Feature_Names_Abr,Candidate_Abr))
                Feature_Names_Abr{ii}=Name;
            else  % take the candidate as a valid abbreviation
                Feature_Names_Abr{ii}=Candidate_Abr;
            end
        else % If not capital letters found take the complete original
            Feature_Names_Abr{ii}=Name;
        end
        
        % Thinking about feature names like 'pH', in case the first letter
        % in the name is lower case, keep it.
        if Name(1)>='a'&&Name(1)<='z'
            Feature_Names_Abr{ii}=[Name(1) Feature_Names_Abr{ii}];
        end
    end
    
    % Now we are ready for some compact plotting of the principal components.
    % Let us perform first 3D visualisation of the three first components by
    % means of biplot and imagesc
    
    FiguresFeat.f3=findobj('type','figure','Name','PCA Directions Pairwise Biplot - White Wine');
    if ishghandle(FiguresFeat.f3)
        close(FiguresFeat.f3)
    end
    FiguresFeat.f3=figure('Name','PCA Directions Pairwise Biplot - White Wine');
    % Take all pairwise combinations of the four components
    PCA_comb=nchoosek(1:size(Princ_comp_90_w,2),2);
    n_PCA_comb=size(PCA_comb,1);
    for ij=1:n_PCA_comb
        % Plot two per rown in a multiple plot
        subplot(ceil(n_PCA_comb/2),2,ij)
        biplot(Princ_comp_90_w(:,PCA_comb(ij,:)),'VarLabels',Feature_Names_Abr);
        xlabel(['P. Component ',num2str(PCA_comb(ij,1))]);
        ylabel(['P. Component ',num2str(PCA_comb(ij,2))]);
    end
    
    % A biplot can help us visualise the directions of the principal
    % components and how the data samples are distributed in the new space.
    
    FiguresFeat.f4=findobj('type','figure','Name','PCA Directions Biplot - White Wine');
    if ishghandle(FiguresFeat.f4)
        close(FiguresFeat.f4)
    end
    FiguresFeat.f4=figure('Name','PCA Directions Biplot - White Wine');
    % This time we provide the function biplot with the scores in order to
    % plot these as a cloud of points in the PCA basis.
    biplot(Princ_comp_90_w(:,1:3),'scores',Features_Norm_PCA_w(:,1:3),'VarLabels',Feature_Names,...
        'color',[.3765 .3765 .3765],'MarkerEdgeColor',[1 128/255 0]);
    xlabel('First Principal Component')
    ylabel('Second Principal Component')
    zlabel('Third Principal Component')
    
    % After looking at the principal directions, we can now turn to the 
    % loadings as we have previously extracted them. In order to better
    % understand the loadings Thurstone brought forward five ideal 
    % conditions of simple structure, the three most important are: 
    % > (1) each variable must have at least one near zero loading
    % > (2) each factor must have near zero loadings for at least m 
    % variables (m is the number of factors)
    % > (3) for each pair of factors there is at least m variables with
    % loadings near zero for one of them and far enough from zero for the 
    % other
    % Now we can objectively assess the loadings.

    FiguresFeat.f5=findobj('type','figure','Name','PCA Loadings Image Display - White Wine');
    if ishghandle(FiguresFeat.f5)
        close(FiguresFeat.f5)
    end
    FiguresFeat.f5=figure('Name','PCA Loadings Image Display - White Wine');
    % Notice how we need to use absolute value to produce the plot as the
    % colour scale requires positive values.  
    imagesc(abs(Princ_comp_Loadings_90_w(:,1:3)))
    colorbar
    colormap parula
    ax=gca;
    ax.YTick=1:n_features;
    ax.XTick=1:3;
    ax.YTickLabel=Feature_Names;
    xlabel('Principal Component')
    ylabel('Features'); title('Weights 3 First Principal Components in White Wine')
    
    % Now we will display the transformed mean coordinates grouped by
    % quality label. We could here use the original data as input to the
    % function parallelcoords and specify a standardisation via PCA as an
    % alternative approach.
    
    FiguresFeat.f6=findobj('type','figure','Name','90% VR PCA Projections by QCLabel - White Wine');
    if ishghandle(FiguresFeat.f6)
        close(FiguresFeat.f6)
    end
    FiguresFeat.f6=figure('Name','90% VR PCA Projections by QCLabel - White Wine');
    parallelcoords(Features_Norm_PCA_w,'Group',QCLabel_w,'Quantile',0.25)
    
    % If by means of using imagesc we could not represent negative weights, 
    % we can use a bar plot for this purpose. In the next figure, we will 
    % show the contribution of each feature to the chosen principal 
    % components in a bar plot. Again we turn to the loadings to study the
    % importance of each feature in each principal component. 
    
    FiguresFeat.f7=findobj('type','figure','Name','90% VR PCA Composition - White Wine');
    if ishghandle(FiguresFeat.f7)
        close(FiguresFeat.f7)
    end
    FiguresFeat.f7=figure('Name','90% VR PCA Composition - White Wine');
    bar3(Princ_comp_Loadings_90_w)
    colormap summer
    view([35 20])
    xlabel('Principal Components')
    ylabel('Features')
    zlabel('Loadings')
    ax=gca;
    ax.YTick=1:n_features;
    ax.YTickLabel=Feature_Names;
    ax.YTickLabelRotation = -40;
    
    % And finally we will display a star plot of the absolute value of the
    % contribution of each feature to the four principal components.
    
    FiguresFeat.f8=findobj('type','figure','Name','90% VR PCA Composition Star - White Wine');
    if ishghandle(FiguresFeat.f8)
        close(FiguresFeat.f8)
    end
    FiguresFeat.f8=figure('Name','90% VR PCA Composition Star - White Wine');
    subplot(2,1,1:2)
    glyphplot(abs(Princ_comp_Loadings_90_w'),'standardize','off','VarLabels',...
        Feature_Names,'ObsLabels',...
        {'PC 1','PC 2','PC 3','PC 4','PC 5','PC 6','PC 7'});
   % NOTE: You can use the cursor tool to explore legends and values within
   % the start plot
    
    disp(' ')
    disp('Finished Chapter A ')
    
    user_option=input('Would you like to close the figures from this chapter? Y/N [N]:','s');
    if isempty(user_option)
        user_option='n';
    end
    if strcmpi('y',user_option)
        
        for vv=1:8
            figname=strcat('f',num2str(vv));
            if ishghandle(FiguresFeat.(figname))&&strcmp(get(FiguresFeat.(figname),'BeingDeleted'),'off')
                close(FiguresFeat.(figname));
            end
        end
    end
    
   
    disp(' ')
    disp(['--------------------------------------------------------------------------------------------';...
        '------------------------------------     END     -------------------------------------------';...
        '--------------------------------------------------------------------------------------------'])
    
    
    
    fprintf('\nProgram paused. Press enter to continue.\n\n');
    pause;

    
end % PCAswitch


if FAswitch % {CHAPTER B}
    
    disp(' ')
    disp(['--------------------------------------------------------------------------------------------';...
        '------------------------------------  CHAPTER B  -------------------------------------------';...
        '--------------------------------------------------------------------------------------------'])
    
    %% Section 4-a
    % In this section we will perform Factor Analysis to reduce the
    % dimensionality of the data. Factor analysis attempts to find
    % unobserved latent statistical variables to describe the samples. As
    % opposed to PCA, where a linear unique transformation is found from
    % the original features to the principal components (variance in the 
    % data is assumed to be accounted for by means of its principal 
    % components only and hence there is no error variance explained), in 
    % factor analysis we focus on defining the variables as a linear 
    % combination of common factors plus an additional contribution of 
    % unique factors. Thus,each feature in the original data can be 
    % expressed as a linear combination of common factors plus a factor 
    % unique to such feature. 
    % 
    % In more simple words, when we perform factor analysis over a dataset
    % we attempt to identify a number of hidden statistical variables that
    % represent our data whilst acknoledging the existance of measurement
    % error and unique variance per feature. The mathematical formulation 
    % for FA is:
    
    % Given our matrix of samples X (features are columns)
    % |-----------------------------|
    % | X = Mu + Lambda·f + Epsylon | 
    % |-----------------------------|
    % Where f ~ N(0,I), Epsylon ~ N(0,Psi) and Mu represents the mean value 
    % of the features in the original data. Now if we assume the data has 
    % been converted to zero mean we have:
    % |------------------------|
    % | X = Lambda·f + Epsylon | 
    % |------------------------|
    % Where the first term reflects the common variance in the data caused 
    % by feature interaction and Epsylon models the individual variance and
    % error in the measurements.  
    
    % We will start by extracting the data from FA on the red wine using 
    % 3 common factors. The ouputs from 'factoran' in Matlab are:
    
    % - Lambda is the matrix of common factor loadings and it used to
    % understand how the different features are composed by these common
    % factors. 
    % - Psi is the max likelihood estimate of the specific variances of the
    % original features expressed as a column vector. 
    % |---------------------------------|
    % |    Cov(X)=Lambda*Lambda' + Psi  |
    % |         Psi=Cov(Epsylon)        |
    % |---------------------------------|
    % - T is the factors loading rotation matrix. For details on rotation: 
    % [http://stats.stackexchange.com/questions/151653/what-is-the-intuitive-reason-behind-doing-rotations-in-factor-analysis-pca-how]
    % - stats is a structure containing statistical information on the
    % transformation.
    % - The last output are the scores from the original data converted into
    % the new common factor. Only computed if the input type is raw sample
    % data (see covariance factor analysis for alternative). 
    
    % Factor analysis alows the user to input raw data or the estimated
    % covariance of the training samples. Two methods are supported to
    % extract the scores of the original features on the common factors: 
    % - Bartlett or wls -> Synonyms for a weighted least-squares estimate 
    % that treats the scores matrix as fixed (default)
    % - Thomson or regression -> Synonyms for a minimum mean squared error 
    % prediction that is equivalent to a ridge regression (regularised
    % regression). 
    
    % Additionally, the user can specify a rotation option for the common 
    % factors (multiple orthogonal and oblique options) and the
    % initialisation algorithm for factor analysis. For more details type
    % 'doc factoran' in Matlab command window
     
    % We will extract 3 common factors and specify the default options for
    % scores extraction and no rotation
    [Lambda_r,Psi_r,T_FA_r,stats_FA_r,Features_Norm_FA_r]=factoran(Features_r_Norm,...
        3,'xtype','data','scores','wls','rotate','none');
    % Here we receive a warning regarding low values in Psi_r (below 0.005) 
    % which indicates the estimation of the factor scores is sensitive and
    % the likelihood might have multiple local maxima. Possibel solutions
    % are to reduce the number of common factors to represent our data or
    % specify a lower tolerance for the maximum likelihood estimation
    % problem. Here we will ignore the warning as we are only extracting
    % three common factors for visualisation convenience. For more details
    % see: 
    % [http://www.mathworks.com/help/releases/R2011a/toolbox/stats/factoran.html]
    
    % We will now display the content of each of the common factors as
    % expressed using the original features (essentially the factors
    % loading matrix) using different approaches. First a biplot with the
    % scores as a cloud of points.
    
    FiguresFeat.f9=findobj('type','figure','Name','Factor Analysis Loadings Matrix for 3 CF - Red Wine');
    if ishghandle(FiguresFeat.f9)
        close(FiguresFeat.f9)
    end
    FiguresFeat.f9=figure('Name','Factor Analysis Loadings Matrix for 3 CF - Red Wine');
    
    biplot(Lambda_r,'VarLabels',Feature_Names,'scores',Features_Norm_FA_r,...
        'color',[.3765 .3765 .3765],'MarkerEdgeColor',[1 128/255 0])
    xlabel('Common Factor 1');ylabel('Common Factor 2');zlabel('Common Factor 3');
    
    % Now we will plot as an image the absolute value of the factor
    % loadings matrix. 
    
    FiguresFeat.f10=findobj('type','figure','Name','Factor Analysis Abs(Loadings Matrix) for 3 CF - Red Wine');
    if ishghandle(FiguresFeat.f10)
        close(FiguresFeat.f10)
    end
    FiguresFeat.f10=figure('Name','Factor Analysis Abs(Loadings Matrix) for 3 CF - Red Wine');
    imagesc(abs(Lambda_r))
    colorbar
    colormap summer
    ax=gca;
    ax.YTick=1:n_features;
    ax.YTickLabel=Feature_Names;
    ax.XTick=1:3;
    xlabel('Common Factors')
    ylabel('Original Features')
    
    % Again we need to understand if there are negative contributions, 
    % hence in the next figures we will use bar plots to visualise the 
    % contribution of each feature on the common factors and the 
    % composition of the factors by feature
    
    FiguresFeat.f11=findobj('type','figure','Name','Features Contribution to Abs(Loadings Matrix) by Feature - Red Wine');
    if ishghandle(FiguresFeat.f11)
        close(FiguresFeat.f11)
    end
    FiguresFeat.f11=figure('Name','Features Contribution to Abs(Loadings Matrix) by Feature - Red Wine');

    bar(abs(Lambda_r'))
    legend(Feature_Names)
    xlabel('Common Factors'); ylabel('Abs(Factor Loading)');
    
    FiguresFeat.f12=findobj('type','figure','Name','Features Contribution to Abs(Loadings Matrix) by Factor - Red Wine');
    if ishghandle(FiguresFeat.f12)
        close(FiguresFeat.f12)
    end
    FiguresFeat.f12=figure('Name','Features Contribution to Abs(Loadings Matrix) by Factor - Red Wine');

    bar(abs(Lambda_r))
    legend({'Common Factor 1';'Common Factor 2';'Common Factor 3'})
    xlabel('Features'); ylabel('Abs(Factor Loading)');
    ax = gca;
    ax.XTick = 1:n_features;
    ax.XTickLabel = Feature_Names;
    ax.XTickLabelRotation = 30;
    
    disp(' ')
    disp('Finished Chapter B ')
    
    user_option=input('Would you like to close the figures from this chapter? Y/N [N]:','s');
    if isempty(user_option)
        user_option='n';
    end
    if strcmpi('y',user_option)
        
        for vv=9:12
            figname=strcat('f',num2str(vv));
            if ishghandle(FiguresFeat.(figname))&&strcmp(get(FiguresFeat.(figname),'BeingDeleted'),'off')
                close(FiguresFeat.(figname));
            end
        end
    end
    
    disp(' ')
    disp(['--------------------------------------------------------------------------------------------';...
        '------------------------------------     END     -------------------------------------------';...
        '--------------------------------------------------------------------------------------------'])
    fprintf('\nProgram paused. Press enter to continue.\n\n');
    pause;

    
end % FAswitch


if DRswitch&&(FSswitch||PCAswitch) % {CHAPTER C}
    
    disp(' ')
    disp(['--------------------------------------------------------------------------------------------';...
        '------------------------------------  CHAPTER C  -------------------------------------------';...
        '--------------------------------------------------------------------------------------------'])
    
    %% Section 5-a
    % Now that we have performed either PCA or FA to reduce the
    % dimensionality of our test data, we will study the benefit of using
    % the reduced data to train a classification problem on the quality
    % labels of the data. Training the algorithm using the reduced data
    % would require time and, in case of success improving the performance
    % over the use of the original data, arise in the analyst the
    % demand to further investigate the meaning of the principal
    % components or common factors used. We will not cover this aspect in
    % this script. 
    
    % The fact that the dimension of the reduced data is in the order of
    % half the dimension of the original data suggests it might be possible
    % that a simpler approach to classification in terms of model 
    % complexity could lead to improved errors. This, however, will not be 
    % researched in this script. 
    
    % We will compare the performance and trinaing time for a multi class
    % SVM using both the original features vs the transformed features.
    % Notice how I used the word transformed to refer to either PCA or FA.
    % In our simple analysis we will fix the following inputs:
    % > Polynomial Kernel of order 2
    % > Prior uniform
    % > Default solver SMO
    % > No Kernel scaling
    % > 3-Fold cross validation
    % > 1 vs All ECOC algorithm
    
    % And we will perform a brute force search over the regularisation box
    % constraint and take the best result.
    
    % Let us begin by training the model on Red / White wine using the 
    % original features first. Then we will train the model using the 
    % transformed features. As we performed PCA over the white wine data
    % and Factor Analysis over the red wine data, we will train either
    % model depending on the availability of the previous data, that means,
    % whether the analysis was performed on Chapter B.  
    
    % Define the box constraint candidate values. Penalty on 
    % misclassification distance 
    BoxConsVals=[0.1 1 10 100];
    
    % Note we will not perform any search in the optimisation of the kernel
    % scale as this would mean a significant increase in the training time.
    
   
    
    if PCAswitch 
        
        SVM_model_w=fitcecoc(0,0);
        SVM_model_w_PCA=fitcecoc(0,0);
        Val_Err_w=ones(size(BoxConsVals));
        Val_Err_w_PCA=Val_Err_w;
        Red_Time=zeros(size(BoxConsVals));
        NonRed_Time=Red_Time;
        
        if Parallelswitch
            
            % Start the parallel pool to ensure more predictable results
            delete(gcp('nocreate'));
            poolobj=parpool('local',n_cores);
            
        end
        
        % Create wait bar  
        h_wait = waitbar(0,'Please wait','Name','SVM Box Constraint Iterations');
        for ii=1:length(BoxConsVals)
            
            C=BoxConsVals(ii);
            
            % Update the waitbar
            waitbar(ii/length(BoxConsVals),h_wait,['Training SVM on White Wine using Box Constraint C = ',...
                num2str(C)]);
            
            % We will create an SVM template using fixed settings
            
            Template_Learner=templateSVM('KernelFunction','polynomial',...
                'PolynomialOrder',2,'BoxConstraint',C,'Prior','uniform',...
                'Solver','SMO');
            
            % Fit an error correcting output codes classifier using a one 
            % vs all approach. Use parallelisation if available.
            
            if Parallelswitch
                
                % Set options to use parallelisation
                ecoc_options = statset('UseParallel',1);
                % Train on original features
                tic
                SVM_model_w=fitcecoc(Features_w_Norm,QCLabel_w,'Learners',...
                    Template_Learner,'Coding','onevsall',...
                    'KFold',3,'Options',ecoc_options);
                NonRed_Time(ii)=toc;
                
                % Train on transfomed features from PCA (scores)
                tic
                SVM_model_w_PCA=fitcecoc(Features_Norm_PCA_w,QCLabel_w,'Learners',...
                    Template_Learner,'Coding','onevsall',...
                    'KFold',3,'Options',ecoc_options);
                Red_Time(ii)=toc;
                
            else
                % Train models without using parallelisation
                % Train on original features
                tic
                SVM_model_w=fitcecoc(Features_w_Norm,QCLabel_w,'Learners',...
                    Template_Learner,'Coding','onevsall','KFold',3);
                NonRed_Time(ii)=toc;
                
                % Train on transfomed features from PCA (scores)
                tic
                SVM_model_w_PCA=fitcecoc(Features_Norm_PCA_w,QCLabel_w,'Learners',...
                    Template_Learner,'Coding','onevsall','KFold',3);
                Red_Time(ii)=toc;
            end
            
            % Extract k-fold validation error
            Val_Err_w(ii)=SVM_model_w.kfoldLoss;
            Val_Err_w_PCA(ii)=SVM_model_w_PCA.kfoldLoss;
            
        end
        
        delete(h_wait)   % DELETE the waitbar; don't try to CLOSE it
        
        if Parallelswitch
            
            % Close parallel pool
            delete(gcp('nocreate'));
        end
        
        % Extract best performers for each of the two approaches.
        [Best_SVM_w_Err,Best_SVM_w_ind]=min(Val_Err_w);
        [Best_SVM_w_PCA_Err,Best_SVM_w_PCA_ind]=min(Val_Err_w_PCA);
        
        % Show results
        disp({'Feature Dimensionality Reduction - WHITE WINE QC Label:';...
            'SVM 1 vs All ECOC - Polynomial Kernel order 2 - Prior Uniform - 5 Fold Cross Validation';...
            ['Best error using standardised original features is: ',num2str(Best_SVM_w_Err*100,'%.4g'),...
            ' [%] using a box constraint equal to ',num2str(BoxConsVals(Best_SVM_w_ind))];...
            ['Best error using the four first Principal Components is: ',num2str(Best_SVM_w_PCA_Err*100,'%.4g'),...
            ' [%] using a box constraint equal to ',num2str(BoxConsVals(Best_SVM_w_PCA_ind))]});
        
        disp(' ')
        
         disp(['Time required to train each SVM for all possible box constraint values ',...
            'with White Wine original features in seconds '])
        disp(NonRed_Time)
        disp('vs using PCA reduced features ')
        disp(Red_Time)
        
    end
    
    
    %% Section 5-b
    % Now we will perform the same analysis but if data is available for
    % the Factor Analysis of the red wine features.
    
     if FAswitch
        
        SVM_model_r=fitcecoc(0,0);
        SVM_model_r_FA=fitcecoc(0,0);
        Val_Err_r=ones(size(BoxConsVals));
        Val_Err_r_FA=Val_Err_r;
        Red_Time=zeros(size(BoxConsVals));
        NonRed_Time=Red_Time;
        
        if Parallelswitch
            
            % Start the parallel pool to ensure more predictable results
            delete(gcp('nocreate'));
            poolobj=parpool('local',n_cores);
            
        end
        
        % Create wait bar  
        h_wait = waitbar(0,'Please wait','Name','SVM Box Constraint Iterations');
        
        for ii=1:length(BoxConsVals)
            
            C=BoxConsVals(ii);
            
            % Update the waitbar
            waitbar(ii/length(BoxConsVals),h_wait,['Training SVM on White Wine using C = ',...
                num2str(C)]);
            
            % We will create an SVM template using fixed settings
            
            Template_Learner=templateSVM('KernelFunction','polynomial',...
                'PolynomialOrder',2,'BoxConstraint',C,'Prior','uniform',...
                'Solver','SMO');
            
            % Fit an error correcting output codes classifier using a one
            % vs all approach. Use parallelisation if available.
            
            if Parallelswitch
                
                % Set options to use parallelisation
                ecoc_options = statset('UseParallel',1);
                % Train on original features
                tic
                SVM_model_r=fitcecoc(Features_r_Norm,QCLabel_r,'Learners',...
                    Template_Learner,'Coding','onevsall',...
                    'KFold',3,'Options',ecoc_options);
                NonRed_Time(ii)=toc;
                
                % Train on transformed features using FA (scores)
                tic
                SVM_model_r_FA=fitcecoc(Features_Norm_FA_r,QCLabel_r,'Learners',...
                    Template_Learner,'Coding','onevsall',...
                    'KFold',3,'Options',ecoc_options);
                Red_Time(ii)=toc;
                
            else
                % Train models without using parallelisation
                % Train on original features
                tic
                SVM_model_r=fitcecoc(Features_r_Norm,QCLabel_r,'Learners',...
                    Template_Learner,'Coding','onevsall','KFold',3);
                NonRed_Time(ii)=toc;
                
                % Train on transformed features using FA (scores)
                tic
                SVM_model_r_FA=fitcecoc(Features_Norm_FA_r,QCLabel_r,'Learners',...
                    Template_Learner,'Coding','onevsall','KFold',3);
                Red_Time(ii)=toc;
            end
            
            % Extract k-fold validation error
            Val_Err_r(ii)=SVM_model_r.kfoldLoss;
            Val_Err_r_FA(ii)=SVM_model_r_FA.kfoldLoss;
            
        end
        
        delete(h_wait)   % DELETE the waitbar
       
        if Parallelswitch
            
            % Close parallel pool
            delete(gcp('nocreate'));
        end
        
        % Extract best performers for each of the two approaches.
        [Best_SVM_r_Err,Best_SVM_r_ind]=min(Val_Err_r);
        [Best_SVM_r_FA_Err,Best_SVM_r_FA_ind]=min(Val_Err_r_FA);
        
        % Show results
        disp({'Feature Dimensionality Reduction - RED WINE QC Label:';...
            'SVM 1 vs All ECOC - Polynomial Kernel order 2 - Prior Uniform - 5 Fold Cross Validation';...
            ['Best error using standardised original features is: ',num2str(Best_SVM_r_Err*100,'%.4g'),...
            ' [%] using a box constraint equal to ',num2str(BoxConsVals(Best_SVM_r_ind))];...
            ['Best error using three Common Factors is: ',num2str(Best_SVM_r_FA_Err*100,'%.4g'),...
            ' [%] using a box constraint equal to ',num2str(BoxConsVals(Best_SVM_r_FA_ind))]});
        
        disp(' ')
        
        disp(['Time required to train each SVM for all possible box constraint values ',...
            'with Red Wine original features in seconds '])
        disp(NonRed_Time)
        disp('vs using FA reduced features ')
        disp(Red_Time)
        
    end
    
    
    
    
    disp(' ')
    disp(['--------------------------------------------------------------------------------------------';...
        '------------------------------------     END     -------------------------------------------';...
        '--------------------------------------------------------------------------------------------'])
    fprintf('\nProgram paused. Press enter to continue.\n\n');
    pause;

end % DRswitch

if FSswitch % {CHAPTER D}
    
    disp(' ')
    disp(['--------------------------------------------------------------------------------------------';...
        '------------------------------------  CHAPTER D  -------------------------------------------';...
        '--------------------------------------------------------------------------------------------'])
    
    %% Section 6-a
    % In the previous sections form this script we have investigated
    % feature reduction through the use of Common Factors and Principal
    % Components. These approaches are frequently chosen when the physical
    % meaning of the data features available is uncertain or in search of 
    % discovering new relationships in the data available that we have not 
    % identified before. In addition, Factor Analysis is frequently used
    % when it is suspected that the data might have been affected by high
    % levels of noise and there is an underlying statistical distribution
    % to be discovered. 
    
    % There are other problems where the selection and extraction of the
    % features was performed via expert knowledge or following a strict
    % methodology, hence it would be desirable to retain the original
    % features as extracted. However, some of these features might
    % constitute uncorrelated noise or decrease the classification ability
    % of a specific algorithm. Hence, in this section we will introduce
    % Feature Selection, a process by which a subset of features is
    % selected from the original set in order to improve the performance of
    % the classification algorithm, reduce computation load and memory
    % requirements. 
    
    % Latest research acknowledges the existence of four feature selection
    % methods with variants:
    
    % > Wrapper Methods: These methods rely on the use of the target
    % classification algorithm performance to choose the set of features
    % that produce the best results. Because of this, wrapper methods are
    % heavily influenced by the algorithm and hyperparameters choice and
    % require complex cross validation strategies to ensure unbiased
    % results. 
    % Common wrapper methods include sequential forwards and backwards
    % feature selection, a methodology by which we initially take an empty 
    % or the complete set of features and sequentially attempt to add or 
    % remove one feature until there is no further performance improvement.
    % Because of their nature, wrapper methods can easily be biased towards
    % the algorithm applied, hence it is good practice to split the data
    % points in a cross validation structure where feature selection is
    % applied to the training set prior to training. 
    
    % > Filter Methods: These methods are based on the evaluation of the
    % correlation between features, the class labels and other features
    % from the total set. Hence, the idea behind is to discard those
    % features that can be proved independent from the class labels or the
    % class labels and the rest of features. In research it is frequent to 
    % define a criterion and a threshold value below which no feature is
    % kept. Thus, a feature conditionally independent from the class label,
    % can be discarded as irrelevant. Criteria frequently used include:
    % - Pearson Correlation
    % - Mutual Information
    % - Distance metrics by label
    
    % Some of these methods suffer the drawback of ignoring complex
    % correlations that can exist amongst multiple features but do not
    % require the evaluation of the classification algorithm and hence are
    % much more time efficient than wrapper methods. 
    
    % The concept of a Markov Blanket can be very helpful when making the
    % most of filter methods. It is defined as follows:
    % Let M be some set of features which does not contain Fi (specific
    % feature). We say M is a Markov Blanket for Fi if Fi is conditionally
    % independent from F-M-{Fi} given M (where F is the total set of 
    % features available). When two features are removed using the Markov
    % Blanket criterion, the remaining set of features constitute a Markov
    % Blanket for both, making this an excellent criterion for sequential
    % feature elimination. Calculating a Markov Blanket can be a very hard
    % problem, luckily there are multiple heuristics that aim at its
    % approximation once existence is assumed. 
    
    % > Embedded Methods: This designation is used for methods on which
    % feature elimination is perform within the classification algorithm.
    % Hence the computational cost is reduced with respect to wrapper
    % methods. Some example include the elimination of features based on
    % SVM factors or low excitation of nodes in Neural Networks.
    
    % > Hybrid Methods: These combine filter and wrapper methods to find a
    % trade off in large feature sets. The idea is to define a coefficient
    % of hybridisation refering to how much selection will be performed via
    % wrapper methods as opposed to filter methods. The idea is to first
    % apply filter methods to eliminate a number of features followed by
    % wrapper classifier evaluation to continue and repeat multiple 
    % iterations of the complete process. 
    
    % For more information or guides on feature selection you can review
    % this publication.
    % [http://jmlr.csail.mit.edu/papers/volume3/guyon03a/guyon03a.pdf]
    
    % Or check this great presentation on Feature Selection: 
    % [http://research.cs.tamu.edu/prism/lectures/pr/pr_l11.pdf]
    
    % Matlab offers built in functionalities to perform wrapper feature
    % selection via sequential forwards or backwards approaches. In this
    % chapter of the script we will apply both to the red and white wine 
    % data to understand whether the elimination of features based on 
    % K-Nearest-Neighbours classification leads to results improvement.
    
    % For all the subsequent analyses we will fix the number of neighbours
    % to 15, the distance metric to cosine (one minus the cosine of the 
    % included angle between observations ), the Prior to be empirical, as 
    % we know the number of samples in QC classes C and D are higher, and 
    % inverse distance weights. We will use 30% cross validation for all
    % the analyses.
    
    % We will begin by defining the partitions for both red and white
    % wines. 
    
    %% First the red wine, let us create a CV partition
    HoldOut_percent=0.3;
    partition_r=cvpartition(QCLabel_r,'HoldOut',HoldOut_percent);
    
    % Generate training and test sets
    TrainData_r=Features_r_Norm(partition_r.training,:);
    TrainLabel_r=QCLabel_r(partition_r.training);
    TestData_r=Features_r_Norm(partition_r.test,:);
    TestLabel_r=QCLabel_r(partition_r.test);
    
    % Then the white wine.
    partition_w=cvpartition(QCLabel_w,'HoldOut',HoldOut_percent);
    
    % Generate training and test sets
    TrainData_w=Features_w_Norm(partition_w.training,:);
    TrainLabel_w=QCLabel_w(partition_w.training);
    TestData_w=Features_w_Norm(partition_w.test,:);
    TestLabel_w=QCLabel_w(partition_w.test);
    
    %% Now let us train the model using all the features for the red wine
    KNN_model_r=fitcknn(TrainData_r,TrainLabel_r,'NumNeighbors',15,...
        'Distance','cosine','DistanceWeight','inverse','Prior','empirical');
    KNN_trainErr_r=100*resubLoss(KNN_model_r);
    KNN_validErr_r=100*loss(KNN_model_r,TestData_r,TestLabel_r);
    
    % Predict on the validation set
    KNN_pred_r=predict(KNN_model_r,TestData_r);
    
    disp({['KNN classifier including all features for red wine',...
        ' Error:'];['On train set: ',num2str(KNN_trainErr_r,'%.4g'),'%'];...
        ['On validation set: ',num2str(KNN_validErr_r,'%.4g'),'%']});
    
    % Calculate the confusion matrix for this classifier
    [conf_mtrx_KNN_r,conf_mtrx_KNN_r_label]=confusionmat(TestLabel_r,KNN_pred_r);
    
    FiguresFeat.f13=findobj('type','figure','Name','KNN with All Features - Confusion Matrix Red Wine - Image');
    if ishghandle(FiguresFeat.f13)
        close(FiguresFeat.f13)
    end
    FiguresFeat.f13=figure('Name','KNN with All Features - Confusion Matrix Red Wine - Image');
    
    % Display the confusion matrix as an image scaled to cover the full
    % colormap
    imagesc(conf_mtrx_KNN_r)
    % Display colour bar legend
    colorbar
    % Change colour map
    colormap summer
    
    ylabel('True Class')
    xlabel('Predicted Class')
    ax=gca;
    ax.XTick=1:5; ax.XTickLabel=categories(conf_mtrx_KNN_r_label);
    ax.YTick=1:5; ax.YTickLabel=categories(conf_mtrx_KNN_r_label);
    
    % Add labels at the center of the image
    [xpos,ypos] = meshgrid(1:size(conf_mtrx_KNN_r,1),1:size(conf_mtrx_KNN_r,2));
    text(xpos(:),ypos(:),num2str(conf_mtrx_KNN_r(:),'%d'),'HorizontalAlignment','Center')
    
    %% Now repeat the training for the white wine
    
    KNN_model_w=fitcknn(TrainData_w,TrainLabel_w,'NumNeighbors',15,...
        'Distance','cosine','DistanceWeight','inverse','Prior','empirical');
    KNN_trainErr_w=100*resubLoss(KNN_model_w);
    KNN_validErr_w=100*loss(KNN_model_w,TestData_w,TestLabel_w);
    
    % Predict on the validation set
    KNN_pred_w=predict(KNN_model_w,TestData_w);
    
    disp({['KNN classifier including all features for white wine',...
        ' Error:'];['On train set: ',num2str(KNN_trainErr_w,'%.4g'),'%'];...
        ['On validation set: ',num2str(KNN_validErr_w,'%.4g'),'%']});
    
    % Calculate the confusion matrix for this classifier
    [conf_mtrx_KNN_w,conf_mtrx_KNN_w_label]=confusionmat(TestLabel_w,KNN_pred_w);
    
    FiguresFeat.f14=findobj('type','figure','Name','KNN with All Features - Confusion Matrix White Wine - Image');
    if ishghandle(FiguresFeat.f14)
        close(FiguresFeat.f14)
    end
    FiguresFeat.f14=figure('Name','KNN with All Features - Confusion Matrix White Wine - Image');
    
    % Display the confusion matrix as an image scaled to cover the full
    % colormap
    imagesc(conf_mtrx_KNN_w)
    % Display colour bar legend
    colorbar
    % Change colour map
    colormap summer
    
    ylabel('True Class')
    xlabel('Predicted Class')
    ax=gca;
    ax.XTick=1:5; ax.XTickLabel=categories(conf_mtrx_KNN_w_label);
    ax.YTick=1:5; ax.YTickLabel=categories(conf_mtrx_KNN_w_label);
    
    % Add labels at the center of the image
    [xpos,ypos] = meshgrid(1:size(conf_mtrx_KNN_w,1),1:size(conf_mtrx_KNN_w,2));
    text(xpos(:),ypos(:),num2str(conf_mtrx_KNN_w(:),'%d'),'HorizontalAlignment','Center')
    
    %% Section 6-b
    % Now that we have a reference value we will perform feature selection
    % and compare the results after fetaures are removed. We will start 
    % with the red wine samples as before. 
    
    % The function sequentialfs requires an error function capable of
    % generating cross validation results, this means when both train and
    % test sets are presented to it. Hence, the most intuitive approach is
    % to define a model function and embed the latter within an error
    % calculation function. We will specify our cross validation to use the
    % CV object we have created earlier. 
    
    % Define prediction and error functions
    predFcn = @(xtrain,ytrain,xtest) predict(fitcknn(xtrain,ytrain,...
        'Numneighbors',15,'Distance','cosine','DistanceWeight','inverse',...
        'Prior','empirical'),xtest);
    
    % A very important note on defining the error function is that 
    % sequentialfs divides the sum of the values returned by errFcn across
    % all test sets by the total number of test observations. Accordingly, 
    % errFcn should not divide its output value by the number of test 
    % observations. We will use the number of misclassifications as the
    % error value. 
    errFcn = @(xtrain,ytrain,xvalid,yvalid) nnz(yvalid ~= predFcn(xtrain,ytrain,xvalid));
    
    % Extract selected features using forwards sequential feature
    % selection. Use parallelisation if available. We will request two
    % outputs from the feature selection function: The first is a logical
    % vector indicating the selected features and the second a structure
    % containing historical data from the algorithm. 
    if Parallelswitch
        
        % Start the parallel pool to ensure more predictable results
        delete(gcp('nocreate'));
        poolobj=parpool('local',n_cores);
        
        [KNN_selFeat_fwd_r,history_selFeat_fwd_r]=sequentialfs(errFcn,...
            Features_r_Norm,QCLabel_r,...
            'cv',partition_r,'direction','forward','options',...
            statset('Display','iter','UseParallel',true));  
    else
        
        [KNN_selFeat_fwd_r,history_selFeat_fwd_r]=sequentialfs(errFcn,...
            Features_r_Norm,QCLabel_r,...
            'cv',partition_r,'direction','forward','options',...
            statset('Display','iter'));
    end
    
    % We can extract the minimum error that caused search to stop by
    % looking at the history of criterion evaluation and extracting the
    % lowest value. 
    disp(' ')
    disp({['KNN classifier forwards feature selection for red wine',...
        ' Error:'];['On validation set: ',...
        num2str(100*min(history_selFeat_fwd_r.Crit),'%.4g'),'%'];'Selected Features:'});
    disp(Feature_Names(KNN_selFeat_fwd_r))
    
    % Now use backwards approach
    if Parallelswitch
        
        [KNN_selFeat_bwd_r,history_selFeat_bwd_r]=sequentialfs(errFcn,...
            Features_r_Norm,QCLabel_r,...
            'cv',partition_r,'direction','backward','options',...
            statset('Display','iter','UseParallel',true));   
    else
        
        [KNN_selFeat_bwd_r,history_selFeat_bwd_r]=sequentialfs(errFcn,...
            Features_r_Norm,QCLabel_r,...
            'cv',partition_r,'direction','backward','options',...
            statset('Display','iter'));
    end
    
    disp(' ')
    disp({['KNN classifier backwards feature selection for red wine',...
        ' Error:'];['On validation set: ',...
        num2str(100*min(history_selFeat_bwd_r.Crit),'%.4g'),'%'];'Selected Features:'});
    disp(Feature_Names(KNN_selFeat_bwd_r))
    
    %% Section 6-c
    % Here we will repeat the feature selection process but using the white
    % wine data.
    
    if Parallelswitch
        
        [KNN_selFeat_fwd_w,history_selFeat_rand_w]=sequentialfs(errFcn,...
            Features_w_Norm,QCLabel_w,...
            'cv',partition_w,'direction','forward','options',...
            statset('Display','iter','UseParallel',true));
    else
        
        [KNN_selFeat_fwd_w,history_selFeat_rand_w]=sequentialfs(errFcn,...
            Features_w_Norm,QCLabel_w,...
            'cv',partition_w,'direction','forward','options',...
            statset('Display','iter'));
    end
    
    disp(' ')
    disp({['KNN classifier forwards feature selection for white wine',...
        ' Error:'];['On validation set: ',...
        num2str(100*min(history_selFeat_rand_w.Crit),'%.4g'),'%'];'Selected Features:'});
    disp(Feature_Names(KNN_selFeat_fwd_w))
    
    % Now use backwards approach
    
    if Parallelswitch
    
        [KNN_selFeat_bwd_w,history_selFeat_bwd_w]=sequentialfs(errFcn,...
            Features_w_Norm,QCLabel_w,...
            'cv',partition_w,'direction','backward','options',...
            statset('Display','iter','UseParallel',true));          
    else
        
        [KNN_selFeat_bwd_w,history_selFeat_bwd_w]=sequentialfs(errFcn,...
            Features_w_Norm,QCLabel_w,...
            'cv',partition_w,'direction','backward','options',...
            statset('Display','iter'));
    end
    
    disp(' ')
    disp({['KNN classifier backwards feature selection for white wine',...
        ' Error:'];['On validation set: ',...
        num2str(100*min(history_selFeat_bwd_w.Crit),'%.4g'),'%'];'Selected Features:'});
    disp(Feature_Names(KNN_selFeat_bwd_w))
    
    % The remaining task would be to assess the characteristics of those
    % features rejected by either forwards or backwards selection, perhaps
    % by means of using filter methods to strengthen the justification
    % behind any feature removal. It is also possible to combine backwards
    % and forwards while imposing that any feature chosen by forward
    % selection cannot be removed by backward selection. This could easily
    % be done by taking the logical OR result from the logical vectors
    % produded by the two approaches. 
    
    %% Section 6-d 
    % Sequential feature selection shows what could be defined as a history 
    % drawback. This only means that forward search is unable to reassess
    % the elimination of features that become obsolete after the addition
    % of further features while backwards selection is unable to reconsider
    % the addition of a feature already removed. Two approaches are
    % commonly taken to overcome this problem. 
    
    % Significant research has shown the benefits of floating methods, 
    % Plus-L minus-R selection or Bidirectional Search; methods which
    % combine forward and backward search to provide some backtracking
    % capability. An example can be found here:
    % {Somol, Petr, et al. "Adaptive floating search methods in feature 
    % selection." Pattern recognition letters 20.11 (1999): 1157-1163.}
    % [http://perclass.com/research/somol99_PRL.pdf]
    
    % Randomised search enables the analyst to include heuristics to widen
    % the search space and explore further combinations. We will now
    % perform a very simple example on modifications over sequential search
    % by means of random logic (note these approaches are suitable for
    % large number of features, here we only count on eleven). 
    
    % Let us assume we define a subset of around a third of the features 
    % available in the data and we force sequential forward search to keep 
    % these features. We can assess how randomised alterations of SFS 
    % improve or decrease the performance of the chosen set. We will be 
    % using white wine data exclusively for this example.
    
    disp(['---------------------------------------------------------------';
        '--------------- Randomised Brute Force Search -----------------';...
        '---------------------------------------------------------------']);
    
    % Repeat 10 times
    for uu=1:10
        % Index to random third of features
        Ind_Forced=randi([1 n_features],1,floor(n_features/3));
        % Repeat generation ensuring no repetition of features in the
        % random selection
        while length(Ind_Forced)>length(unique(Ind_Forced))
            Ind_Forced=randi([1 n_features],1,3);
        end
        % Preapre boolean vector to present to the function sequentialfs in
        % order to specify the features that must be kept
        KNN_Forced_selFeat_w=boolean(zeros(1,n_features));
        KNN_Forced_selFeat_w(Ind_Forced)=true;
       
        
        if Parallelswitch
            % Specify the option keepin to declare features that must be
            % kept
            [KNN_selFeat_rand_w,history_selFeat_rand_w]=sequentialfs(errFcn,...
                Features_w_Norm,QCLabel_w,'keepin',KNN_Forced_selFeat_w,...
                'cv',partition_w,'direction','forward','options',...
                statset('Display','iter','UseParallel',true));
        else
            
            [KNN_selFeat_rand_w,history_selFeat_rand_w]=sequentialfs(errFcn,...
                Features_w_Norm,QCLabel_w,'keepin',KNN_Forced_selFeat_w,...
                'cv',partition_w,'direction','forward','options',...
                statset('Display','iter'));
        end
        
        disp(' ')
        disp({['KNN classifier randomised forwards feature selection for white wine',...
            ' Error:'];['On validation set: ',...
            num2str(100*min(history_selFeat_rand_w.Crit),'%.4g'),'%'];...
            'Imposed Selected Features:'});
        disp(Feature_Names(KNN_Forced_selFeat_w))
        disp('Optimal Selected Features:')
        disp(Feature_Names(KNN_selFeat_rand_w))
        
        
    end
    
    disp(['---------------------------------------------------------------';...
        '---------------------------------------------------------------']);
    
    
    
    % Close parallel pool
    delete(gcp('nocreate'));
        
    disp(' ')
    disp('Finished Chapter D ')
    
    user_option=input('Would you like to close the figures from this chapter? Y/N [N]:','s');
    if isempty(user_option)
        user_option='n';
    end
    if strcmpi('y',user_option)
        
        for vv=13:14
            figname=strcat('f',num2str(vv));
            if ishghandle(FiguresFeat.(figname))&&strcmp(get(FiguresFeat.(figname),'BeingDeleted'),'off')
                close(FiguresFeat.(figname));
            end
        end
    end
    
    disp(' ')
    disp(['--------------------------------------------------------------------------------------------';...
        '------------------------------------     END     -------------------------------------------';...
        '--------------------------------------------------------------------------------------------'])
    
end % FSswitch

%--------------------------------------------------------------------------
%--------------------------------------------------------------------------


































































