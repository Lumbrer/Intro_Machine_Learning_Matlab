%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% $Author: Lumbrer $    $Date: 2016/12/20 $    $Revision: 2.2$
% Copyright: Francisco Lumbreras
%
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% File Name: VinhoVerde_Quality_SupervisedLearning.m
% Description: Script to perform supervised learning via a number of
% different algorithms on the data from Cortez et al. 2009 'Modeling wine
% preferences by data mining from physicochemical properties'.
% 
% This file has been coceived as a script but can be easily converted into
% a parameterised GUI to select data and train different types of 
% algorithms on Matlab. In the first section of the script a number of 
% switches are defined to activate or deactivate the different chapters of 
% the script. These can all run independently from each other.
%
% In this script, we will analyse some of the built in functions that 
% Matlab provides to perform supervised learning. The chapters here defined
% are:
%
% A) A study on clustering via K-Nearest Neighbours and the influence of
% applying different distance metrics and distance factors. For v. 2016b
% users, Bayesian optimisation is used to optimise hyperparameters and an
% alternative brute force optimisation within a selected hyperspace is
% offered for users of earlier versions. Simple parallelisation is used to
% reduce the computation time. 
%
% B) A study on Naive Bayes classification and the possible distributions
% that are predefined as options in this analysis. The assumptions from
% Naive Bayes uses independency accross feature probabilities given a 
% classification label (these probabilities are typically assumed 
% multinomial or Gaussian) and here we will challenge this assumption for 
% the case of continuous Gaussian distributions. 
%
% C) An analysis of Gaussian Discriminant analysis in a binary decision
% problem and the different assumptions that can be made when selecting
% multivariate Gaussians for the feature posteriori distributions
% P(fi|Class). We will use functions to check the validity of Gaussian
% assumptions and examine covariances across features and labels.
%
% D) A chapter covering decision trees as classifiers and the tools offered
% to train and constraint these structures. 
%
% E) The introduction of SVMs in Matlab, different Kernel approaches and
% solvers that can be used for this maximum margin binary classifier.
% Parallelisation is used to perform brute force optimisation of
% hyperparameters and different regularisation strategies are analysed.
% An additional section within this chapter covers the use of Error
% Correcting Output Codes to apply SVM classification to multiclass
% problems. 
%
% F) A brief chapter on pattern recognition Neural Networks with code on
% how to generate, train and examine the performance of a Network. This
% chapter includes a subsession analysis the different options that can be
% parameterised in a Neural Netowrk. Activation functions, architecture 
% optimisation and training functions are revised.
% Convolutional Networks, Deep Belief Networks and Auto Regressive Networks
% are not covered in this section and for such analysis Python libraries
% are preferred by the author.
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
FiguresSup=struct;


%% Define switches to activate each of the chapters of the script
% Switches for chapters A, B, C, D, E & F
KNNswitch=boolean(1);        % A - Activate k-Nearest Neighbours analysis
NBswitch=boolean(1);         % B - Activate Naive Bayes classification 
                             % analysis
GDswitch=boolean(1);         % C - Activate the Gaussian Discriminant 
                             % analysis
DTswitch=boolean(1);         % D - Activate the chapter of the script 
                             % introducing Decision Trees
SVMswitch=boolean(1);        % E - Activate Support Vector Machine analysis
NNswitch=boolean(1);         % F - Activate Neural Network analysis

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
% Extract the features in a numeric matrix. We will use rescaling to [0,1]
% for both types of wines in this script.

Features=DataClean{:,1:end-1};

% Extract number of features and their names
n_features=size(Features,2);
Feature_Names=DataClean.Properties.VariableNames(1:end-1);

% Rescale White & Red wine data to the range [0,1] and extract nummeric 
% feature arrays
WhiteData_Norm=normTable(WhiteData,{'Range'});
Features_w=WhiteData{:,:};
Features_w_Norm=WhiteData_Norm{:,:};
RedData_Norm=normTable(RedData,{'Range'});
Features_r=RedData{:,:};
Features_r_Norm=RedData_Norm{:,:};


if KNNswitch % {CHAPTER A}
    
    
    disp(' ')
    disp(['--------------------------------------------------------------------------------------------';...
        '------------------------------------  CHAPTER A  -------------------------------------------';...
        '--------------------------------------------------------------------------------------------'])
    %% Section 3-a
    % One of the simplest form of classification is via the use of the
    % k-Nearest Neighbours algorithm. In this approach, each of the test
    % samples is assigned the class resulting after a majority vote accross
    % the 'k' nearest training neighbours extracted by means of
    % minimising a specific distance criterion.
    
    % Firstly we will train a model without using a CV partition and fix K
    % to the two closest neighbours and the distance to be Minkowski with
    % exponent three. Here we use the red wine data exclusively. 
    
    KNN_model_basic=fitcknn(Features_r_Norm,QCLabel_r,'NumNeighbors',2,...
        'Distance','minkowski','Exponent',3);
    
    KNN_basic_pred=predict(KNN_model_basic,Features_r_Norm);
    KNN_basic_Err=KNN_basic_pred~=QCLabel_r;
    KNN_basic_Err_percent=100*nnz(KNN_basic_Err)/length(KNN_basic_Err);
    
    disp(['Error of KNN classifier for Red Wine using k=3 and Minkowski',...
        ' distance with Exponent=3 is: ',num2str(KNN_basic_Err_percent,'%.4g'),...
        ' % Misclassification']);
    
    %% Section 3-b
    % K-Nearest Neighbours is notorious for producing high levels of
    % overfitting for small values of K. This is reasonable as the model
    % depicts a non linear boundary between the datapoints that contours 
    % the training set in a shape dependent on the distance metric chosen.
    
    % We will now study how the validation error on a 30% hold-out set
    % varies as the number of neighbours considered is increased. The
    % distance metric used will remain as Minkowski with p=3.
    
    % Firstly, let us create a CV partition
    HoldOut_percent=0.3;
    partition=cvpartition(QCLabel_r,'HoldOut',HoldOut_percent);
    
    % Generate training and test sets
    TrainData_r=Features_r_Norm(partition.training,:);
    TrainLabel_r=QCLabel_r(partition.training);
    TestData_r=Features_r_Norm(partition.test,:);
    TestLabel_r=QCLabel_r(partition.test);
    
    % Preallocate the error vectors and classification k-NN models
    numneighbours=[1:9,10:10:40];
    KNN_Iter_Err=zeros(length(numneighbours),2);
    KNN_Iter=fitcknn(0,0);
    % Create an index to loop along the analyses as these are completed and
    % results must be stored
    ind_Err=1;
    
    % Loop and populate training and test errors
    for kk=numneighbours
        KNN_Iter=fitcknn(TrainData_r,TrainLabel_r);
        KNN_Iter.Distance='Minkowski';
        KNN_Iter.DistParameter=3;
        KNN_Iter.NumNeighbors=kk;
        % Training error in percentage
        KNN_Iter_Err(ind_Err,1)=100*KNN_Iter.resubLoss;
        % Test error in percentage
        KNN_Iter_Err(ind_Err,2)=100*KNN_Iter.loss(TestData_r,TestLabel_r);
        % Increase index
        ind_Err=ind_Err+1;
    end
    clear ind_Err
    
    % Plot the results
    FiguresSup.f1=findobj('type','figure','Name','Iterative k-NN Error on Red Wine');
    if ishghandle(FiguresSup.f1)
        close(FiguresSup.f1)
    end
    FiguresSup.f1=figure('Name','Iterative k-NN Error on Red Wine');
    
    plot(numneighbours',KNN_Iter_Err(:,1),'o-',numneighbours',...
        KNN_Iter_Err(:,2),'o-');
    grid on
    xlabel('Number Neighbours (k)')
    ylabel('30% Hold Out Error [%]')
    legend('Training Set','Validation Set')
    
    %% Section 3-c
    % We will now fix the value of K and visualise the prediction error 
    % over the validation set. This time we will use the cosine distance (1
    % minus the cosine between each pair of observations) and we will 
    % introduce distance weights equal to the inverse of the distance 
    % (closer points are more influential on class decision)
    
    KNN_Iter=fitcknn(TrainData_r,TrainLabel_r,'Distance','cosine',...
        'DistanceWeight','inverse','NumNeighbors',10);
    
    % Predict label on test data
    TestLabel_r_pred=KNN_Iter.predict(TestData_r);
    
    % Let us extract the confusion matrix for the validation set. The first
    % output of confusionmat is the confusion matrix where the rows
    % correspond to the true label of the sample. The second output
    % represents the order in which classes are sorted in the rows and
    % columns of the confusion matrix. The latter is a categorical array if
    % the input was categorical.
    
    [conf_mtrx_KNN,conf_mtrx_KNN_label]=confusionmat(TestLabel_r,TestLabel_r_pred);
    
    % We will plot this confusion matrix as a bar plot
    FiguresSup.f2=findobj('type','figure','Name','k-NN (k=10) Confusion Matrix on Red Wine');
    if ishghandle(FiguresSup.f2)
        close(FiguresSup.f2)
    end
    FiguresSup.f2=figure('Name','k-NN (k=10) Confusion Matrix on Red Wine');
    
    % In bar3 the of the input rows appear as the y axis and hence the y 
    % axis will describe the true class of the samples.
    bar3(conf_mtrx_KNN)
    legend(categories(conf_mtrx_KNN_label),'Location','EastOutside')
    ylabel('True Class')
    xlabel('Predicted Class')
    ax=gca;
    ax.XTickLabel=categories(conf_mtrx_KNN_label);
    ax.YTickLabel=categories(conf_mtrx_KNN_label);
    
    FiguresSup.f3=findobj('type','figure','Name','k-NN (k=10) Confusion Matrix on Red Wine - Image');
    if ishghandle(FiguresSup.f3)
        close(FiguresSup.f3)
    end
    FiguresSup.f3=figure('Name','k-NN (k=10) Confusion Matrix on Red Wine - Image');
    
    % An alternative display for the confusion matrix, as an image scaled 
    % to cover the full colormap specified by the user
    imagesc(conf_mtrx_KNN)
    % Display colour bar legend
    colorbar
    % Change colour map
    colormap parula
    
    ylabel('True Class')
    xlabel('Predicted Class')
    ax=gca;
    ax.XTick=1:5; ax.XTickLabel=categories(conf_mtrx_KNN_label);
    ax.YTick=1:5; ax.YTickLabel=categories(conf_mtrx_KNN_label);
    
    % Add labels at the center of the image
    [xpos,ypos] = meshgrid(1:size(conf_mtrx_KNN,1),1:size(conf_mtrx_KNN,2));
    text(xpos(:),ypos(:),num2str(conf_mtrx_KNN(:),'%d'),'HorizontalAlignment','Center')
    
    %% Section 3-d
    % Let us now run an optimisation over the hyperparameters of a KNN 
    % model by means of using k-fold validation with 10 folds over all the 
    % data available. In the case of brute force optimisation we will use 
    % hold-out validation with 30% test data and without reordering to 
    % improve computation time
    
    % For the lucky Matlab 2016b users
    if strcmp(version('-release'),'2016b')
        KNN_optml_hyp=fitcknn(Features_r_Norm,QCLabel_r,'KFold',10,...
            'OptimizeHyperparameters',{'Distance','DistanceWeight',...
            'NumNeighbors'},'HyperparameterOptimizationOptions)',...
            struct('AcquisitionFunctionName','expected-improvement-plus'),...
            'ShowPlots',true);
        % Here are two great references to learn about Gaussian Process
        % Optmisation of hyperparameters:
        % [https://arxiv.org/pdf/0912.3995.pdf]
        % [https://papers.nips.cc/paper/4522-practical-bayesian-optimization-of-machine-learning-algorithms.pdf]
        
    elseif Parallelswitch % Coding bayesian hyperparameter optimisation 
                          % could lead me to terrible mistakes so we will 
                          % use brute force optimisation with 
                          % parallelisation
        % Define the hyperparameters search hyperspace
        % Distance metrics:
        Dist_metrics={'seuclidean','hamming','spearman','minkowski',...
            'mahalanobis','chebychev','cityblock'};
        ind_Dist=1:length(Dist_metrics);
        % Number of neighbours
        ZNeighbours=[1:9 10:5:80];
        ind_Neighbours=1:length(ZNeighbours);
        % Distance Weights
        Dist_weights={'equal','inverse','squaredinverse'};
        ind_weights=1:length(Dist_weights);
        
        % Preallocate matrix of costs to retrieve results from different
        % workers
        Cost_Optm=zeros(ind_Neighbours(end),ind_weights(end),ind_Dist(end));
        
        % Start the parallel pool to ensure more predictable results.
        % Clear any if existent
        delete(gcp('nocreate'));
        % Start paraller pool
        poolobj=parpool('local',n_cores);
        
        % Loop in parallel over distance metrics
        parfor kk=ind_Dist
            
            dist_metric=Dist_metrics{kk};
            
            % Preallocate cost 3D array
            cost_optm=zeros(length(ZNeighbours),length(Dist_weights));
            
            for jj=ind_weights
                
                dist_weight=Dist_weights{jj};
                
                for ii=ind_Neighbours
                    
                    numneighbours_opt=ZNeighbours(ii);
                    % Train model
                    KNN_optml_hyp=fitcknn(TrainData_r,TrainLabel_r,...
                        'Distance',dist_metric,'DistanceWeight',dist_weight,...
                        'NumNeighbors',numneighbours_opt);
                    % Extract test error 
                    cost_optm(ii,jj)=KNN_optml_hyp.loss(TestData_r,TestLabel_r);
                    
                end
            end
            % Assemble the results 3D array
            Cost_Optm(:,:,kk)= cost_optm;
        end
        % Close pool
        delete(gcp('nocreate'))
        
        % Retrieve best result from 3D array and identify indexes using
        % ind2sub
        KNN_optml_bestcost=min(min(min(Cost_Optm,[],3),[],2));
        [ii,jj,kk]=ind2sub(size(Cost_Optm),find(Cost_Optm == KNN_optml_bestcost));
        
        disp({['Best Distance Matrix using 30% Hold Out for Red Wine was: ',Dist_metrics{kk}];
            ['With distance weights ',Dist_weights{jj}];
            ['And number of neighbours ',num2str(ZNeighbours(ii))];
            ['The test error for these settings is ',num2str(100* KNN_optml_bestcost,'%.4g'),...
            '% Misclassification']});
        
    end
    
    disp(' ')
    disp('Finished Chapter A ')
    
    user_option=input('Would you like to close the figures from this chapter? Y/N [N]:','s');
    if isempty(user_option)
        user_option='n';
    end
    if strcmpi('y',user_option)
              
        for vv=1:3
            figname=strcat('f',num2str(vv));
            if ishghandle(FiguresSup.(figname))&&strcmp(get(FiguresSup.(figname),'BeingDeleted'),'off')
                close(FiguresSup.(figname));
            end
        end
        
    end
    
    disp(' ')
    disp(['--------------------------------------------------------------------------------------------';...
        '------------------------------------     END     -------------------------------------------';...
        '--------------------------------------------------------------------------------------------'])
    fprintf('\nProgram paused. Press enter to continue.\n\n');
    pause;

    
end %KNNswitch


if NBswitch % {CHAPTER B}
    
    disp(' ')
    disp(['--------------------------------------------------------------------------------------------';...
        '------------------------------------  CHAPTER B  -------------------------------------------';...
        '--------------------------------------------------------------------------------------------'])
    
    %% Section 4-a
    % In this section we will perform Naive Bayes classification on the
    % data. This approach assumes the probability of observing features
    % given a class label is independent across features and hence
    % simplifies the calculation of the a posteriori label distribution.
    % Matlab allows the user to specify the distribution to be
    % considered for the predicition features given a class label. The
    % default most commonly used are Gaussian, Multinomial and
    % Multivariate Multinomial.
    
    % Let us now use the same partition over the red wine data (30%
    % hold-out) to fit a Naive Bayes model. The model will use a
    % multivariate multinomial assuming each feature follows a
    % multinomial model within each class. Notice that to use a Multinomial
    % distribution features are expected to be binary (typical setting for
    % NLP problems like spam detection or text clustering).
    
    % Recalculate CV partition if not done before
    if ~KNNswitch
        HoldOut_percent=0.3;
        partition=cvpartition(QCLabel_r,'HoldOut',HoldOut_percent);
        % Generate training and test sets
        TrainData_r=Features_r_Norm(partition.training,:);
        TrainLabel_r=QCLabel_r(partition.training);
        TestData_r=Features_r_Norm(partition.test,:);
        TestLabel_r=QCLabel_r(partition.test);
    end
    
    % The optional input 'Distribution' can be set to 'normal' (Gaussian
    % assumption), 'mn' (multinomial on observations), 'mvmn'
    % (multivariate multinomial on each feature of the observations) and
    % 'kernel' (Kernel density estimate for each class). When
    % 'Distribution' is set to 'mn', the input data must be nonnegative 
    % integers
    
    % The default option is 'normal' or a Gaussian fit for each feature.
    % An alternative way to think of this approach is an anomaly
    % detection algorithm with separated normal distributions for each 
    % feature in the observations. P(F1,F2...Fn|class) approximated as 
    % P(F1|class)·P(F2|class)·... ·P(Fn|class)
    
    % Specify the Distribution option, choose from 
    % {'normal','mvmn','mn','kernel'}
    NB_distribution='mvmn';
    
    % Ensure multinomial is not chosen unless positive integer features
    if strcmp(NB_distribution,'mn')&&(~isinteger(Features_r_Norm)||any(any(Features_r_Norm<0)))
        NB_distribution='mvmn';
    end
    
    % In order to be able to train a Multivariate Multinomial we will use a
    % trick on the data. We will divide all data by 0.01 and round the
    % results to make all features a set of integer values ranging from 0
    % to 100 and since we do not have any binary features, we will force 
    % the algorithm to treat all features as categorical, imagine instead
    % of having integers from 0 to 100 we had strings taking discrete
    % values '1', '2' ... '100' refering to the weight of a feature on a
    % sample chemical composition. 
    
    % Train the model
    NB_model_r=fitcnb(round(TrainData_r/0.01),TrainLabel_r,'Distribution',NB_distribution,...
        'CategoricalPredictors','all');
    
    % Predict on the validation set
    NB_pred=predict(NB_model_r,round(TestData_r/0.01));
    
    % Get train and test error
    NB_trainErr=100*resubLoss(NB_model_r);
    NB_validErr=100*loss(NB_model_r,round(TestData_r/0.01),TestLabel_r);
    
    disp({['Multivariate Multinomial Naive Bayes classifier',...
        ' Error:'];['On train set: ',num2str(NB_trainErr,'%.4g'),'%'];...
        ['On validation set: ',num2str(NB_validErr,'%.4g'),'%']});
    
    % Calculate the confusion matrix for this classifier
    [conf_mtrx_NB,conf_mtrx_NB_label]=confusionmat(TestLabel_r,NB_pred);
    
    FiguresSup.f4=findobj('type','figure','Name','MVMN Naive Bayes Confusion Matrix on Red Wine - Image');
    if ishghandle(FiguresSup.f4)
        close(FiguresSup.f4)
    end
    FiguresSup.f4=figure('Name','MVMN Naive Bayes Confusion Matrix on Red Wine - Image');
    
    % Display the confusion matrix as an image scaled to cover the full
    % colormap
    imagesc(conf_mtrx_NB)
    % Display colour bar legend
    colorbar
    % Change colour map
    colormap autumn
    
    ylabel('True Class')
    xlabel('Predicted Class')
    ax=gca;
    ax.XTick=1:5; ax.XTickLabel=categories(conf_mtrx_NB_label);
    ax.YTick=1:5; ax.YTickLabel=categories(conf_mtrx_NB_label);
    
    % Add labels at the center of the image
    [xpos,ypos] = meshgrid(1:size(conf_mtrx_NB,1),1:size(conf_mtrx_NB,2));
    text(xpos(:),ypos(:),num2str(conf_mtrx_NB(:),'%d'),'HorizontalAlignment','Center')
    
    %% Section 4-b
    % In order to perform NB analysis we need to make a strong
    % assumption on the features conditional dependency, which in
    % multiple ocassions proves to be worth due to the amount of
    % difficulty removed from the problem of estimating the posteriori
    % distribution. We can look at how good the data for each label
    % fits a normal distribution or a kernel. In particular, for a
    % normal distribution we can use the Jarque-Bera test or the
    % Lilliefors test.
    
    % First let us pick a random feature and generate a probability
    % density function histogram for each of the labels present in the
    % red wine data
    
    red_wine_labels=categories(QCLabel_r);
    num_red_labels=length(red_wine_labels);
    Rand_Feature_n=randi([1,length(Feature_Names)],1,'int8');
    
    FiguresSup.f5=findobj('type','figure','Name','Red Wine Feature pdf Histograms');
    if ishghandle(FiguresSup.f5)
        close(FiguresSup.f5)
    end
    FiguresSup.f5=figure('Name','Red Wine Feature pdf Histograms');
    hold on;
    
    % Plot for each label the histogram
    for ii=1:num_red_labels
        
        histogram(Features_r_Norm(QCLabel_r==red_wine_labels{ii},Rand_Feature_n),...
            'Normalization','pdf','FaceAlpha',0.4);
        xlabel(Feature_Names{Rand_Feature_n});
    end
    hold off;
    legend(red_wine_labels)
    
    % Now let us plot how good each feature for each label fits a normal
    % distribution. For this we can use the probplot function.
    % We will also take the opportunity to plot the kernel smoothing
    % density estimate. For the latter we will specify the Epanechnikov
    % kernel
    
    for ii=1:num_red_labels
        % Dynamically generate field name for struct to store the figures
        subfigure=strcat('subf_QC_',red_wine_labels{ii});
        FiguresSup.f6.(subfigure)=findobj('type','figure','Name',['Probability Plot (Gaussian) of Features for QCLabel ',...
            red_wine_labels{ii}]);
        if ishghandle(FiguresSup.f6.(subfigure))
            close(FiguresSup.f6.(subfigure))
        end
        FiguresSup.f6.(subfigure)=figure('Name',['Probability Plot (Gaussian) of Features for QCLabel ',...
            red_wine_labels{ii}]);
        % Within each label, for all the features...
        for jj=1:n_features
            subplot(ceil(n_features/2),2,jj);
            probplot(Features_r_Norm(QCLabel_r==red_wine_labels{ii},jj))
            title(Feature_Names{jj})
        end
        
        FiguresSup.f7.(subfigure)=findobj('type','figure','Name',['Kernel Smoothing Estimated pdf of Features for QCLabel ',...
            red_wine_labels{ii}]);
        if ishghandle(FiguresSup.f7.(subfigure))
            close(FiguresSup.f7.(subfigure))
        end
        FiguresSup.f7.(subfigure)=figure('Name',['Kernel Smoothing Estimated pdf of Features for QCLabel ',...
            red_wine_labels{ii}]);
        
        % Again, for all the features...
        for jj=1:n_features
            subplot(ceil(n_features/2),2,jj);
            ksdensity(Features_r_Norm(QCLabel_r==red_wine_labels{ii},jj),...
                'kernel','epanechnikov','function','pdf')
            title(Feature_Names{jj})
        end
    end
    
    % Now let us test the the quality of a Gaussian fit using the
    % Jarque-Bera test for each feature under each label. The JB test
    % returs the values h and p ([h,p]=jbtest(data)). Where h is a
    % binary indicator of a good fit being 0 if the data is normal. p
    % represents the probability of observing a test statistic as
    % extreme as, or more extreme than, the observed value under
    % hypothesis that the data fits a normal distribution.
    
    % A small value of p, below a typical significance level of 0.05,
    % indicates a BAD fit. THE INVERSE DOES NOT APPLY AS GUARANTEE OF
    % NORMALITY, hence high p values cannot be taken as indicators of
    % normality.
    
    % Preallocate space for p and h for each feature within each QC label.
    h=ones(num_red_labels,n_features);
    p=h;
    
    for ii=1:num_red_labels
        for jj=1:n_features
            
            [h(ii,jj),p(ii,jj)]=jbtest(Features_r_Norm(QCLabel_r==red_wine_labels{ii},jj));
        end
        % we add another loop to avoid the inclusion of warning messages
        % from jbtest if the significance level is below minimum
        disp(['For QC Label ',red_wine_labels{ii},...
            ' the following features are POOR Gaussian fits',...
            ' according to the Jarque-Bera Test']);
        for jj=1:n_features
            if p(ii,jj)<0.05
                disp(['  -> ',Feature_Names{jj}])
            end
        end
        disp('---------------------------------------------------')
    end
    
    % Let us now add some labeling to the JB Test results and display
    % these.
    disp('Table indicating good Gaussian fits (0 Indicates GOOD FIT)');
    h=array2table(h);
    % Each column corresponds to a feature and we need to label the table.
    h.Properties.VariableNames=Feature_Names;
    % Add the red wine labels as a legend to the rows. 
    h=[table(red_wine_labels),h];
    disp(h);
    
    disp(' ')
    disp('Finished Chapter B')
    
    user_option=input('Would you like to close the figures from this chapter? Y/N [N]:','s');
    if isempty(user_option)
        user_option='n';
    end
    if strcmpi('y',user_option)
        
        for vv=4:7
            figname=strcat('f',num2str(vv));
            if vv==6||vv==7
                for ii=1:num_red_labels
                    % Dynamically generate field name for struct to store the figures
                    subfigure=strcat('subf_QC_',red_wine_labels{ii});
                    
                    if ishghandle(FiguresSup.(figname).(subfigure))&&strcmp(get(FiguresSup.(figname).(subfigure),'BeingDeleted'),'off')
                        close(FiguresSup.(figname).(subfigure));
                    end
                end
            else
                if ishghandle(FiguresSup.(figname))&&strcmp(get(FiguresSup.(figname),'BeingDeleted'),'off')
                    close(FiguresSup.(figname));
                end
            end
        end
        
    end
    
    disp(' ')
    disp(['--------------------------------------------------------------------------------------------';...
        '------------------------------------     END     -------------------------------------------';...
        '--------------------------------------------------------------------------------------------'])
    fprintf('\nProgram paused. Press enter to continue.\n\n');
    pause;

    
end %if NBswitch

if GDswitch % {CHAPTER C}
    
    disp(' ')
    disp(['--------------------------------------------------------------------------------------------';...
        '------------------------------------  CHAPTER C  -------------------------------------------';...
        '--------------------------------------------------------------------------------------------'])
    %% Section 5-a
    % In this chapter we will perform Gaussian Discriminant Analysis on the
    % red wine data in an attempt to predict the quality label. During the
    % training phase, GDA estimates the parameters of a multivariate 
    % Gaussian distribution for each class. After doing this, a posteriori
    % probabilities of the class labels given the observations are 
    % calculated and a class is chosen to minimise the expected 
    % classification cost.
    
    % Matlab allows the user to specify different assumptions over the
    % parameters of the Gaussians created for each class. This option is
    % provided via the input 'discrimType' and can be specified as:
    
    % linear - quadratic - diagLinear - diagQuadratic - pseudoLinear -
    % pseudoQuadratic
    
    % In the presence of singular covariance matrices this method may fail
    % and hence the following options are preferred:
    
    % diagLinear - Diagonal version, same covariance assumed for all 
    % classes, which causes linear classification borders
    
    % diagQuadratic - Diagonal of quadratic, each class has different
    % covariance and these are fully populated
    
    % pseudoLinear / pseudoQuadratic - same as above but using the
    % pseudoinverse of the covariance matrix
    
    % We will train a discriminant model using the pseudoQuadratic
    % covariance approach, specifying a uniform prior distribution for
    % classes (could specify empirical to use the frequency of each class 
    % on the observations provided). We will activate cross validation with
    % 5 folds.
    
    GD_model_r=fitcdiscr(Features_r_Norm,QCLabel_r,'DiscrimType',...
        'pseudoQuadratic','prior','uniform','CrossVal','on',...
        'KFold',5);
    
    % Predict on all the data. 
    GD_pred=GD_model_r.kfoldPredict;
    
    % Get train average error and test error for all the samples (all are 
    % used once for validation). The user can specify whether to average
    % train error across the folds or to take a different approach like the
    % worst performance. 
    
    GD_trainErr=100*(GD_model_r.kfoldLoss);
    GD_validErr=100*sum(GD_pred==QCLabel_r)/(length(GD_pred));
    
    disp({['Gaussian Discriminant PseudoQuadratic Covariance',...
        ' Error:'];['Average from 5 folds on train set: ',num2str(GD_trainErr,'%.4g'),'%'];...
        ['Validation: ',num2str(GD_validErr,'%.4g'),'%']});
    
    % Calculate the confusion matrix for this classifier
    [conf_mtrx_GD,conf_mtrx_GD_label]=confusionmat(QCLabel_r,GD_pred);
    
    FiguresSup.f8=findobj('type','figure','Name','Gaussian Discriminant Confusion Matrix on Red Wine - Image');
    if ishghandle(FiguresSup.f8)
        close(FiguresSup.f8)
    end
    FiguresSup.f8=figure('Name','Gaussian Discriminant Confusion Matrix on Red Wine - Image');
    % Display the confusion matrix as an image scaled to cover the full
    % colormap
    imagesc(conf_mtrx_GD)
    % Display colour bar
    colorbar
    % Change colour map
    colormap summer
    
    ylabel('True Class')
    xlabel('Predicted Class')
    ax=gca;
    ax.XTick=1:5; ax.XTickLabel=categories(conf_mtrx_GD_label);
    ax.YTick=1:5; ax.YTickLabel=categories(conf_mtrx_GD_label);
    
    % Add labels at the center of the image
    [xpos,ypos] = meshgrid(1:size(conf_mtrx_GD,1),1:size(conf_mtrx_GD,2));
    text(xpos(:),ypos(:),num2str(conf_mtrx_GD(:),'%d'),'HorizontalAlignment','Center')
    
    
    %% Section 5-b
    
    % Gaussian Discriminant Analysis
    % [http://cs229.stanford.edu/notes/cs229-notes2.pdf] assumes normal
    % distribution for the feature samples within each class. This imposes
    % a strong assumption over the data and hence it is always good 
    % practice to verify how good a fit for the feature data a Gaussian is;
    % but it is also important to understand if impossing a linear 
    % classification edge is reasonable. In other words, how do covariance 
    % matrices change accross classes if the multivariate Gaussian is a
    % valid assumption.
    
    % For the latter test it is possible to use the vartestn function in
    % Matlab. Here the user can specify the type of test in order to use
    % approaches that are less sensitive to deviations from Gaussian
    % distributions than the Barlett's test. For this function, a low
    % output p value (p below threshold 0.05 typically) indicates that the
    % hypothesis of similar covariances should be rejected.
    
    % Firstly, let us extract the covariance of the features across the red
    % wine samples.
    
    Cov_r_Norm=cov(Features_r_Norm);
    
    FiguresSup.f9=findobj('type','figure','Name','Standardised Covariance Features on Red Wine');
    if ishghandle(FiguresSup.f9)
        close(FiguresSup.f9)
    end
    FiguresSup.f9=figure('Name','Standardised Covariance Features on Red Wine');
    
    % Let us now plot a 3D bar plot of the covariance matrix. If the non
    % diagonal elements follow a uniform distribution it could indicate
    % suitability of the linear discriminant assumption (similar
    % covariances).
    bar3(Cov_r_Norm)
    colormap winter
    
    ax=gca;
    ax.YTickLabel=Feature_Names; ax.XTickLabel=Feature_Names;
    zlabel('Standardised Covariance');
    
    % Now let us perform the Brown-Forsythe test to check equal covariances
    % across the features and for each feature vs QC labels as well.
    
    [p_BFTest_r,stats_BFTest_r]=vartestn(Features_r_Norm,...
        'TestType','BrownForsythe','Display','off');
    
    disp(['--------------------------------------------------------------------';'']);
    if p_BFTest_r<0.05
        disp('Assumption across red wine features of same covariance rejected');
    else
        disp('Assumption across red wine features of same covariance accepted');
    end
    
    
    
    % Now for each feature vs quality labels
    
    % Preallocate for the test result and labels for the plot. We will use
    % the test results to check for hypothesis rejection and display an
    % interrogation mark in the case of hypothesis acceptance as we cannot
    % assume high values of p indicate similar covariance.
    
    Features_p_BFTest_r=zeros(n_features,1);
    Features_p_BFTest_Labels={};
    
    for jj=n_features:-1:1
        Features_p_BFTest_r(jj)=vartestn(Features_r_Norm(:,jj),QCLabel_r);
        % If below probability threshold, rejection
        if Features_p_BFTest_r(jj)<0.05
            Features_p_BFTest_Labels{jj}='N';
        else % Further checks required...
            Features_p_BFTest_Labels{jj}='?';
        end
    end
    
    FiguresSup.f10=findobj('type','figure','Name','Variance Test Features on Red Wine vs QCLabel');
    if ishghandle(FiguresSup.f10)
        close(FiguresSup.f10)
    end
    FiguresSup.f10=figure('Name','Variance Test Features on Red Wine vs QCLabel');
    % Plot the results and add a label to explain if the hypothesis of same
    % covariance has been accepted.
    scatter(1:n_features,Features_p_BFTest_r,'filled')
    grid on;
    ax=gca;
    ax.XTickLabel=Feature_Names;
    ylabel('p value vartestn against QC Label - Red Wine')
    title('Same Cov as Classes?')
    
    % Add labels at the center of the image
    [xpos,ypos] = meshgrid(1:n_features,mean(Features_p_BFTest_r));
    text(xpos(:),ypos(:),Features_p_BFTest_Labels,'HorizontalAlignment','Center')
    
    disp(' ')
    disp('Finished Chapter C')
    
    user_option=input('Would you like to close the figures from this chapter? Y/N [N]:','s');
    if isempty(user_option)
        user_option='n';
    end
    if strcmpi('y',user_option)
              
        for vv=8:10
            figname=strcat('f',num2str(vv));
            if ishghandle(FiguresSup.(figname))&&strcmp(get(FiguresSup.(figname),'BeingDeleted'),'off')
                close(FiguresSup.(figname));
            end
        end
        
        h_boxplot=findobj('type','figure','Tag','boxplot');
        for ii=1:length(h_boxplot)
            if ishghandle(h_boxplot(ii))&&strcmp(get(h_boxplot(ii),'BeingDeleted'),'off')
                close(h_boxplot(ii));
            end
        end
        
        h_table=findobj('type','figure','Tag','table');
        for ii=1:length(h_table)
            if ishghandle(h_table(ii))&&strcmp(get(h_table(ii),'BeingDeleted'),'off')
                close(h_table(ii));
            end
        end
        
    end
    
    disp(' ')
    disp(['--------------------------------------------------------------------------------------------';...
        '------------------------------------     END     -------------------------------------------';...
        '--------------------------------------------------------------------------------------------'])
    fprintf('\nProgram paused. Press enter to continue.\n\n');
    pause;

    
end %if GDswitch


if DTswitch % {CHAPTER D}
    
    disp(' ')
    disp(['--------------------------------------------------------------------------------------------';...
        '------------------------------------  CHAPTER D  -------------------------------------------';...
        '--------------------------------------------------------------------------------------------'])
    
    %% Section 6-a
    % In this section we will perform classification based on Decision
    % Trees. These are predictors that make a categorical choice on a
    % sample based on binary decisions on the values of its features.
    
    % Decision trees are prone to overfitting and hence it is crucial
    % to ensure a correct cross validation strategy to understand the
    % generalisation error.
    
    % First, let us generate a 70/30 cross validation structure over the
    % red wine dataset if it has not been done before.
    if ~(KNNswitch||NBswitch)
        HoldOut_percent=0.3;
        partition=cvpartition(QCLabel_r,'HoldOut',HoldOut_percent);
        % Generate training and test sets
        TrainData_r=Features_r_Norm(partition.training,:);
        TrainLabel_r=QCLabel_r(partition.training);
        TestData_r=Features_r_Norm(partition.test,:);
        TestLabel_r=QCLabel_r(partition.test);
    end
    
    % Now we will fit the Decision Tree model to the train data
    DT_model_r=fitctree(TrainData_r,TrainLabel_r);
    
    % And visualise the result using the view method and requesting a graph
    % display
    view(DT_model_r,'Mode','graph');
    
    % Calculate training and validation error for the decision tree
    % classifier
    DT_trainErr=100*resubLoss(DT_model_r);
    DT_validErr=100*loss(DT_model_r,TestData_r,TestLabel_r);
    
    disp({['Decision Tree classifier',...
        ' Error:'];['On train set: ',num2str(DT_trainErr,'%.4g'),'%'];...
        ['On validation set: ',num2str(DT_validErr,'%.4g'),'%']});
    
    % Calculate predictions on test data
    DT_pred=predict(DT_model_r,TestData_r);
    
    % And compute the confusion matrix for this classifier
    [conf_mtrx_DT,conf_mtrx_DT_label]=confusionmat(TestLabel_r,DT_pred);
    
    FiguresSup.f11=findobj('type','figure','Name','Decision Tree Confusion Matrix on Red Wine - Image');
    if ishghandle(FiguresSup.f11)
        close(FiguresSup.f11)
    end
    FiguresSup.f11=figure('Name','Decision Tree Confusion Matrix on Red Wine - Image');
    % Display the confusion matrix as an image scaled to cover the full
    % colormap
    imagesc(conf_mtrx_DT)
    % Add a colour bar
    colorbar
    % Change the colour map
    colormap cool
    ylabel('True Class')
    xlabel('Predicted Class')
    ax=gca;
    ax.XTick=1:5; ax.XTickLabel=categories(conf_mtrx_DT_label);
    ax.YTick=1:5; ax.YTickLabel=categories(conf_mtrx_DT_label);
    
    % Add labels at the center of the image to represent the values on the
    % confusion matrix
    [xpos,ypos] = meshgrid(1:size(conf_mtrx_DT,1),1:size(conf_mtrx_DT,2));
    text(xpos(:),ypos(:),num2str(conf_mtrx_DT(:),'%d'),'HorizontalAlignment','Center')
    
    disp(' ')
    disp('Finished Chapter D')
    
    user_option=input('Would you like to close the figures from this chapter? Y/N [N]:','s');
    if isempty(user_option)
        user_option='n';
    end
    if strcmpi('y',user_option)
              
        for vv=11:11
            figname=strcat('f',num2str(vv));
            if ishghandle(FiguresSup.(figname))&&strcmp(get(FiguresSup.(figname),'BeingDeleted'),'off')
                close(FiguresSup.(figname));
            end
        end
        
    end
    
    disp(' ')
    disp(['--------------------------------------------------------------------------------------------';...
        '------------------------------------     END     -------------------------------------------';...
        '--------------------------------------------------------------------------------------------'])
    fprintf('\nProgram paused. Press enter to continue.\n\n');
    pause;


end %if DTswitch

if SVMswitch % {CHAPTER E}
    
    disp(' ')
    disp(['--------------------------------------------------------------------------------------------';...
        '------------------------------------  CHAPTER E  -------------------------------------------';...
        '--------------------------------------------------------------------------------------------'])
    
    %% Section 7-a
    % In this section we will train a binary Support Vector Machine
    % classifier. This algorithm finds the maximum margin hyperplane that
    % separates two classes. The support vectors are the closest points
    % to the hyperplane and the only information needed to predict the
    % label on a new sample. SVMs are amongst the most popular
    % classifiers as they tend to perform well on generalisation.
    % However, outliers can deeply influyence the quality of the
    % classification hyperplane, hence soft-margin regularisation can be
    % applied to ignore outliers in the interest of minimising
    % classification error. For this option we can set the penalty
    % factor using a positive integer as an additional input on the
    % option Boxconstraint. The denomination box constraint comes from the
    % fact that solving a SVM problem is typically performed using its dual
    % formulation and in the latter, a cost on misclassification becomes a
    % maximum constrain for the Lagrange multiplier assigned to such
    % constraint, hence it limits it to be inside a positive box. 
    % [http://www.robots.ox.ac.uk/~az/lectures/ml/lect3.pdf]
    
    % It is always recommended to standardise features before performing
    % SVM classification.
    
    % For multiclass classification using SVM, it is frequent to use a
    % one vs all approach and train a total of N classifiers where N is the
    % number of labels available on the data. For this problem we will use
    % the Error Correcting Output Codes approach. See:
    % [http://www.jair.org/media/105/live-105-1426-jair.pdf]
    
    % Let us begin with binary classification. First we will extract the 
    % red wine data from labels C or D only (dominant populations)
    
    Features_r_CD_Norm=Features_r_Norm(QCLabel_r=='C'|QCLabel_r=='D',:);
    QCLabel_r_CD=removecats(QCLabel_r(QCLabel_r=='C'|QCLabel_r=='D'),...
        {'A','B','E'});
    
    % We will be using K-fold cross validation to test the influence of
    % the following parameters on a SVM:
    % > Linear SVM vs Gaussian Kernel vs Polynomial Kernel, all Kernels
    % with auto scaling. 
    Kernels={'none','rbf','linear','polynomial'};
    Kernel_Poly_order=(2:4);
    % > Soft-margin box constraint on the Lagrange multipliers. Penalty
    % on misclassified elements
    BoxCons=[0.1 1 10 20:20:80];
    % > Prior distribution: Uniform vs Empirical (use a uniform label prior
    % distribution or the frequency in which these appear in the training
    % data
    Prior={'empirical','uniform'};
    % We will use the SMO solver for all the analyses (when possible)
    % [http://luthuli.cs.uiuc.edu/~daf/courses/optimization/Papers/smoTR.pdf]
    % and the default scoring mechanism to label misclassified data
    % [1 if correct, -1 if incorrect]
    
    % In this section we will once again use brute force minima search, the
    % usage of Bayesian optimisation or alternative optimisation methods 
    % like random search might improve the ratio between performance gain 
    % and number of function evaluations (understood here as the training 
    % of a SVM model per candidate to be studied)
    
    
    % Preallocate the results to store cross validation error (K-Folds) and
    % a brief description on the settings used. 
    num_SVM_trained=(length(Kernels)-1+...
        length(Kernel_Poly_order))*length(BoxCons)*length(Prior);
    for jj= num_SVM_trained:-1:1
        SVM_Results(jj)=struct('CVErr',[],'Desc','');
    end
    % Initialise variable to temporarily hold analysis description
    Description='';
    
    % Placeholder to store best
    SVM_BestErr=1; %Initialised to worst case 100% error
    SVM_BestDesc={'Empty'};
    
    % Create an index to track the number of SVM trained
    ind_SVM=1;
    
    for ind1=1:length(Kernels)
        
        for BCons=BoxCons
            
            for ind2=1:length(Prior)
                
                switch Kernels{ind1}
                    
                    case 'none'
                        
                        % Train a linear SVM
                        SVM_model_r_CD=fitcsvm(Features_r_CD_Norm,...
                            QCLabel_r_CD,'BoxConstraint',BCons,...
                            'Prior',Prior{ind2},...
                            'KFold',10,'Solver','SMO');
                        % Store cross validation error
                        SVM_Results(ind_SVM).CVErr=SVM_model_r_CD.kfoldLoss;
                        Description=['SVM 10KFold - Kernel ','None',...
                            ' - BoxConstraint ',num2str(BCons),...
                            ' - Prior ',Prior{ind2},' - Solver ','SOM'];
                        % Store description as a legend to understand
                        % the performance of each configuration.
                        SVM_Results(ind_SVM).Desc=Description;
                        
                    case 'rbf'
                        % Repeat as above but instead this time we will
                        % train a Gaussian Kernel SVM
                        SVM_model_r_CD=fitcsvm(Features_r_CD_Norm,...
                            QCLabel_r_CD,'BoxConstraint',BCons,...
                            'Prior',Prior{ind2},...
                            'KernelFunction','rbf',...
                            'KFold',10,'Solver','SMO');
                        SVM_Results(ind_SVM).CVErr=SVM_model_r_CD.kfoldLoss;
                        Description=['SVM 10KFold - Kernel ','Gaussian',...
                            ' - BoxConstraint ',num2str(BCons),...
                            ' - Prior ',Prior{ind2},' - Solver ','SOM'];
                        SVM_Results(ind_SVM).Desc=Description;
                        
                    case 'linear'
                        % Repeat as above but instead this time we will
                        % train a Linear Kernel SVM
                        SVM_model_r_CD=fitcsvm(Features_r_CD_Norm,...
                            QCLabel_r_CD,'BoxConstraint',BCons,...
                            'Prior',Prior{ind2},...
                            'KernelFunction','linear',...
                            'KFold',10,'Solver','SMO');
                        SVM_Results(ind_SVM).CVErr=SVM_model_r_CD.kfoldLoss;
                        Description=['SVM 10KFold - Kernel ','Linear',...
                            ' - BoxConstraint ',num2str(BCons),...
                            ' - Prior ',Prior{ind2},' - Solver ','SOM'];
                        SVM_Results(ind_SVM).Desc=Description;
                        
                    case 'polynomial'
                        
                        for poly_order=Kernel_Poly_order
                            % Repeat as above but instead this time we will
                            % train a Polynomial Kernel SVM
                            SVM_model_r_CD=fitcsvm(Features_r_CD_Norm,...
                                QCLabel_r_CD,'BoxConstraint',BCons,...
                                'Prior',Prior{ind2},...
                                'KernelFunction','polynomial',...
                                'PolynomialOrder',poly_order,...
                                'KFold',10,'Solver','SMO');
                            SVM_Results(ind_SVM).CVErr=SVM_model_r_CD.kfoldLoss;
                            Description=['SVM 10KFold - Kernel ','Polynomial of order ',...
                                num2str(poly_order),' - BoxConstraint ',num2str(BCons),...
                                ' - Prior ',Prior{ind2},' - Solver ','SOM'];
                            SVM_Results(ind_SVM).Desc=Description;
                            
                            % Within the polynomial loop we need to check
                            % for best error improvement as there is a for
                            % loop along the Kernel order values
                            
                            if SVM_model_r_CD.kfoldLoss<SVM_BestErr
                                SVM_BestErr=SVM_model_r_CD.kfoldLoss;
                                SVM_BestDesc=SVM_Results(ind_SVM).Desc;
                            end
                            % And increase the number of evaluations
                            % counter
                            ind_SVM=ind_SVM+1;
                            
                            % Update the user on status 20% of time
                            if (randi([1,5],1,'int8')==1)
                            disp('Completed SVM training for following parameters: ')
                            disp(Description);
                            end
                            
                        end
                    otherwise
                        
                end
                
                % Only for the case we did not check already an improvement
                % over the best
                if ~strcmp(Kernels{ind1},'polynomial')
                    if SVM_model_r_CD.kfoldLoss<SVM_BestErr
                        SVM_BestErr=SVM_model_r_CD.kfoldLoss;
                        SVM_BestDesc=SVM_Results(ind_SVM).Desc;
                    end
                    % And increase the number of evaluations
                    % counter
                    ind_SVM=ind_SVM+1;
                    
                    % Update the user on status 20% of time
                    if (randi([1,5],1,'int8')==1)
                    disp('Completed SVM training for following parameters: ')
                    disp(Description);
                    end
                end
                              
            end
            
        end
    end
    
    % Report best result to the user.
    disp(['Best SVM performance on classification of C and D QC Labels',...
        'from Red Wine according to 10 K-Fold cross validation was:'])
    disp([num2str(100*SVM_BestErr,'%.4g'), '%']);
    disp('Using the configuration:');
    disp(SVM_BestDesc);
    
    
    %% Section 7-b
    % We will now perform multiclass classification with SVM. As we have
    % mentioned before, it is possible to train in a one vs all approach 
    % but it might also be desirable to train a one vs one by grouping the 
    % labels in different couples and training N(N-1)/2 classifiers, where
    % N is the number of labels available in the train data.
    
    % In order to adapt binary classifiers for a multiclass problem in 
    % Matlab we need to use the Error-Correcting Output Codes classifier. 
    % This approach requires the definition of a template classifier (let 
    % it be SVM or KNN or similar) as well as a coding matrix that 
    % specifies which classes are used for individual classifier training. 
    % For example, let us assume we have 3 classes A, B & C and we want to 
    % train a one vs one algorithm at a time. Then when we use fitcecoc
    % method we need to provide the following coding matrix (note how we
    % train as many learners as classes by default but not compulsory as
    % the existence of numerous classes might call for time efficiency):
    
    %              Learner 1    Learner 2    Learner 3
    %              -----------------------------------
    %      Class A     1      |      1     |     0    |
    %      Class B    -1      |      0     |     1    |
    %      Class C     0      |     -1     |    -1    |
    %              ------------------------------------
    %
    
    % In the case of 1 vs all we would need to specify the following matrix:
    
    %              Learner 1    Learner 2    Learner 3
    %              -----------------------------------
    %      Class A     1      |     -1     |    -1    |
    %      Class B    -1      |      1     |    -1    |
    %      Class C    -1      |     -1     |     1    |
    %              ------------------------------------
    %
    
    % The rows of these code matrices can be read as binary words. In order
    % to predict the class of a new element all learners are evaluated and
    % the class whose binary word is closest to the binary word generated 
    % for the new sample is assigned to it. Note how the coding matrix must
    % have a full rank so that two rows are never equal and redundant
    % learners are not trained. 
    
    % Let us now try this approach using the red wine data. First we will
    % define the partition if it has not been done before.
    
    if ~(KNNswitch||NBswitch||DTswitch)
        HoldOut_percent=0.3;
        partition=cvpartition(QCLabel_r,'HoldOut',HoldOut_percent);
        % Generate training and test sets
        TrainData_r=Features_r_Norm(partition.training,:);
        TrainLabel_r=QCLabel_r(partition.training);
        TestData_r=Features_r_Norm(partition.test,:);
        TestLabel_r=QCLabel_r(partition.test);
    end
    
    % We will create an SVM template using fixed settings
    
    Template_Learner=templateSVM('KernelFunction','polynomial',...
        'PolynomialOrder',4,'BoxConstraint',60,'Prior','uniform',...
        'Solver','SMO');
    
    % Fit an error correcting output codes classifier to the training set
    % using a one vs one coding matrix and a one vs all. Use 
    % parallelisation if available.
    
    if Parallelswitch
        
        % Set options to use parallelisation
        ecoc_options = statset('UseParallel',1);
        % Start the parallel pool to ensure more predictable results
        delete(gcp('nocreate'));
        poolobj=parpool('local',n_cores);
        
        % Train 1vs1 and 1vsAll models. Notice how the input field 'Coding'
        % to the function fitcecoc specifies the coding matrix. The string 
        % options predefined are:
        % 'onevsall' (default) | 'allpairs' | 'binarycomplete' | 
        % 'denserandom' | 'onevsone' | 'ordinal' | 'sparserandom' | 
        % 'ternarycomplete' |
        % Alternatively, a numeric coding matrix can be entered manually. 
        
        ECOC_1vs1_model_r=fitcecoc(TrainData_r,TrainLabel_r,'Learners',...
            Template_Learner,'Coding','onevsone','Options',ecoc_options);
        ECOC_1vsAll_model_r=fitcecoc(TrainData_r,TrainLabel_r,'Learners',...
            Template_Learner,'Coding','onevsall','Options',ecoc_options);
        
        % Close parallel pool
        delete(gcp('nocreate'));
    else
        % Train models without using parallelisation
        ECOC_1vs1_model_r=fitcecoc(TrainData_r,TrainLabel_r,'Learners',...
            Template_Learner,'Coding','onevsone');
        ECOC_1vsAll_model_r=fitcecoc(TrainData_r,TrainLabel_r,'Learners',...
            Template_Learner,'Coding','onevsall');
    end
    
    % Predict on the validation set and calculate errors for training and
    % validation.
    
    ECOC_1vs1_trainErr=100*resubLoss(ECOC_1vs1_model_r);
    ECOC_1vs1_validErr=100*loss(ECOC_1vs1_model_r,TestData_r,TestLabel_r);
    
    ECOC_1vsAll_trainErr=100*resubLoss(ECOC_1vsAll_model_r);
    ECOC_1vsAll_validErr=100*loss(ECOC_1vsAll_model_r,TestData_r,TestLabel_r);
    
    disp({['Soft-Margin SVM 1 vs 1 ECOC classifier using Polynomial Kernel of order 4',...
        ' Error:'];['On train set: ',num2str( ECOC_1vs1_trainErr,'%.4g'),'%'];...
        ['On validation set: ',num2str( ECOC_1vs1_validErr,'%.4g'),'%']});
    
    disp({['Soft-Margin SVM 1 vs All ECOC classifier using Polynomial Kernel of order 4',...
        ' Error:'];['On train set: ',num2str( ECOC_1vsAll_trainErr,'%.4g'),'%'];...
        ['On validation set: ',num2str( ECOC_1vsAll_validErr,'%.4g'),'%']});
    
    % Calculate predictions on test data
     ECOC_1vs1_pred=predict( ECOC_1vs1_model_r,TestData_r);
    
     ECOC_1vsAll_pred=predict( ECOC_1vsAll_model_r,TestData_r);
     
    % Calculate the confusion matrix for these classifiers
    [conf_mtrx_ECOC_1vs1,conf_mtrx_ECOC_1vs1_label]=confusionmat(TestLabel_r, ECOC_1vs1_pred);
    
    [conf_mtrx_ECOC_1vsAll,conf_mtrx_ECOC_1vsAll_label]=confusionmat(TestLabel_r, ECOC_1vsAll_pred);
    
    FiguresSup.f12=findobj('type','figure','Name','ECOC SVM 1 vs 1 Confusion Matrix on Red Wine - Image');
    if ishghandle(FiguresSup.f12)
        close(FiguresSup.f12)
    end
    FiguresSup.f12=figure('Name','ECOC SVM 1 vs 1 Confusion Matrix on Red Wine - Image');
    % Display the confusion matrix as an image scaled to cover the full
    % colormap
    imagesc(conf_mtrx_ECOC_1vs1)
    % Display colour bar
    colorbar
    % Change colour map
    colormap parula
    ylabel('True Class')
    xlabel('Predicted Class')
    ax=gca;
    ax.XTick=1:5; ax.XTickLabel=categories(conf_mtrx_ECOC_1vs1_label);
    ax.YTick=1:5; ax.YTickLabel=categories(conf_mtrx_ECOC_1vs1_label);
    
    % Add labels at the center of the image
    [xpos,ypos] = meshgrid(1:size(conf_mtrx_ECOC_1vsAll,1),1:size(conf_mtrx_ECOC_1vsAll,2));
    text(xpos(:),ypos(:),num2str(conf_mtrx_ECOC_1vsAll(:),'%d'),'HorizontalAlignment','Center')
    
    FiguresSup.f13=findobj('type','figure','Name','ECOC SVM 1 vs All Confusion Matrix on Red Wine - Image');
    if ishghandle(FiguresSup.f13)
        close(FiguresSup.f13)
    end
    FiguresSup.f13=figure('Name','ECOC SVM 1 vs All Confusion Matrix on Red Wine - Image');
    % Display the confusion matrix as an image scaled to cover the full
    % colormap
    imagesc(conf_mtrx_ECOC_1vsAll)
    % Display colour bar
    colorbar
    % Change colour map
    colormap spring
    ylabel('True Class')
    xlabel('Predicted Class')
    ax=gca;
    ax.XTick=1:5; ax.XTickLabel=categories(conf_mtrx_ECOC_1vsAll_label);
    ax.YTick=1:5; ax.YTickLabel=categories(conf_mtrx_ECOC_1vsAll_label);
    
    % Add labels at the center of the image
    [xpos,ypos] = meshgrid(1:size(conf_mtrx_ECOC_1vsAll,1),1:size(conf_mtrx_ECOC_1vsAll,2));
    text(xpos(:),ypos(:),num2str(conf_mtrx_ECOC_1vsAll(:),'%d'),'HorizontalAlignment','Center')
    
    disp(' ')
    disp('Finished Chapter E')
    
    user_option=input('Would you like to close the figures from this chapter? Y/N [N]:','s');
    if isempty(user_option)
        user_option='n';
    end
    if strcmpi('y',user_option)
              
        for vv=12:13
            figname=strcat('f',num2str(vv));
            if ishghandle(FiguresSup.(figname))&&strcmp(get(FiguresSup.(figname),'BeingDeleted'),'off')
                close(FiguresSup.(figname));
            end
        end
        
    end
    
    disp(' ')
    disp(['--------------------------------------------------------------------------------------------';...
        '------------------------------------     END     -------------------------------------------';...
        '--------------------------------------------------------------------------------------------'])
    fprintf('\nProgram paused. Press enter to continue.\n\n');
    pause;

    
end %if SVMswitch


if NNswitch % {CHAPTER F}
    
    disp(' ')
    disp(['--------------------------------------------------------------------------------------------';...
        '------------------------------------  CHAPTER F  -------------------------------------------';...
        '--------------------------------------------------------------------------------------------'])
    
    %% Section 8-a 
    % In this section we will briefly discuss Neural Networks, a wide field
    % that currently receives great attention within research. Neural
    % Networks are non linear classifiers inspired by the human brain and
    % composed by a set of interconnected nodes that continuously adapt to
    % new training information. Neural Networks have shown great
    % performance on complex problems such as image classification or
    % biometric recognition, see:
    % [https://papers.nips.cc/paper/4136-tiled-convolutional-neural-networks.pdf]
    
    % The simplest version of a NN is composed by an input layer, one or
    % more hidden layers and an output layer, all of these fully
    % interconnected. The input to the neural network are
    % the features of each training sample and feed-forward is used to
    % produce a prediction on the label of each sample. Thus, the hidden
    % layer values are the result of applying an activation function
    % [Sigmoid, tanh, ReLU, ELU, arctan, PReLU..
    % (https://en.wikipedia.org/wiki/Activation_function)] to set of linear
    % combinations of the values in the previous layer (using coefficients
    % denominated weights) plus scalar offsets called biases. The latter 
    % are excluded from any regularisation strategy as it would be counter
    % intuitive to add a penalisation on the modification of a bias.
    
    % Recent research results have proved that a great portion of the
    % success of a Neural Network can be attibuted to its architecture. 
    % Hence much research on Neural Networks focuses on hyperparameters
    % optimisation. Any NN can be parameterised by multiple choices as:
    % > Size of input layer (Downsample data? How deep auto regression?)
    % > Size of the output layer (Single node per class? Softmax layer?)
    % > Number of hidden layers
    % > Number of nodes per hidden layer
    % > Level of interconnection between layers. (Fully connected? Drop-out
    % during training?)
    % > Activation function
    % > Activation threshold for binary layers
    
    % In the case of convolutional neural networks 
    % [http://neuralnetworksanddeeplearning.com/index.html], there is an
    % additional need to define:
    % > Size of local receptive fields
    % > Number of local receptive fields
    % > Tiling of local receptive fields (Common weights? Tiled convnets?)
    % > Number of pooling units
    % > Pooling strategy
    % > Drop-out implementation if desired
    % > ...
    
    % All the complexity of NN comes with drawbacks. The incorrect
    % selection of the architecture can lead to overfitting, a poor choice
    % on the activation function might cause vanishing gradient (once the
    % latests layers have converged during training, their low gradient is
    % back propagated and prevents the earliest layers from training
    % efficiently) or it might be difficult to identify simpler
    % architectures with similar or equal performance that would mean a
    % reduction in both training and evaluation times. 
    
    % For all of these reasons, topics like Bayesian Optimisation are
    % gaining popularity as they offer an inexpensive way to optimise NN
    % hyperparameters by replacing the need to train a network by the
    % evaluation of a fantasy function in unknown regions. The fantasy
    % function aims at finding a trade-off between exploitation (seek
    % around points that produces satisfactory results after evaluation of
    % the network) and exploration (seek in areas of high variance due to
    % uncertainty). However, even these methods suffer from difficulties
    % like defining its own hyperparameters (hyperparameters within a
    % hyperparameter optimisation problem) or defining a robust acquisition
    % function (the fantasy function used to pick new points to evaluate). 
    
    % For all of these reasons and more other I find this field exciting
    % and so do the millions of people who follow research on AI. So let us
    % introduce some NN code in Matlab. 
    
    % Neural Networks in Matlab can be generated and translated into a
    % Simulink model via the NNTRAINTOOL GUI. Here, however, we will hand
    % code the analysis using the properties of the Neural Network Object
    % in Matlab. 
    
    
    % Let us begin by generating a very simple NN model on the red wine
    % data. For vectorisation reasons, it is easier to work with the data
    % in a matrix whose columns represent data samples while rows represent
    % different features. Networks expect matrices of 0s and 1s as
    % labelling data. Hence to convert a categorical array into such a
    % matrix with each column corresponding to a data sample we need the
    % transpose of the output produced by the function dummyvar or
    % alternatively to code it manually. 
    
    
    % We will initialise a pattern recognition network with two hidden 
    % layers with 15 and 10 neurons respectively and specifying the default
    % train function as sequential conjugate gradient (back propagation) 
    % and performance function as cross entropy (same as used in logistic 
    % regression).
    
    NN_Patrn_model_r=patternnet([15 10],'trainscg','crossentropy');
    
    % We will now configure train, validation and test sets. It is common
    % practice to include a validation set to check the performance of the
    % network during training but also to train hyperparameters if desired.
    % We will go for a 60% train, 20% validation, 20% testing
    
    NN_Patrn_model_r.divideParam.trainRatio=0.6;
    NN_Patrn_model_r.divideParam.valRatio=0.2;
    NN_Patrn_model_r.divideParam.testRatio=0.2;
    
    % We will now train the network by using the standardised features as
    % an input and adapting the labels to the binary matrix format expected
    % by a NN. Notice here the function call (train) in a pass by value
    % object fashion. The second output from this method is a structure 
    % that contains data about the training such as the indexes for 
    % training, validation and testing populations. 
    
    [NN_Patrn_model_r,NN_Patrn_trainStats]=train(NN_Patrn_model_r,...
        Features_r_Norm',dummyvar(QCLabel_r)');
    
    % And we will visualise the NN through the Matlab GUI
    view(NN_Patrn_model_r)
    
    
    % Let us now explore the parameters in this Neural Network
    
    disp('Number of inputs refers to the number of input sets seen by the NN');
    disp(NN_Patrn_model_r.numInputs)
    
    disp('Number of layers includes the output but not input')
    disp(NN_Patrn_model_r.numLayers)
    
    disp('Check which layers have bias units assigned')
    disp(NN_Patrn_model_r.biasConnect)
    
    disp(['Check the input connections, the ij element refers to whether',...
    ' the ith layer receives any input from the jth layer'])
    disp(NN_Patrn_model_r.layerConnect)
    
    disp('Check which layers are linked to the output layer')
    disp(NN_Patrn_model_r.outputConnect)
    
    disp(['Check the number of time steps of past inputs that must be',...
        ' supplied to simulate the network']) 
    disp(NN_Patrn_model_r.numInputDelays)
    
    disp(['Check the number of time steps of past layer outputs that must',...
        ' be supplied to simulate the network.'])
    disp( NN_Patrn_model_r.numLayerDelays)
    
    disp('Check total number of weights + bias units in the Net')
    disp(NN_Patrn_model_r.numWeightElements)
    
    disp('Visualise the train function and derivative funtion:')
    disp(NN_Patrn_model_r.trainFcn)
    disp(NN_Patrn_model_r.derivFcn)
    
    disp(['Now we will check the method, mode and parameterisation used to',...
        ' divide the data for training, validation and testing if applicable'])
    disp(NN_Patrn_model_r.divideFcn)
    disp(NN_Patrn_model_r.divideMode)
    disp(NN_Patrn_model_r.divideParam)
    
    disp('Check the initialisation function') 
    % For details on random initialisation requirements see:
    %[https://www.coursera.org/learn/machine-learning/lecture/ND5G5/random-initialization]
    disp( NN_Patrn_model_r.initFcn)

    disp('Visualise performance and plot functions')
    disp(NN_Patrn_model_r.performFcn)
    disp(NN_Patrn_model_r.plotFcns)
    
    % In case the dimensions of the NN are changed by the user
    if NN_Patrn_model_r.numLayers>=3
    
    disp('Visualise weight values for input layers, hidden layers and biases')
    disp('Input layer weights')
    % To explore the weights we retrieve the ij element of a cell array of
    % numeric matrices where the ij element represents the weights of the
    % propagation matrix to the ith layer from the jth input
    disp(NN_Patrn_model_r.IW{1,1})
    disp('Layer 1 to 2 (Theta12)')
    % To explore the weights we retrieve the ij element of a cell array of
    % numeric matrices where the ij element represents the weights of the
    % propagation matrix to the ith layer from the jth layer
    disp(NN_Patrn_model_r.LW{2,1})
    disp('Layer 2 to Output (Theta23)')
    disp(NN_Patrn_model_r.LW{3,3})
    disp('Visualise the bias vector for each layer')
    disp('Layer 1')
    disp(NN_Patrn_model_r.b{1})
    disp('Layer 2')
    disp(NN_Patrn_model_r.b{2})
    disp('Layer 3')
    disp(NN_Patrn_model_r.b{3})
    
    end
    
    %% Section 8-b
    
    % In this section we will verify the performance of the network on the
    % classification problem. 
    
    % Let us predict the class by extracting the max label from the output
    % layer. Notice how to evaluate the prediction we use a method with the
    % same name as the object created to store the network.
    
    NN_Patrn_test_results=NN_Patrn_model_r(Features_r_Norm(NN_Patrn_trainStats.testInd,:)');
    
    % Extract maximum location (QC Label) and convert to column vector. 
    [~,NN_Patrn_pred]=max(NN_Patrn_test_results,[],1);
    NN_Patrn_pred=NN_Patrn_pred';
    
    % Evaluate the performance on the test using the confusion matrix.
    % First extract the true labels from the test set.
    TestLabel_NN_r=QCLabel_r(NN_Patrn_trainStats.testInd);
    
    % Now generate binary data indicating the pertenence class as expected
    % by a NN classifier
    NN_DummyTruth=dummyvar(TestLabel_NN_r)';
    % We use categorical in the following statement to ensure that even
    % though predictions in the test set might not generate all the labels,
    % the confusion matrix is populated knowing these labels are present in
    % the prediction problem.
    NN_DummyPred=dummyvar(categorical(NN_Patrn_pred,1:5))';
    
    % Plot the confusion matrix using the method designed for NN data
    % format(Output from a softmax layer). Notice the use of the test set
    % only.
    FiguresSup.f14=findobj('type','figure','Name','Confusion (plotconfusion)');
    if ishghandle(FiguresSup.f14)
        close(FiguresSup.f14)
    end
    FiguresSup.f14=figure;
    plotconfusion(NN_DummyTruth,NN_DummyPred)
    
    % Plot the Receiver Operating Characteristic for the test set. The
    % receiver operating characteristic depicts how classification varies
    % as the threshold for clasification is modified. Hence it gives an 
    % idea of how well discrimination was modelled in the network. A curve
    % that occupies the upper left corner on this plot means good
    % performance of the classifier. 
    FiguresSup.f15=findobj('type','figure','Name','Receiver Operating Characteristic (plotroc)');
    if ishghandle(FiguresSup.f15)
        close(FiguresSup.f15)
    end
    FiguresSup.f15=figure;
    plotroc(NN_DummyTruth,NN_Patrn_test_results);
    
    % Extract the validation error on the train set for this particular
    % Pattern NN. This value can be checked by using the complete confusion
    % matrix plot from the Neural Network GUI. 
    NN_Pattern_validErr=100*sum(NN_Patrn_pred~=double(TestLabel_NN_r))/length(NN_Patrn_pred);
    
    disp(['Pattern Neural Network error on test set equal to: ',...
        num2str(NN_Pattern_validErr,'%.4g'),'%']);
  
    disp(' ')
    disp('Finished Chapter F')
    
    user_option=input('Would you like to close the figures from this chapter? Y/N [N]:','s');
    if isempty(user_option)
        user_option='n';
    end
    if strcmpi('y',user_option)
              
        for vv=14:15
            figname=strcat('f',num2str(vv));
            if ishghandle(FiguresSup.(figname))&&strcmp(get(FiguresSup.(figname),'BeingDeleted'),'off')
                close(FiguresSup.(figname));
            end
        end
        
    end
    
    disp(' ')
    disp(['--------------------------------------------------------------------------------------------';...
        '------------------------------------     END     -------------------------------------------';...
        '--------------------------------------------------------------------------------------------'])
    
end %if NNswitch


%--------------------------------------------------------------------------
%--------------------------------------------------------------------------


































































