%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% $Author: Lumbrer $    $Date: 2016/12/21 $    $Revision: 0.2$
% Copyright: Francisco Lumbreras
%
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% File Name: VinhoVerde_Quality_EnsembleLearning.m
% Description: Script to introduce model refinement and ensemble learning
% via the use as a practical example of Classifications Trees on the data 
% from Cortez et al. 2009 'Modeling wine preferences by data mining from 
% physicochemical properties'.
%
% This file has been coceived as a script but can be easily converted into
% a parameterised GUI to select data and activate different types of
% algorithms on Matlab. In the first section of the script a number of
% switches are defined to activate or deactivate the different chapters of
% the script. These can all run independently from each other.
%
% In this script, we will study how we can perform deeper model diagnosis
% and increase the ability of a learner by means of combining multiple
% entities in order to improve accuracy. The chapters we will cover in the
% script are the following:
%
% A) The first chapter will cover model refinement. We will first introduce 
% classification tree pruning as a complexity reduction approach followed 
% by embedded feature selection alternatives available when classification
% trees are generated. After selecting a group of features we will use
% cross validation to check the improvement of the model in terms of
% generalisation. This chapter should be understood as a practical 
% introduction to model refinement via the use of decision tree classifiers.  
%
% B) The second chapter introduces Forests of Trees, an example of ensemble
% learning applied to multiple classification trees. In this section we
% will cover the function fitensemble provided by Matlab and that allows
% the analyst to create ensemble learners for any type of weak learner and
% describe how ensemble learnes can be used to fight overfitting. 
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
FiguresEnsem=struct;


%% Define switches to activate each of the chapters of the script
% Switches for chapters A & B
MRswitch=boolean(1);         % A - Activate Model Refinement
ELswitch=boolean(1);         % B - Activate Ensemble Learning

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
% zero mean and 1 std in this script.

Features=DataClean{:,1:end-1};

% Extract number of features and their names
n_features=size(Features,2);
Feature_Names=DataClean.Properties.VariableNames(1:end-1);

% Standardise White & Red wine data to zero mean and unitary std and
% extract nummeric feature arrays. Notice how this is performed AFTER
% splitting the data in Red and White wines
WhiteData_Norm=normTable(WhiteData,{'Centre'});
Features_w=WhiteData{:,:};
Features_w_Norm=WhiteData_Norm{:,:};
RedData_Norm=normTable(RedData,{'Centre'});
Features_r=RedData{:,:};
Features_r_Norm=RedData_Norm{:,:};

% Now let us create a CV partition
HoldOut_percent=0.3;
partition=cvpartition(QCLabel_r,'HoldOut',HoldOut_percent);

% Generate training and test sets
TrainData_r=Features_r_Norm(partition.training,:);
TrainLabel_r=QCLabel_r(partition.training);
TestData_r=Features_r_Norm(partition.test,:);
TestLabel_r=QCLabel_r(partition.test);


if MRswitch % {CHAPTER A}
    
    
    disp(' ')
    disp(['--------------------------------------------------------------------------------------------';...
        '------------------------------------  CHAPTER A  -------------------------------------------';...
        '--------------------------------------------------------------------------------------------'])
    %% Section 3-a
    % Some complex classification models like Neural Networks or
    % Classification Trees have methods developed specifically to diagnose
    % the paramters of the model during learning and even to reduce feature
    % dimensionality based on node activation or tree leaf population.
    % Thus, a node that triggers for any training sample in a neural
    % network is nothing more than a waste of energy and a tree leaf
    % that splits only a very few number of samples is likely to decrease
    % generalisation performance and increases the computational and memory
    % load of the model.
    
    % In this chapter we will create a classification tree to predict
    % the quality label of the Red Wine and we will analyse the importance
    % of different features and the benefits of pruning the last leafs on
    % the tree. In order to assess the benefits of these refinements we
    % will use cross validation.
    
    % First, we will create a classification or decision tree model and 
    % store training time.
    tic
    DT_model_r=fitctree(TrainData_r,TrainLabel_r);
    DT_time=toc;
    
    % Then visualise the result using the view method and requesting a text
    % display
    view(DT_model_r,'Mode','text');
    
    % Calculate training and validation error for the decision tree
    % classifier
    DT_trainErr=100*resubLoss(DT_model_r);
    DT_validErr=100*loss(DT_model_r,TestData_r,TestLabel_r);
    
    disp({['Single Decision Tree classifier',...
        ' Error:'];['On train set: ',num2str(DT_trainErr),'%'];...
        ['On validation set: ',num2str(DT_validErr),'%']});
    
    % Calculate predictions on test data
    DT_pred=predict(DT_model_r,TestData_r);
    
    % And compute the confusion matrix for this classifier
    [conf_mtrx_DT,conf_mtrx_DT_label]=confusionmat(TestLabel_r,DT_pred);
    
    FiguresEnsem.f1=findobj('type','figure','Name','Single Decision Tree Confusion Matrix on Red Wine - Image');
    if ishghandle(FiguresEnsem.f1)
        close(FiguresEnsem.f1)
    end
    FiguresEnsem.f1=figure('Name','Single Decision Tree Confusion Matrix on Red Wine - Image');
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
    text(xpos(:),ypos(:),num2str(conf_mtrx_DT(:),'%d'),'HorizontalAlignment','Center');
    
    
    
    %% Section 3-b
    % Now that we have trained a simple model we will use it to study
    % feature importance and select a subset of features to train a reduced
    % model. Feature importance in a decision tree can be evaluated as the
    % sum of the changes in the classification error function (mean squared
    % error is the default in Matlab) due to splits on each feature and
    % dividing the sum by the number of tree nodes. We can think of this as
    % how much did the decisions made on each feature account for the
    % total reduction in the error function.
    
    % When we use the method predictorImportance for a decision tree model
    % the minimum importance generated for the features is 0.
    % We will extract feature importance and plot it sorted in descend 
    % order.
    
    DT_featImportance=predictorImportance(DT_model_r);
    
    [~,Ind_featImportance]=sort(DT_featImportance,'descend');
    
    FiguresEnsem.f2=findobj('type','figure','Name','Single Decision Tree Features Importance');
    if ishghandle(FiguresEnsem.f2)
        close(FiguresEnsem.f2)
    end
    FiguresEnsem.f2=figure('Name','Single Decision Tree Features Importance');
    
    bar(DT_featImportance(Ind_featImportance))
    set(gca,'XTick',1:11,'XTickLabel',Feature_Names(Ind_featImportance),...
        'XTickLabelRotation',90)
    xlabel('Features')
    ylabel('Importance on Single D. Tree')
    
    
    % Now by inspecting the data we can see that the first five features
    % stand out of the rest even if only by a small difference. We could
    % take either the first five or nine features by importance in order to
    % reduce the model. We will take only five to stress the simplicity of
    % the reduced model against the original.
    tic
    DT_model_red_r=fitctree(TrainData_r(:,Ind_featImportance(1:5)),TrainLabel_r);
    DT_red_time=toc;
    
    % Now we can visualise the result of the reduced dimension tree using 
    % the view method and requesting a text display again
    view(DT_model_red_r,'Mode','text');
    
    % Calculate training and validation error for the decision tree
    % classifier
    DT_red_trainErr=100*resubLoss(DT_model_red_r);
    DT_red_validErr=100*loss(DT_model_red_r,...
        TestData_r(:,Ind_featImportance(1:5)),TestLabel_r);
    
    disp({['Single Decision Tree classifier using reduced feature set based on Importance',...
        ' Error:'];['On train set: ',num2str(DT_red_trainErr,'%.4g'),'%'];...
        ['On validation set: ',num2str(DT_red_validErr,'%.4g'),'%']});
    
    disp({'Single Decision Tree classifier training CPU time:'...
        ;['On complete features set: ',num2str(DT_time),'s'];...
        ['On reduced feature set based on Importance: ',num2str(DT_red_time),'s'];...
        ['Time Reduction equals ',num2str((DT_time-DT_red_time)/DT_time*100,'%.3g'),'%']});
    
    % Calculate predictions on test data
    DT_red_pred=predict(DT_model_red_r,TestData_r(:,Ind_featImportance(1:5)));
    
    % And compute the confusion matrix for this classifier
    [conf_mtrx_DT_red,conf_mtrx_DT_red_label]=confusionmat(TestLabel_r,DT_red_pred);
    
    FiguresEnsem.f3=findobj('type','figure','Name','Single Decision Tree on Reduced Feature Set Confusion Matrix on Red Wine - Image');
    if ishghandle(FiguresEnsem.f3)
        close(FiguresEnsem.f3)
    end
    FiguresEnsem.f3=figure('Name','Single Decision Tree on Reduced Feature Set Confusion Matrix on Red Wine - Image');
    % Display the confusion matrix as an image scaled to cover the full
    % colormap
    imagesc(conf_mtrx_DT_red)
    % Add a colour bar
    colorbar
    % Change the colour map
    colormap autumn
    ylabel('True Class')
    xlabel('Predicted Class')
    ax=gca;
    ax.XTick=1:5; ax.XTickLabel=categories(conf_mtrx_DT_red_label);
    ax.YTick=1:5; ax.YTickLabel=categories(conf_mtrx_DT_red_label);
    
    % Add labels at the center of the image to represent the values on the
    % confusion matrix
    [xpos,ypos] = meshgrid(1:size(conf_mtrx_DT_red,1),1:size(conf_mtrx_DT_red,2));
    text(xpos(:),ypos(:),num2str(conf_mtrx_DT_red(:),'%d'),'HorizontalAlignment','Center');
    
    % We can see how for a much shorter training time and model complexity
    % we can reproduce the same level of performance, hence this particular
    % model needs further working on different directions.
    
    %% Section 3-c
    % We have seen that selecting only the features with high level of
    % importance has led to much shorter training time but scarcely
    % improved the performance of the model (as we expected). An 
    % alternative to perform model simplification using Decision Trees is 
    % to populate a complex tree and then prune the branches at the 
    % extremities of the tree.
    
    % In order to prune the extremities of a decision tree we iteratively
    % assess the effect of pruning to different depths and using cross
    % validation we select the smallest model whose error lies within one
    % standard error of the minimum cost (or an alternative error 
    % criterion).
    
    % First we will understand how deep our tree model allows us to prune
    % and store such value minus one in order to specify we will tolerate
    % pruning provided that at least one branch is left. PruneList
    % generates a list per node of the number of pruning levels existing on
    % the path under such node. The maximum of this list is the greatest
    % level of pruning considered in order to maintain error within desired
    % bound (one standard error from minimum).
    
    max_prune=max(DT_model_r.PruneList)-1;
    
    % We will extract the best prunnning by means of using cross validation
    % error accross the possible alternatives. We will limit prune level
    % and specify tree size to be chosen by looking at the smallest tree
    % whose cost is within one standard error of the minimum cost.
    % This method will allow us to inspect the error (we would expect this
    % to increase as we prune), the standard error based on sampling, the
    % number of leaf nodes as we prune the tree and the best pruning level.
    [DT_E,DT_SE,DT_nLeaf,DT_BestPrune]=cvLoss(DT_model_r,'SubTrees',...
        0:max_prune,'TreeSize','se','KFold',10);
    
    % We will report the change. Note how the first result reported was for
    % no pruning as we defined the SubTrees field as a vector beginning at
    % 0 - no pruning.
    disp(['Standard error approach suggests tree pruning from ',num2str(DT_nLeaf(1)),...
        ' leaf nodes to ',num2str(DT_nLeaf(1+DT_BestPrune)),' leaf nodes.',...
        ' Best pruning level equals ',num2str(DT_BestPrune)]);
    
    % Now we will prune the model.
    Dt_model_pruned_r=prune(DT_model_r,'level',DT_BestPrune);
    
    % And compare both on a graph display
    
    view(DT_model_r,'mode','graph');
    view(Dt_model_pruned_r,'mode','graph');
    
    % NOTE: See how pruning the tree has led to the elimination of class
    % prediction for quality labels A and E. This means such categories 
    % will never be correctly predicted by the pruned model. This is a 
    % result of the poor data distribution in the training and test sets, 
    % where the population of B, C and D quality label samples clearly 
    % outweights that of A and E.
    
    % We could correct this behaviour by producing artificial train data
    % from A and E by means of introducing small perturbations to the data
    % but that would ideally require expert domain knowledge.
    
    
    disp(' ')
    disp('Finished Chapter A ')
    
    user_option=input('Would you like to close the figures from this chapter? Y/N [N]:','s');
    if isempty(user_option)
        user_option='n';
    end
    if strcmpi('y',user_option)
        
        for vv=1:3
            figname=strcat('f',num2str(vv));
            if ishghandle(FiguresEnsem.(figname))&&strcmp(get(FiguresEnsem.(figname),'BeingDeleted'),'off')
                close(FiguresEnsem.(figname));
            end
        end
    end
    
    
    disp(' ')
    disp(['--------------------------------------------------------------------------------------------';...
        '------------------------------------     END     -------------------------------------------';...
        '--------------------------------------------------------------------------------------------'])
    
    fprintf('\nProgram paused. Press enter to continue.\n\n');
    pause;
    
    
end % MRswitch


if ELswitch % {CHAPTER B}
    
    disp(' ')
    disp(['--------------------------------------------------------------------------------------------';...
        '------------------------------------  CHAPTER B  -------------------------------------------';...
        '--------------------------------------------------------------------------------------------'])
    
    %% Section 4-a
    % Complex decision trees have the ability to match the structure of any
    % dataset provided that there are no inconsistencies in feature values
    % across different classes. Because of this, training a single decision
    % tree typically leads to good training error but very poor
    % generalisation error. Overfitting is hence a problem we have treated
    % so far on this script by means of pruning our single decision tree
    % trained over a hold-out cross validated population and attempting to
    % remove features based on their importance.
    
    % The fact that decision trees are prone to overfitting can be also
    % referred as by stating that these are weak learners. A weak learner
    % shows high sensitivity to the data used to train it.
    
    % An alternative approach to fighting this overfitting problem is to
    % create several trees or a forest of tree learners and comparing the
    % predictions for each sample in oprder to select the class with the
    % highest number of votes. This approach enables us to assess the
    % quality of the forest as a correct decision made unanimously shows
    % greater confidence in the forest than a contentious one.
    
    % Ensemble learning can be performed using alternative learners and
    % thus given any weak learner, Matlab allows us to define an ensemble
    % learner by grouping multiple entities. We will show how to perform
    % the latter in this section.
    
    % We will strat by performing bootstrap aggregation, also referred to
    % as bagging, to create an ensemble learner of decision trees. We will
    % ask for the out of the bag prediction performance to check how error
    % varies with the number of trees considered.
    
    if Parallelswitch
        
        % Set options to use parallelisation
        Bagger_options = statset('UseParallel',1);
        % Start the parallel pool to ensure more predictable results
        delete(gcp('nocreate'));
        poolobj=parpool('local',n_cores);
        DTBag_model_r=TreeBagger(100,TrainData_r,TrainLabel_r,'OOBPred',...
            'on','Options',Bagger_options);
        
        % Close parallel pool
        delete(gcp('nocreate'));
        
    else
        
        DTBag_model_r=TreeBagger(100,TrainData_r,TrainLabel_r,'OOBPred','on');
    end
    % Predict using the bag model. Note how the method predict for a bagged
    % tree learner produces a cell array and hence we need to convert it to
    % a categorical array.
    
    DTBag_pred=categorical(predict(DTBag_model_r,TestData_r));
    
    % Extract and display the error from this learner. We cannot use the
    % method loss for a TreeBagger object
    DTBag_validErr=100*nnz(DTBag_pred ~= TestLabel_r)/length(TestLabel_r);
    
    disp(['Hold-out Cross Validation Error for Forest of 100 Trees in Red Wine Equals ',...
        num2str(DTBag_validErr,'%.4g'),'%']);
    
    % And compute the confusion matrix for this classifier
    [conf_mtrx_DTBag,conf_mtrx_DTBag_label]=confusionmat(TestLabel_r,DTBag_pred);
    
    FiguresEnsem.f4=findobj('type','figure','Name',...
        '100 Decision Trees Forest Confusion Matrix on Red Wine - Image');
    if ishghandle(FiguresEnsem.f4)
        close(FiguresEnsem.f4)
    end
    FiguresEnsem.f4=figure('Name',...
        '100 Decision Trees Forest Confusion Matrix on Red Wine - Image');
    % Display the confusion matrix as an image scaled to cover the full
    % colormap
    imagesc(conf_mtrx_DTBag)
    % Add a colour bar
    colorbar
    % Change the colour map
    colormap parula
    ylabel('True Class')
    xlabel('Predicted Class')
    ax=gca;
    ax.XTick=1:5; ax.XTickLabel=categories(conf_mtrx_DTBag_label);
    ax.YTick=1:5; ax.YTickLabel=categories(conf_mtrx_DTBag_label);
    
    % Add labels at the center of the image to represent the values on the
    % confusion matrix
    [xpos,ypos] = meshgrid(1:size(conf_mtrx_DTBag,1),1:size(conf_mtrx_DTBag,2));
    text(xpos(:),ypos(:),num2str(conf_mtrx_DTBag(:),'%d'),'HorizontalAlignment','Center');
    
    % Now we will display how error changes with the number of trees
    % considered.
    
    FiguresEnsem.f5=findobj('type','figure','Name',...
        'Out-of-bag Error for Forest of Decision Trees on Red Wine');
    if ishghandle(FiguresEnsem.f5)
        close(FiguresEnsem.f5)
    end
    FiguresEnsem.f5=figure('Name','Out-of-bag Error for Forest of Decision Trees on Red Wine');
    plot(100*oobError(DTBag_model_r))
    xlabel('Number of Trees in Bag (Forest Size)')
    ylabel('Out-of-bag Error [%]')
    
    %% Section 4-b
    % In this section we will provide an alternative way to define a Forest
    % of Classification Trees in Matlab but only using the important
    % features as selected by the importance criterion in chapter A.
    
    % Hence we will repeat the calculation of such relevant features but
    % will not display any information as that was already done earlier in
    % Chapter A. 
    
    if ~MRswitch
        DT_model_r=fitctree(TrainData_r,TrainLabel_r);
        DT_featImportance=predictorImportance(DT_model_r);
        
        [~,Ind_featImportance]=sort(DT_featImportance,'descend');
    end
    
    DTBag_red_model_r=fitensemble(TrainData_r(:,Ind_featImportance(1:5)),...
        TrainLabel_r,'Bag',100,'Tree','Type','Classification');
    
    % Predict on validation set and calculate error using loss method. Note
    % that this time the output is categorical and the second output is a
    % discrete probability distribution on the class label for each
    % training sample based on the output from the ensemble learner.
    [DTBag_red_pred,DTBag_red_allpred] = predict(DTBag_red_model_r,...
        TestData_r(:,Ind_featImportance(1:5)));
    DTBag_red_validErr = 100*loss(DTBag_red_model_r,...
        TestData_r(:,Ind_featImportance(1:5)),TestLabel_r);
    
    disp(' ')
    disp(['Hold-out CV Error for Forest of 100 Trees using',...
        ' Reduced Feature Set based on Importance from Single Tree Predictor in Red Wine Equals ',...
        num2str(DTBag_red_validErr,'%.4g'),'%']);
    disp('Using features:')
    disp(Feature_Names(Ind_featImportance(1:5)))
    
    % And compute the confusion matrix for this classifier
    [conf_mtrx_DTBag_red,conf_mtrx_DTBag_red_label]=confusionmat(TestLabel_r,DTBag_red_pred);
    
    FiguresEnsem.f6=findobj('type','figure','Name',...
        '100 Decision Trees Forest using Reduced Feature Subset Confusion Matrix on Red Wine - Image');
    if ishghandle(FiguresEnsem.f6)
        close(FiguresEnsem.f6)
    end
    FiguresEnsem.f6=figure('Name',...
        '100 Decision Trees Forest using Reduced Feature Subset Confusion Matrix on Red Wine - Image');
    % Display the confusion matrix as an image scaled to cover the full
    % colormap
    imagesc(conf_mtrx_DTBag_red)
    % Add a colour bar
    colorbar
    % Change the colour map
    colormap parula
    ylabel('True Class')
    xlabel('Predicted Class')
    ax=gca;
    ax.XTick=1:5; ax.XTickLabel=categories(conf_mtrx_DTBag_red_label);
    ax.YTick=1:5; ax.YTickLabel=categories(conf_mtrx_DTBag_red_label);
    
    % Add labels at the center of the image to represent the values on the
    % confusion matrix
    [xpos,ypos] = meshgrid(1:size(conf_mtrx_DTBag_red,1),1:size(conf_mtrx_DTBag_red,2));
    text(xpos(:),ypos(:),num2str(conf_mtrx_DTBag_red(:),'%d'),'HorizontalAlignment','Center');
    
    
    disp(' ')
    disp('Finished Chapter B ')
    
    user_option=input('Would you like to close the figures from this chapter? Y/N [N]:','s');
    if isempty(user_option)
        user_option='n';
    end
    if strcmpi('y',user_option)
        
        for vv=4:6
            figname=strcat('f',num2str(vv));
            if ishghandle(FiguresEnsem.(figname))&&strcmp(get(FiguresEnsem.(figname),'BeingDeleted'),'off')
                close(FiguresEnsem.(figname));
            end
        end
    end
    
    disp(' ')
    disp(['--------------------------------------------------------------------------------------------';...
        '------------------------------------     END     -------------------------------------------';...
        '--------------------------------------------------------------------------------------------'])
    fprintf('\nProgram paused. Press enter to continue.\n\n');
    pause;
    
    
end % ELswitch


%--------------------------------------------------------------------------
%--------------------------------------------------------------------------


































































