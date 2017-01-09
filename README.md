# Intro_Machine_Learning_Matlab

A compilation of built-in functionalities to apply multiple Machine Learning algorithms with Matlab. Requires the Machine Learning, Parallel Computing, Statistical and Optimisation toolboxes. 

The purpose of this work is to provide an insight into some of the methods and tools available within Matlab to perform Machine Learning. It is not intended to be used as part of a software package although I hope it can help develop algorithms in a more efficient way to those involved in hands-on Machine Learning research as that was the original purpose of this work as part of a broader project on feature learning.

## Vinho Verde Wine Quality Dataset

At the moment the data from 
> Cortez et al. 2009 - 'Modeling wine preferences by data mining from physicochemical properties'.

is the only dataset used and hence a single subfolder has been created at the root of the repository. 

Within this folder there are four main scripts and additional folders including functions to remove poor quality data samples, standardisation and rescaling of the data and additional plot functionalities. The four main scripts cover the following points:

### Data Visualisation

1-*VinhoVerde_Quality_ViewData.m*: This script contains code to visualise the multiclass data and different statistical metrics split by class.

### Unsupervised Learning

2-*VinhoVerde_Quality_UnsupervisedLearning.m*: Unsupervised learning applied to this wine dataset including four chapters.

     - Chapter 2.A) A study of the data natural clustering via self organising maps (SOM) 
     including multiple visualisations of the latter. 
    
     - Chapter 2.B) A study of K-means clustering on the data and the parameterisation of
     the algorithm. In this chapter, simple parallelisation is used to study
     the influence of varying parameters and different criteria are explored
     to extract an optimal number of clusters. Additionally, data is
     visualised giving preference to the clusters showing greatest sparsity
     accross the data. 
    
     - Chapter 2.C) A brief tutorial on the tools in Matlab used to perform Hierarchical
     clustering and evaluate the suitability of such analysis to the sample
     data
    
     - Chapter 2.D) A study on EM Mixture of Gaussian clustering including 2D
     visualisation of the data to better understand the assumptions behind
     this approach to clustering. 

### Supervised Learning
 
3-*VinhoVerde_Quality_SupervisedLearning.m*: Supervised learning applied to wine type (red vs white) and quality label within each wine  type. This script includes 6 chapters.

     - Chapter 3.A) A study on clustering via K-Nearest Neighbours and the influence of
     applying different distance metrics and distance factors. For v. 2016b
     users, Bayesian optimisation is used to optimise hyperparameters and an
     alternative brute force optimisation within a selected hyperspace is
     offered for users of earlier versions. Simple parallelisation is used to
     reduce the computation time. 

     - Chapter 3.B) A study on Naive Bayes classification and the possible distributions
     that are predefined as options in this analysis. The assumptions from
     Naive Bayes uses independency accross feature probabilities given a 
     classification label (these probabilities are typically assumed 
     multinomial or Gaussian) and here we will challenge this assumption for 
     the case of continuous Gaussian distributions. 

     - Chapter 3.C) An analysis of Gaussian Discriminant analysis in a binary decision
     problem and the different assumptions that can be made when selecting
     multivariate Gaussians for the feature posteriori distributions
     P(fi|Class). We will use functions to check the validity of Gaussian
     assumptions and examine covariances across features and labels.

     - Chapter 3.D) A chapter covering decision trees as classifiers and the tools offered
     to train and constraint these structures.  

     - Chapter 3.E) The introduction of SVMs in Matlab, different Kernel approaches and
     solvers that can be used for this maximum margin binary classifier.
     Parallelisation is used to perform brute force optimisation of
     hyperparameters and different regularisation strategies are analysed.
     An additional section within this chapter covers the use of Error
     Correcting Output Codes to apply SVM classification to multiclass
     problems. 

     - Chapter 3.F) A brief chapter on pattern recognition Neural Networks with code on
     how to generate, train and examine the performance of a Network. This
     chapter includes a subsession analysis the different options that can be
     parameterised in a Neural Netowrk. Activation functions, architecture 
     optimisation and training functions are revised.
     Convolutional Networks, Deep Belief Networks and Auto Regressive Networks
     are not covered in this section and for such analysis Python libraries
     are preferred by the author.
 
### Dimensionality Reduction
 
4-*VinhoVerde_Quality_FeaturesReduction.m*: Script to perform dimensionality reduction and feature selection on the data including four chapters.

    - Chapter 4.A) The first chapter covers Principal Components Analysis and how it can
     be used to transform the dimensionality of a feature set whilst
     maintaining variance information from the original data. This section
     covers different methods to extract Principal Components and the feature
     scores to be used in latter training of an algorithm as well as methods
     to intuitively visualise the Principal Components.

    - Chapter 4.B) The second chapter introduces Factor Analysis and the circumstances
     under which one might consider such algorithm to perform dimensionality
     reduction. In addition, the extraction of feature scores and the division
     of variance into individual, common and error is discussed. Methods to
     intuitively visualise factor loadings are included.

    - Chapter 4.C) This chapter covers the use of reduced scores from both Principal
     Components and Common Factors to train a classification model and compare
     the performance and execution time with the use of the original features
     by means of a practical example.

    - Chapter 4.D) The last chapter covers Feature Selection methods and in particular
     the built-in functions available in Matlab to perform wrapper selection.
     Given a practical example with fixed hyperparameters, the performance
     imporvement after sequential forwards and backwards elimination is
     studied. Furthermore, a brief example on randomised single search is
     provided to explain the limitations of sequential approaches.

### Ensemble Learning

5-*VinhoVerde_Quality_EnsembleLearning.m*: Introduction to model refinement and ensemble learning via the use of decision trees asa practical example. 

    - Chapter 5.A) The first chapter will cover model refinement. We will first introduce 
     classification tree pruning as a complexity reduction approach followed 
     by embedded feature selection alternatives available when classification
     trees are generated. After selecting a group of features we will use
     cross validation to check the improvement of the model in terms of
     generalisation. This chapter should be understood as a practical 
     introduction to model refinement via the use of decision tree classifiers.  

    - Chapter 5.B) The second chapter introduces Forests of Trees, an example of ensemble
     learning applied to multiple classification trees. In this section we
     will cover the function fitensemble provided by Matlab and that allows
     the analyst to create ensemble learners for any type of weak learner and
     describe how ensemble learnes can be used to fight overfitting. 


*Note: All the scripts use switches that manually allow to enable or disable different chapters. All chapters within a script can run independently form each other and at termination of each chapter the user is given the option to close all the figures generated during the last chapter run.* 
