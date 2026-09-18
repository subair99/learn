# ECG Arrhythmia Classification 

<p align="center">
  <img src="./images/ECG_Arrhythmia_Diagram.jpg">
</p>


## Project overview

The Machine Learning Zoomcamp Course is at the midterm project phase where we are expected to create a project that demonstrates the knowledge we have gathered so far. I intent to complete the task in stages and this is the first of the lot where I will select the data I will use for the project. The data I have selected is the [ECG Arrhythmia Classification Dataset](https://www.kaggle.com/datasets/sadmansakib7/ecg-arrhythmia-classification-dataset/data?select=INCART+2-lead+Arrhythmia+Database.csv).

The dataset contains features extracted two-lead ECG signal (lead II, V) from the MIT-BIH Arrhythmia dataset (Physionet). Additionally, programs have been used to extract relevant features from ECG signals to enable classification into regular or irregular heartbeats. The diagram of the signal is attached.

There are four different versions of the dataset with each employing 2-lead ECG features. The datasets from PhysioNet are MIT-BIH Supraventricular Arrhythmia Database, MIT-BIH Arrhythmia Database, St Petersburg INCART 12-lead Arrhythmia Database, and Sudden Cardiac Death Holter Database. The one I selected is the MIT-BIH Arrhythmia Database which is a compressed comma separated file of size 16.7MB and 43.3MB after extraction.


This project was implemented as a requirement for the completion of 
[MACHINE LEARNING Zoomcamp](https://github.com/DataTalksClub/machine-learning-zoomcamp) -
a free course about LLMs and RAG.


## Technologies

- Python 3.10
- Docker
- Flask
- XGBoost 
- Scikit-Learn
- Pandas
- Gunicorn
- Matplotlib 
- Seaborn


## Exploratory Data Analysis 

### First training 
After selecting the dataset, it was then explored and found to consist of 100689 rows and 34 columns. No duplicates or missing values were found, the list of the columns are  'record', 'type', '0_pre-RR', '0_post-RR', '0_pPeak', '0_tPeak', '0_rPeak', '0_sPeak', '0_qPeak', '0_qrs_interval', '0_pq_interval', '0_qt_interval', '0_st_interval', '0_qrs_morph0', '0_qrs_morph1', '0_qrs_morph2', '0_qrs_morph3', '0_qrs_morph4', '1_pre-RR', '1_post-RR', '1_pPeak', '1_tPeak', '1_rPeak', '1_sPeak', '1_qPeak', '1_qrs_interval', '1_pq_interval', '1_qt_interval', '1_st_interval', '1_qrs_morph0', '1_qrs_morph1', '1_qrs_morph2', '1_qrs_morph3', '1_qrs_morph4'.

The 'type' column is the target, and it consists of the variables N which represents normal while VEB, SVEB, F, and Q represents arrhythmia. To transform the project from a multi-class problem to a bi-class one, N was changed to normal while VEB, SVEB, F, and Q were all individually change to arrhythmia, and finally normal was encoded as 0 and arrhythmia was encoded as 1. The count of the target was taken, and it showed that 0 is 90083 and 1 is 10606, this is an imbalanced data that requires attention. The method used to solve this problem was the application of Synthetic Minority Oversampling TEchnique (SMOTE) which creates synthetic samples to make the 1s count balanced with the Os for training purpose only.

The next step was to conduct the first training to obtain the Receiver Operating Characteristic Area Under the Curve (ROC UAC) score of the selected models and the result is as listed below:

Logistic Regression Algorithm - - - - -0.9567  
Random Forest Classifier - - - - - - - 0.9989  
XGB Classifier - - - - - - - - - - - - 0.9955  

The diagram below is a sample of the dataset’s statistics.

<p align="center">
  <img src="./images/Dataset_Statistics.jpg">
</p>


### Second training
After the first training, the dataset was checked for collinearity and the following were observed:
Correlation between '0_pre-RR' and '1_pre-RR' is 1.00
Correlation between '0_post-RR' and '1_post-RR' is 1.00
Correlation between '0_qPeak' and '0_qrs_morph0' is 1.00
Correlation between '1_qPeak' and '1_qrs_morph0' is 1.00
Correlation between '1_qPeak' and '1_qrs_morph1' is 0.97
Correlation between '1_qrs_morph1' and '1_qrs_morph2' is 0.96
Correlation between '1_qrs_morph2' and '1_qrs_morph3' is 0.95
Correlation between '1_qrs_morph3' and '1_qrs_morph4' is 0.96

So, the following columns '1_pre-RR', '1_post-RR', '0_qrs_morph0', '1_qrs_morph0', '1_qrs_morph1', '1_qrs_morph2', '1_qrs_morph3', '1_qrs_morph4' were dropped from the dataset resulting in 100689 rows and 26 columns.

A second training was conducted, and the ROC UAC score are listed below:

Logistic Regression Algorithm - - - - -0.9521  
Random Forest Classifier - - - - - - - 0.9991  
XGB Classifier - - - - - - - - - - - - 0.9951  

The diagram below is the heatmap of the dataset.

<p align="center">
  <img src="./images/Heatmap.png">
</p>


### Third training
After the second training it was observed that the columns removed did not affect the score of the models that much. The dataset was then checked for outliers with boxplots, and it was found that the columns, '0_pre-RR', '0_post-RR', '0_qrs_interval', '0_pq_interval', '0_qt_interval', '0_st_interval', '1_qrs_interval', '1_pq_interval', '1_qt_interval', '1_st_interval’ had several outliers which were reduced by using the outlier removal function. After the process the data was found to consist of 98644 rows and 26 columns which means that only 2.03% of the rows were removed.

A third training was conducted, and the ROC UAC scores are listed below:

Logistic Regression Algorithm - - - - -0.9521  
Random Forest Classifier - - - - - - - 0.9991  
XGB Classifier - - - - - - - - - - - - 0.9951  

The diagram below is the boxplot of a section of the dataset before and after outliers’ removal.

<p align="center">
  <img src="./images/Outliers.jpg">
</p>


## Feature Importances and Models Tuning

After removing the outliers, the feature importance was computed for the three retained algorithms, feature importance refers to techniques for determining the degree to which different features, or variables, impact a machine learning model’s prediction. It is useful for data comprehension, Model Improvement and Model Interpretability, and dimensionality reduction. Feature importance of the three models were computed using the elimination method where a feature is eliminated from the training and the score obtained is compared with that of complete features. The eliminated feature with the highest difference is the most important for the model. For Logistic Regression Algorithm the three most importance features are '0_pre-RR', 'record' and '0_rPeak', for Random Forest Classifier they are 'record', '0_pre-RR' and '0_pq_interva' and for XGB Classifier they are '0_pre-RR', 'record' and '1_qPeak', with more in the attached diagram.

The models tuning which is essential to ensure a model performs at its best was then conducted. Tuning a model allows for the selection of the best hyperparameters and settings for an algorithm to learn from, which help in achieving the highest rate of performance possible. Tuning a pretrained model allows for its adoption to suit more specialized use cases while maintaining its original capabilities. Default hyper parameter configurations are unlikely to provide optimal performance for any use case, so each model and data set combination requires its own tuning. Tuning a model is a critical step in the machine learning workflow, but it can be challenging, expensive and time-consuming. To save time and cost in training, it is important to understand the parameters being tuned and the parameter range for which the model checks its performance. The tuning results indicated that the best n_estimators for Logistic Regression Algorithm is 0.6, the best n_estimators and max_depth for the Random Forest Classifier are 175 and 9 respectively, and the best eta and max_depth for the XGB Classifier are 0.3 and 8 respectively.

The diagram below show the feature importances of the three models.

<p align="center">
  <img src="./images/Feature_Importance.jpg">
</p>


## Final Models

The tuning of the models made the best hyperparameters available so the best models can be created and analysed by plotting their ROC curve, precision-recall curve and f1 curve. At this stage I went back to all the results with the aim of analysing the reasons for the value obtained. Before applying SMOTE on the data, the results obtained were:

model_score_lra_1 = 0.955  
part_score_lra_1 = 0.6444  
model_score_rfc_1 = 0.9977  
part_score_rfc_1 = 0.9752  
model_score_xgb_1 = 0.993  
part_score_xgb_1 = 0.9047  

Note that the model_score_lra is from the Logistic Regression Algorithm, rfc is from the Random Forest Classifier and xgb is from the XGB Classifier. The part_score is the result obtained when the arrhythmia data is predicted with the current model as explained in the last publication. After applying SMOTE on the data, the results obtained were:

model_score_lra_2 = 0.9567  
part_score_lra_2 = 0.7174  
model_score_rfc_2 = 0.9989  
part_score_rfc_2 = 0.9892  
model_score_xgb_2 = 0.9955  
part_score_xgb_2 = 0.9746  

Clearly, the application of SMOTE had a positive effect on the prediction of the arrhythmia data as the score increased for all the models, so its good practice to apply it whenever the data is imbalanced. The results obtained after collinearity check and outliers’ removal are:

model_score_lra_3 = 0.9507  
part_score_lra_3 = 0.6737  
model_score_rfc_3 = 0.9993  
part_score_rfc_3 = 0.9901  
model_score_xgb_3 = 0.9948  
part_score_xgb_3 = 0.8985  

In this case the results are lower, this not a problem because the most important job of a model in machine learning is to perform to its best on unseen data which will not be possible if almost perfect colinear columns and outliers are present. The results obtained for the final models are:

model_score_lra_4 = 0.9508  
part_score_lra_4 = 0.6737  
model_score_rfc_4 = 0.9974  
part_score_rfc_4 = 0.9733  
model_score_xgb_4 = 0.9964  
part_score_xgb_4 = 0.9717  

In the final publication I will select the model to be used for the project with the criteria for selection, and the deployment.The diagram below show some of the plots of the three models.

<p align="center">
  <img src="./images/Final_Models_Plots.jpg">
</p>


## Selected Model

The selected model for the project is XGB Classifier because it had a considerably high score and a very short time to complete the training as shown below.

time_lra = 7.85
time_rfc = 35.94
time_xgb = 2.85

The notebook code were then refactored to python scripts that include train.py, functions.py, predict-test.py, predict.py, docker file with pipfile and pipfile.lock created when the pipenv environment was initiated and the required programs were installed. The diagrams below show some of the results obtained when the scripts were used to run the project.

<p align="center">
  <img src="./images/run_train.jpg">
</p>

<p align="center">
  <img src="./images/run_gunicorn.jpg">
</p>

<p align="center">
  <img src="./images/run_predict-test.jpg">
</p>

<p align="center">
  <img src="./images/docker_build.jpg">
</p>

<p align="center">
  <img src="./images/docker_image.jpg">
</p>


## Acknowledgements

My greatest appreciation goes to [Alexey Grigorev](https://www.linkedin.com/feed/?highlightedUpdateType=SHARED_BY_YOUR_NETWORK&highlightedUpdateUrn=urn%3Ali%3Aactivity%3A7254859071965548545#:~:text=post%20number%201-,Alexey%20Grigorev,-Alexey%20Grigorev) for his invaluable opportunity.
