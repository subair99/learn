import re
import pickle
import functions
import numpy as np
import pandas as pd
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer


#  Define a function to split the data.
def split_full(df):
    
    df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=22)

    df_full_train = df_full_train.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    y_full_train = df_full_train.type.values
    y_test = df_test.type.values

    del df_full_train['type']
    del df_test['type']

    sm = SMOTE(random_state=22)
    df_full_train_sm, y_full_train_sm = sm.fit_resample(df_full_train, y_full_train)

    dv = DictVectorizer(sparse=False)

    full_train_dicts = df_full_train_sm.to_dict(orient='records')
    test_dicts = df_test.to_dict(orient='records')
    
    X_full_train = dv.fit_transform(full_train_dicts)
    X_test = dv.transform(test_dicts)

    return X_full_train, X_test, y_full_train_sm, y_test, dv


# Define a function to create full dtrain and dval.
def dicts_full(df):
    X_full_train, X_test, y_full_train, y_test, dv = split_full(df)

    features = list(dv.get_feature_names_out())
    dtrain = xgb.DMatrix(X_full_train, label=y_full_train, feature_names=features)
    dval = xgb.DMatrix(X_test, label=y_test, feature_names=features)

    return dtrain, dval, y_test, dv


# Define a function to get xgb score.
def score_full(df, params):
    dtrain, dval, y_test, dv = dicts_full(df)
    model = xgb.train(params, dtrain, num_boost_round=10)
    y_pred = model.predict(dval)
    score = roc_auc_score(y_test, y_pred)

    return score, y_test, y_pred, dv, model


# Define the function to process data.
def process_full(dict_data):
    
    cor_rem = ['1_pre-RR', '1_post-RR', '0_qrs_morph0', '1_qrs_morph0', 
               '1_qrs_morph1', '1_qrs_morph2', '1_qrs_morph3', '1_qrs_morph4']

    for col in cor_rem:
        del dict_data[col]

    return dict_data


# Define function to make predictions.
def predict_full(predict_data, dv, model):
    patient = process_full(predict_data)
    X = dv.transform([patient])
    y_pred = model.inplace_predict(X)

    if y_pred >= 0.5:
        print('THE PATIENT HAS ARRHYTHMIA')
    else:
        print('THE PATIENT HAS NO ARRHYTHMIA')

