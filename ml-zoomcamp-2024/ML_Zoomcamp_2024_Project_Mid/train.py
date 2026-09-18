import re
import pickle
import functions
import numpy as np
import pandas as pd


# Parameters.
params = {
    'eta': 0.3, 
    'max_depth': 6,
    'min_child_weight': 1,
    
    'objective': 'binary:logistic',
    'nthread': 8,
    
    'seed': 1,
    'verbosity': 1,
}

output_file = 'xgb_model.bin'


# Data Preparation.
df = pd.read_csv('./data/MIT-BIH_Arrhythmia_Database.csv')


# Transform multi-class labels into binary-class (1 and 0).
df['type'] = df.type.map({'N': 0, 'VEB': 1, 'SVEB': 1, 'F': 1, 'Q': 1})


# Train model.
score_xgb, y_test, y_pred, dv, model = functions.score_full(df, params)

print(f'AUC = {score_xgb}')


# Save the model.
with open(output_file, 'wb') as f_out:
    pickle.dump((dv, model), f_out)

print(f'The model is saved to {output_file}')
