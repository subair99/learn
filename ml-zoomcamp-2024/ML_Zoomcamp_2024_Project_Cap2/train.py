import os
from PIL import Image
from io import BytesIO
from urllib import request
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
import tensorflow.lite as tflite
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import functions

# Create the training dataframe
tr_path = '../data/brain-tumor-mri-dataset/Training'
tr_df = functions.create_dataframe(tr_path)

# Create the full test data
ts_path = '../data/brain-tumor-mri-dataset/Testing'
ts_df_full = functions.create_dataframe(ts_path)

# Create test and valid data for training
vd_df, ts_df = functions.test_valid(ts_df_full)

# Define constants
batch_size = 32
img_size = (299, 299)

# Process the dataframes
tr_gen, vd_gen, ts_gen = functions.process_data(tr_df, vd_df, ts_df)

# Create model
LEARNING_RATE = 0.001
img_shape=(299,299,3)
base_model = base_model = tf.keras.applications.Xception(include_top= False, weights= "imagenet",
                            input_shape= img_shape, pooling= 'max')

model, summary = functions.create_model(base_model, LEARNING_RATE)

# Train model
hist = functions.train_model(tr_gen, vd_gen)

# Save as keras model
tf.keras.models.save_model(model, 'brain_tumor_model.keras')





# Eveluate results
evaluate_results(tr_gen, vd_gen, ts_gen)
