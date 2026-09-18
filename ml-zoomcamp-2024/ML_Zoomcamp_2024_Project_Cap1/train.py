import os
import functions
import numpy as np
import pandas as pd
import tensorflow as tf

# Define seed
SEED = 22
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Define the model
THE_MODEL = 'InceptionV3'

# Define constants
IMG_SIZE = (299, 299)
INPUT_SHAPE = (299, 299, 3)

# Define inputs
BATCH_SIZE = 32
DROP_RATE = 0.5
EP0CHS = 30

# Define optimizer
LEARNING_RATE = 0.00001
OPTIMIZER = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

# Define base model
BASE_MODEL = tf.keras.applications.InceptionV3(
    weights='imagenet',
    include_top=False,
    input_shape=INPUT_SHAPE
)
  
# Define paths
train_path="./data/breast-cancer-detection/train"
test_path="./data/breast-cancer-detection/test"

# Create dataframes
train_df = functions.create_dataframe(train_path)
test_df = functions.create_dataframe(test_path)

# Get generators
train_generator, test_generator = functions.create_data(BATCH_SIZE)

# Get results
model = functions.load_fit(BASE_MODEL, DROP_RATE)

# Save model as h5
tf.keras.models.save_model('breast_cancer.h5')

# Save model as keras
tf.keras.models.save_model('breast_cancer.keras')