
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


# Define function to create dataframe
def create_dataframe(the_path):
    classes, class_paths = zip(*[(label, os.path.join(the_path, label, image))
                                 for label in os.listdir(the_path) if os.path.isdir(os.path.join(the_path, label))
                                 for image in os.listdir(os.path.join(the_path, label))])

    df = pd.DataFrame({'Class Path': class_paths, 'Class': classes})
    
    return df



# Define function to count images
def count_images(df):
    plt.figure(figsize=(15, 7))
    ax = sns.countplot(data=df , y=df['Class'])
    plt.xlabel('')
    plt.ylabel('')
    plt.title('Count of images in each class', fontsize=20)
    ax.bar_label(ax.containers[0])

    return plt.show();



# Define function to create test and valid data for training
def test_valid(ts_df_full):
    vd_df, ts_df = train_test_split(ts_df_full, train_size=0.5, random_state=20, stratify=ts_df_full['Class'])

    return vd_df, ts_df



# Define function to precess the data
def process_data(tr_df, vd_df, ts_df):
    rd_gen = ImageDataGenerator(rescale=1/255,
                          brightness_range=(0.8, 1.2))

    ts_gen = ImageDataGenerator(rescale=1/255)
    
    
    tr_gen = rd_gen.flow_from_dataframe(tr_df, x_col='Class Path',
                                      y_col='Class', batch_size=batch_size,
                                      target_size=img_size)
    
    vd_gen = rd_gen.flow_from_dataframe(vd_df, x_col='Class Path',
                                         y_col='Class', batch_size=batch_size,
                                         target_size=img_size)
    
    ts_gen = ts_gen.flow_from_dataframe(ts_df, x_col='Class Path',
                                      y_col='Class', batch_size=16,
                                      target_size=img_size, shuffle=False)

    return tr_gen, vd_gen, ts_gen



# Define function to show sample of data
def sample_data(tr_gen, ts_gen):
    class_dict = tr_gen.class_indices
    classes = list(class_dict.keys())
    images, labels = next(ts_gen)
    
    plt.figure(figsize=(15, 15))
    
    for i, (image, label) in enumerate(zip(images, labels)):
        plt.subplot(4,4, i + 1)
        plt.imshow(image)
        class_name = classes[np.argmax(label)]
        plt.title(class_name, color='k', fontsize=15)
    
    return plt.show();



# Define function to create model
def create_model(base_model, LEARNING_RATE):
    model = Sequential([
        base_model,
        Flatten(),
        Dropout(rate= 0.3),
        Dense(128, activation= 'relu'),
        Dropout(rate= 0.25),
        Dense(4, activation= 'softmax')
    ])
    
    model.compile(Adamax(learning_rate=LEARNING_RATE),
                  loss= 'categorical_crossentropy',
                  metrics= ['accuracy',
                            Precision(),
                            Recall()])
    
    summary = model.summary()
    
    return model, summary



# Define function to train model
def train_model(tr_gen, vd_gen):
    hist = model.fit(tr_gen,
                 epochs=10,
                 validation_data=vd_gen,
                 shuffle= False)

    return hist



# Define function to show performance
def show_performance(hist):
    tr_acc = hist.history['accuracy']
    tr_loss = hist.history['loss']
    tr_per = hist.history['precision']
    tr_recall = hist.history['recall']
    val_acc = hist.history['val_accuracy']
    val_loss = hist.history['val_loss']
    val_per = hist.history['val_precision']
    val_recall = hist.history['val_recall']
    
    index_loss = np.argmin(val_loss)
    val_lowest = val_loss[index_loss]
    index_acc = np.argmax(val_acc)
    acc_highest = val_acc[index_acc]
    index_precision = np.argmax(val_per)
    per_highest = val_per[index_precision]
    index_recall = np.argmax(val_recall)
    recall_highest = val_recall[index_recall]
    
    Epochs = [i + 1 for i in range(len(tr_acc))]
    loss_label = f'Best epoch = {str(index_loss + 1)}'
    acc_label = f'Best epoch = {str(index_acc + 1)}'
    per_label = f'Best epoch = {str(index_precision + 1)}'
    recall_label = f'Best epoch = {str(index_recall + 1)}'
        
    plt.figure(figsize=(20, 12))
    plt.style.use('fivethirtyeight')
        
    plt.subplot(2, 2, 1)
    plt.plot(Epochs, tr_loss, 'r', label='Training loss')
    plt.plot(Epochs, val_loss, 'g', label='Validation loss')
    plt.scatter(index_loss + 1, val_lowest, s=150, c='blue', label=loss_label)
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 2, 2)
    plt.plot(Epochs, tr_acc, 'r', label='Training Accuracy')
    plt.plot(Epochs, val_acc, 'g', label='Validation Accuracy')
    plt.scatter(index_acc + 1, acc_highest, s=150, c='blue', label=acc_label)
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 2, 3)
    plt.plot(Epochs, tr_per, 'r', label='Precision')
    plt.plot(Epochs, val_per, 'g', label='Validation Precision')
    plt.scatter(index_precision + 1, per_highest, s=150, c='blue', label=per_label)
    plt.title('Precision and Validation Precision')
    plt.xlabel('Epochs')
    plt.ylabel('Precision')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 2, 4)
    plt.plot(Epochs, tr_recall, 'r', label='Recall')
    plt.plot(Epochs, val_recall, 'g', label='Validation Recall')
    plt.scatter(index_recall + 1, recall_highest, s=150, c='blue', label=recall_label)
    plt.title('Recall and Validation Recall')
    plt.xlabel('Epochs')
    plt.ylabel('Recall')
    plt.legend()
    plt.grid(True)
    
    plt.suptitle('Model Training Metrics Over Epochs', fontsize=16)
    
    return plt.show();



# Define function to eveluate results
def evaluate_results(tr_gen, vd_gen, ts_gen):
    train_score = model.evaluate(tr_gen, verbose=1)
    valid_score = model.evaluate(vd_gen, verbose=1)
    test_score = model.evaluate(ts_gen, verbose=1)
    
    print(f"\nTrain Loss: {train_score[0]:.4f}")
    print(f"Train Accuracy: {train_score[1]*100:.2f}%")
    print('-' * 25)
    print(f"Validation Loss: {valid_score[0]:.4f}")
    print(f"Validation Accuracy: {valid_score[1]*100:.2f}%")
    print('-' * 25)
    print(f"Test Loss: {test_score[0]:.4f}")
    print(f"Test Accuracy: {test_score[1]*100:.2f}%")



    # Define the download_image function.
def download_image(url):
    with request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img



    # Define the prepare_image function.
def prepare_image(img, target_size):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size, Image.NEAREST)
    return img



    # Define the pre_process function.
def pre_process(x):
    return x / 255.0