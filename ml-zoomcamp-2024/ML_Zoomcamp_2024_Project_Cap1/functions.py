# Import required modules
import os
import numpy as np
import pandas as pd
import tensorflow as tf



# Define function to create dataframes
def create_dataframe(base_path):

    # Define label.
    labels=os.listdir(base_path)

    # Gather data
    data = []
    for label in labels:
        folder_path = os.path.join(base_path, label)
        image_files = os.listdir(folder_path)
        
        for image in image_files:
            image_path = os.path.join(folder_path, image)
            data.append({'image_path': image_path, 'label': label})

    return pd.DataFrame(data)
    


# Define function to create data
def create_data(BATCH_SIZE):
    
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1.0/255,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1.0/255
    )
    
    train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        x_col='image_path',  
        y_col='label',       
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical'  
    )
    
    test_generator = test_datagen.flow_from_dataframe(
        dataframe=test_df,
        x_col='image_path',
        y_col='label',
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )

    return train_generator, test_generator



# Define function to load and fit model
def load_fit(BASE_MODEL, DROP_RATE):
    
    # Load the model pretrained on ImageNet without the top layers
    base_model = BASE_MODEL
    base_model.trainable = True
    
    # Build custom model on top of base model.
    model = base_model.output
    model = tf.keras.layers.GlobalAveragePooling2D()(model)
    model = tf.keras.layers.Dense(1024, activation='relu')(model)
    model = tf.keras.layers.Dropout(rate=DROP_RATE)(model)
    model = tf.keras.layers.Dense(2, activation='softmax')(model)
    model = tf.keras.models.Model(inputs=base_model.input, outputs=model)
    
    # Compile model
    model.compile(optimizer=OPTIMIZER, loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Fit model
    history = model.fit(
        train_generator,
        validation_data=test_generator,
        epochs=EP0CHS,
    )

    return model