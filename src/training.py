__doc__ = """"""

# Import Python packages
import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# Import installed libraries
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, BatchNormalization, Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.image as mimg
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import keras


import parameters as params
import auxiliary as aux


if __name__ == "__main__":
    print(tf.version.VERSION)

    #%% Dataset breakdown
    file_info : dict = aux.count_files_in_folders(params.DATASET)
    for folder, count in file_info.items():
        print(f'[INFO] : Found {count} images in {folder} subdir')

    # Using tensorflow's ImageDataGenerator to prepare the image-data for training
    # train_gen = ImageDataGenerator(
    #                 width_shift_range = 0.5, height_shift_range = 0.5,
    #                 validation_split = 0.2,
    # )
    # train_set = train_gen.flow_from_directory(params.DATASET, target_size = params.TARGET_IMG_SIZE,
    #                 class_mode = 'binary', subset = 'training',
    #                 batch_size = params.TRAIN_BATCH_SIZE,
    # )
    # val_set = train_gen.flow_from_directory(params.DATASET, target_size = params.TARGET_IMG_SIZE,
    #                 class_mode = 'binary', subset = 'validation',
    #                 batch_size = params.VAL_BATCH_SIZE,
    # )

    train_ds = tf.keras.utils.image_dataset_from_directory(
        params.DATASET, validation_split=0.2,
        image_size=(params.IMG_HEIGHT, params.IMG_WIDHT), 
        batch_size=params.TRAIN_BATCH_SIZE, subset="training",
        seed=20,)
    
    val_ds = tf.keras.utils.image_dataset_from_directory(
        params.DATASET, validation_split=0.2,
        image_size=(params.IMG_HEIGHT, params.IMG_WIDHT),
        batch_size=params.VAL_BATCH_SIZE, subset="validation",
        seed=20,)

    class_names = train_ds.class_names
    print(class_names)

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    # Using pre-trained Resnet-50 layers model to train on our fire-dataset
    # here we are setting include_top as False, as we will add our own dense layers
    # after resnet 50 last layer
    pretrained_resnet50 = tf.keras.applications.ResNet50(
                include_top = False, input_shape = params.INPUT_SHAPE,
                pooling = 'avg', classes = 2, weights = 'imagenet',
    )

    # Here we want last 10 layers to be trainable so freezing first 40 layers
    x = 0
    for layer in pretrained_resnet50.layers:
        layer.trainable = False
        x+=1
        if x == 39:
            break

    # Adding extra Dense layers after Resnet 50 last layer,
    # we do this to increase our models capability to categorise image as having
    # fire or not
    model = Sequential()
    model.add(pretrained_resnet50)
    model.add(Dense(2048, activation='relu'))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(8000, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4000, activation='relu'))
    model.add(Dense(2000, activation='relu'))
    model.add(Dense(1000, activation='relu'))
    model.add(Dense(500, activation='relu'))
    model.add(Dense(250, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(1, activation = 'sigmoid'))
    model.summary()

    #%% Callbacks
    # Using tensorflow's learning-rate-scheduler to change learning rate at each epoch
    # this will help us to find the best learning rate for our model
    learning_rate_callback = tf.keras.callbacks.LearningRateScheduler(
                lambda epoch: 1e-8 * 10**(epoch/20)
    )
    # Using tensorflow's ModelCheckpoint to save best model having less validation loss
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath = params.CHECKPOINT_FILEPATH, monitor = 'val_loss',
        save_best_only = True,
    )

    #%% Compile
    # Using Adam optimizer to optimize our model to better learn on our dataset
    model.compile(optimizer = tf.keras.optimizers.Adam(),
                loss = 'binary_crossentropy',
                metrics = 'accuracy',
    )

    #%% Train model
    # Now time to train our model on fire dataset
    model_hist = model.fit(train_ds, validation_data = val_ds,
            epochs = params.EPOCHS,
            callbacks = [learning_rate_callback,  model_checkpoint_callback,],
    )

    # Save model parameters
    try:
        model.save(params.SAVE_H5_MODEL, save_format = 'h5')
    except Exception:
        model.save(params.SAVE_KERAS_MODEL)


    #%% Graphs
    # Creating graph to visualzie how our model performed at different learning rate
    # and its loss.
    lrs = 1e-8 * (10 ** (np.arange(100) / 20))
    plt.figure(figsize=(10, 6)) # Set the figure size
    plt.grid(True) # Set the grid
    plt.semilogx(lrs, model_hist.history["loss"]) # Plot the loss in log scale
    plt.tick_params('both', length=10, width=1, which='both') # Increase tickmark size
    plt.axis([1e-8, 1e-3, 0, 1]) # Set the plot boundaries
    plt.show()