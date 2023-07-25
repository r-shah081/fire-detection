__doc__ = """"""

# Import Python packages
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# Import installed libraries
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, BatchNormalization, Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.image as mimg
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


# reading data that is required to be trained on
dataset = './fire-detection/data'
good_img_path = dataset + '/good'
bad_img_path = dataset + '/bad'

# print(os.listdir(good_img_path))
# print(os.listdir(bad_img_path))

# Using tensorflow's ImageDataGenerator to prepare the image-data for training
train_gen = ImageDataGenerator(width_shift_range = 0.5, height_shift_range = 0.5,
            validation_split = 0.2,
)
train_set = train_gen.flow_from_directory(dataset, target_size = (300, 300),
            class_mode = 'binary', subset = 'training', batch_size = 8,
)
val_set = train_gen.flow_from_directory(dataset, target_size = (300,300),
            class_mode = 'binary', subset = 'validation', batch_size = 8,
)

# Using pre-trained Resnet-50 layers model to train on our fire-dataset
# here we are setting include_top as False, as we will add our own dense layers
# after resnet 50 last layer
pretrained_resnet50 = tf.keras.applications.ResNet50(
            include_top = False, input_shape = (300, 300, 3),
            pooling = 'avg', classes = 100, weights = 'imagenet',
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

# Using tensorflow's learning-rate-scheduler to change learning rate at each epoch
# this will help us to find the best learning rate for our model
callback_lr = tf.keras.callbacks.LearningRateScheduler(
            lambda epoch: 1e-8 * 10**(epoch/20)
)

# Using tensorflow's ModelCheckpoint to save best model having less validation loss
# modelcheck_lr = tf.keras.callbacks.ModelCheckpoint(
#     filepath = '/kaggle/working/final_model.h5', monitor = 'val_loss',
#     save_best_only = True,
# )

# Using Adam's optimizer to optimize our model to better learn on our dataset
model.compile(optimizer = tf.keras.optimizers.Adam(),
            loss = 'binary_crossentropy',
            metrics = 'accuracy',
)

# Now time to train our model on fire dataset
model_hist = model.fit(train_set, validation_data = val_set,
        epochs = 100, callbacks = [callback_lr,] # modelcheck_lr]
)

# Creating graph to visualzie how our model performed at different learning rate
# and its loss.
lrs = 1e-8 * (10 ** (np.arange(100) / 20))

# Set the figure size
plt.figure(figsize=(10, 6))

# Set the grid
plt.grid(True)

# Plot the loss in log scale
plt.semilogx(lrs, model_hist.history["loss"])

# Increase the tickmarks size
plt.tick_params('both', length=10, width=1, which='both')

# Set the plot boundaries
plt.axis([1e-8, 1e-3, 0, 1])

# Downloading our best model that was picked up by Model-Checkpoint
best_model = tf.keras.models.load_model('/kaggle/working/final_model.h5')

# now checking the final evalutaion of our model
best_model.evaluate(validation_set)

# Funtion to read image and transform image to tensor 
def load_and_prep_image(filename, img_shape = 300):
    img = tf.io.read_file(filename) #read image
    img = tf.image.decode_image(img) # decode the image to a tensor
    img = tf.image.resize(img, size = [img_shape, img_shape]) # resize the image
    return img
# Funtion to read image and give desired output with image
def pred_and_plot(model, filename, class_names):
    """
    Imports an image located at filename, makes a prediction on it with
    a trained model and plots the image with the predicted class as the title.
    """
    # Import the target image and preprocess it
    img = load_and_prep_image(filename)
    
    # Make a prediction
    pred = model.predict(tf.expand_dims(img, axis=0))
    
    if len(pred[0]) > 1: # check for multi-class
        pred_class = class_names[pred.argmax()] # if more than one output, take the max
    else:
        pred_class = class_names[int(tf.round(pred)[0][0])] # if only one output, round

    # Plot the image and predicted class
    sh_image = plt.imread(filename)
    plt.imshow(sh_image)
    plt.title(f"Prediction: {pred_class}")
    plt.axis(False)

    # specifying path to sample image from list of test images.
sample_img = '/kaggle/input/fireornot-firetestset/test/AdobeStock_249454423-scaled.jpeg'

# predefining class names so not to confuse with the output
class_names = ['Not-fire','Fire']

# reading the input and checking the output
pred_and_plot(best_model,sample_img,class_names)