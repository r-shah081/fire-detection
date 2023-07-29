import os
import tensorflow as tf
import matplotlib as plt

def count_files_in_folders(parent_folder) -> dict:
    """Count files in given folder. Return dictionary with folder and count."""
    folder_names = os.listdir(parent_folder)
    file_count = {}

    for folder_name in folder_names:
        folder_path = os.path.join(parent_folder, folder_name)
        if os.path.exists(folder_path) and os.path.isdir(folder_path):
            file_count[folder_name] = len(
                [file for file in os.listdir(folder_path) if os.path.isfile(
                    os.path.join(folder_path, file)
                    )
                ]
            )
        else:
            file_count[folder_name] = 0

    return file_count


def load_and_prep_image(filename, img_shape = 300):
    """Funtion to read image and transform image to tensor."""
    img = tf.io.read_file(filename) #read image
    img = tf.image.decode_image(img) # decode the image to a tensor
    img = tf.image.resize(img, size = [img_shape, img_shape]) # resize the image
    return img


def pred_and_plot(model, filename, class_names):
    """Funtion to read image and give desired output with image.

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

