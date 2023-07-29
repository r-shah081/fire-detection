import keras

from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, BatchNormalization, Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import parameters as params

# To load model
def get_model(model_path):
    loaded_model = keras.models.load_model(model_path)

def create_model():
    model = Sequential()
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
    return model

if __name__ == "__main__":
    # loaded_model = get_model(model_path='C:/Users/roman/OneDrive - McMaster University/Desktop/CPS/fire-detection/model/fire_detection_model.keras')
    
    # print(loaded_model)

    # # now checking the final evalutaion of our model
    # loaded_model.evaluate(val_set)

    # # predefining class names so not to confuse with the output
    # class_names = ['Not-Fire', 'Fire']

    # # reading the input and checking the output
    # pred_and_plot(loaded_model, FIRE_IMGPATH, class_names)
    model = create_model()
    model.load_weights('temp/checkpoint/saved_model.pb')
    loss, acc = model.evaluate(params.FIRE_IMGPATH, verbose=2)
    print(acc)
