# Constants
EPOCHS = 100
TRAIN_BATCH_SIZE = 4
VAL_BATCH_SIZE = 4
TARGET_IMG_SIZE = (150, 150)
INPUT_SHAPE = (150, 150, 3)
IMG_HEIGHT = 150
IMG_WIDHT = 150
BATCH_SIZE = 32

# File paths
DATASET = 'C:\\Users\\roman\\OneDrive - McMaster University\\Desktop\\CPS\\fire-detection\\data\\'
GOOD_IMG_PATH = DATASET + '1'
BAD_IMG_PATH = DATASET + '0'
FIRE_IMGPATH = GOOD_IMG_PATH + '\\1.jpg'
RAND_IMGPATH = BAD_IMG_PATH + '\\1.jpg'
CHECKPOINT_FILEPATH = 'C:\\Users\\roman\\OneDrive - McMaster University\\Desktop\\CPS\\fire-detection\\temp\\checkpoint'
SAVE_KERAS_MODEL = 'C:\\Users\\roman\\OneDrive - McMaster University\\Desktop\\CPS\\fire-detection\\model\\fire_detection_model.keras'
SAVE_H5_MODEL = "'C:\\Users\\roman\\OneDrive - McMaster University\\Desktop\\CPS\\fire-detection\\model\\fire_detection_model.h5'"