# !/usr/bin/env python3

# Importing standard libraries
import os
from datetime import datetime
import numpy as np
import tensorflow as tf

# Keras Deep Learning modules
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.keras.applications.densenet import DenseNet121

# Optimizers
from tensorflow.keras.optimizers import Adam

# Import callbacks
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import ReduceLROnPlateau

# Sklearn modules
from sklearn.utils import class_weight
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from training.srh_model import srh_model
from preprocessing.preprocess import cnn_preprocessing

##############################
img_rows = 300
img_cols = 300
img_channels = 3

class_names = ['carcinoma', 'lymphoma', 'glioma']
total_classes = len(class_names)

def find_pair_factors_for_CNN(x):
    """
    Function to match batch size and iterations for the validation generator
    """
    pairs = []
    for i in range(2, 150):
        test = x/i
        if i * int(test) == x:
            pairs.append((i, int(test)))
    best_pair = pairs[-1]
    assert len(pairs) >= 1, "No pairs found"
    print(best_pair)
    return best_pair

def validation_batch_steps(directory):
    counter = 0
    for roots, dirs, files in os.walk(directory):
        for file in files:
            counter += 1
    return find_pair_factors_for_CNN(counter)

def metric_a(y_true, y_pred):
    classes = tf.argmax(y_true, axis=0)
    class_true = tf.boolean_mask(y_true,tf.equal(classes, 0))
    class_pred = tf.boolean_mask(y_pred,tf.equal(classes, 0))
    return tf.keras.metrics.categorical_accuracy(class_true, class_pred)

def metric_b(y_true, y_pred):
    classes = tf.argmax(y_true, axis=0)
    class_true = tf.boolean_mask(y_true,tf.equal(classes, 1))
    class_pred = tf.boolean_mask(y_pred,tf.equal(classes, 1))
    return tf.keras.metrics.categorical_accuracy(class_true, class_pred)

def metric_c(y_true, y_pred):
    classes = tf.argmax(y_true, axis=0)
    class_true = tf.boolean_mask(y_true,tf.equal(classes, 2))
    class_pred = tf.boolean_mask(y_pred,tf.equal(classes, 2))
    return tf.keras.metrics.categorical_accuracy(class_true, class_pred)

def generator_prediction(model, generator):
    # foward pass on dataset
    cnn_predictions = model.predict(generator, steps=val_steps, verbose=1)
    cnn_predict_1d = np.argmax(cnn_predictions, axis = 1)
    index_validation = validation_generator.classes
    # Overall accuracy
    print(accuracy_score(index_validation, cnn_predict_1d))
    print(confusion_matrix(index_validation, cnn_predict_1d))
    print(classification_report(index_validation, cnn_predict_1d))


if __name__ == '__main__':
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
	
    validation_dir = '/media/chengjia/cheng_hd/data/frozen/valid'
    
    # find the best combination of steps and batch size for the validation generator
    val_batch, val_steps = validation_batch_steps(validation_dir)
    validation_generator = ImageDataGenerator(
        horizontal_flip=False,
        vertical_flip=False,
        preprocessing_function = cnn_preprocessing,
        data_format = "channels_last"
    ).flow_from_directory(
        directory = validation_dir,
        target_size = (img_rows, img_cols), 
        color_mode = 'rgb',
        classes = class_names, 
        class_mode = 'categorical',
        batch_size = val_batch, 
        shuffle = False)
    
    model = load_model(
        'modele-4.hdf5',
        custom_objects={
            'metric_a':metric_a,
            'metric_b':metric_b,
            'metric_c':metric_c
       }
   )
    generator_prediction(model, validation_generator)
