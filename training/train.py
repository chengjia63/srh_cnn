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

# Define a custom loss function if neeeded
def focal_loss(y_true, y_pred):
    gamma = 2.0
    alpha = 0.25
    pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
    pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
    return -K.sum(alpha * K.pow(1. - pt_1, gamma) * 
        K.log(pt_1))-K.sum((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0))

def generator_prediction(model, generator):
    # foward pass on dataset
    cnn_predictions = model.predict(generator, steps=val_steps, verbose=1)
    cnn_predict_1d = np.argmax(cnn_predictions, axis = 1)
    index_validation = validation_generator.classes
    # Overall accuracy
    print(accuracy_score(index_validation, cnn_predict_1d))
    print(confusion_matrix(index_validation, cnn_predict_1d))
    print(classification_report(index_validation, cnn_predict_1d))

def save_model(model, name):
    model.save(name + ".hdf5")

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

if __name__ == '__main__':
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
	
    # Train/validation directories
    training_dir = '/media/chengjia/cheng_hd/data/frozen/train'
    validation_dir = '/media/chengjia/cheng_hd/data/frozen/valid'

    # instantiate train generator
    train_generator = ImageDataGenerator(
        horizontal_flip=False,
        vertical_flip=False,
        preprocessing_function = cnn_preprocessing,
        data_format = "channels_last"
    ).flow_from_directory(
        directory = training_dir,
        target_size = (img_rows, img_cols), 
        color_mode = 'rgb', 
        classes = class_names, 
        class_mode = 'categorical',
        batch_size = 8, 
        shuffle = True)

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
        shuffle = True)

    # find the best combination of steps and batch size for the validation generator
    test_generator = ImageDataGenerator(
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

    # instantiate model
    parallel_model = srh_model(backbone=DenseNet121, gpu_num=1, weights=False)

    # compile model with Adam optimizer
    ADAM = Adam(
        lr=1e-6,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-08,
        decay=0.0)

    parallel_model.compile(
        optimizer=ADAM,
        loss="categorical_crossentropy",
        metrics =['accuracy', metric_a, metric_b, metric_c]) 

    # specify callbacks for training
    early_stop = EarlyStopping(
        monitor='val_accuracy',
        min_delta = 0.05,
        patience=10,
        mode = 'auto')

    checkpoint = ModelCheckpoint(
        'Final_weights.{epoch:02d}-{val_accuracy:.2f}.hdf5',
        monitor='val_accuracy',
        verbose=0,
        save_best_only=True,
        save_weights_only=True,
        mode='auto',
        save_freq='epoch')

    reduce_LR = ReduceLROnPlateau(
        monitor='val_accuracy',
        factor=0.5,
        patience=10,
        verbose=1,
        mode='auto',
        cooldown=0,
        min_lr=0)

    logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(log_dir=logdir)

    callbacks_list = [tensorboard_callback, checkpoint, reduce_LR, tensorboard_callback]

    # class weights to correct for class imbalance
    class_weight = class_weight.compute_class_weight('balanced', 
        np.unique(train_generator.classes), train_generator.classes)
    weight_dict = dict(zip(list(range(0,total_classes)), class_weight))
    
    # fit generator for model training
    parallel_model.fit(
        train_generator,
        steps_per_epoch=500, #10000,
        epochs=200,
        shuffle=True,
        class_weight=weight_dict,
        max_queue_size=30,
        workers=1,
        initial_epoch=0,
        verbose=1,
        validation_data=validation_generator,
        validation_steps=val_steps,
        callbacks=callbacks_list
    )

    save_model(parallel_model, "model")
    # validation set prediction and save model
    generator_prediction(parallel_model, test_generator)
