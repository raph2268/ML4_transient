# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt
import sys,os

from sklearn.model_selection import train_test_split

from keras_tuner import HyperModel
from keras_tuner.tuners import BayesianOptimization
from keras_tuner.tuners import Hyperband
from tensorflow.keras.callbacks import EarlyStopping

import pandas as pd

from mpl_toolkits.axes_grid1 import make_axes_locatable

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #Permet d'enlever les Warning tensorflow

# os.environ['PYTHONHASHSEED'] = '0'
# os.environ['TF_CUDNN_CONVOLUTION_BWD_FILTER_ALGO_DETERMINISTIC']='1'
# print(hash("keras"))

np.random.seed(1) # NumPy
import random
random.seed(2) # Python
tf.random.set_seed(3) # Tensorflow 

from utils import rundir

class CNNHyperModel(HyperModel):
    def __init__(self, input_shape, num_classes, executions_per_trial, max_trials, max_epochs, x_train, x_test, y_train, y_test, batch_size=128, trial_directory=None):
        """
        Initialize the class instance.

        Args:
            input_shape (tuple): The shape of the input data, of image.
            num_classes (int): The number of classes.
            executions_per_trial (int): The number of executions per trial during hyperparameter optimization.
            max_trials (int): The maximum number of trials to run during hyperparameter optimization.
            max_epochs (int): The maximum number of epochs to run during hyperband optimization.
            x_train (ndarray): The training input data.
            x_test (ndarray): The testing input data.
            y_train (ndarray): The training target labels.
            y_test (ndarray): The testing target labels.
            batch_size (int, optional): The batch size for training the model. Defaults to 128.
            trial_directory (str, optional): The directory where trial information will be stored. Defaults to None.
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.executions_per_trial = executions_per_trial
        self.max_trials = max_trials
        self.max_epochs = max_epochs
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.batch_size = batch_size
        self.trial_directory=trial_directory
        self.history=[]
        if trial_directory is not None:
            self.path_trial_directory=os.path.join(rundir, trial_directory)

    def build(self, hp):
        """
        Build the model architecture with hyperparameters.

        Args:
            hp (kerastuner.HyperParameters): Hyperparameters object provided by the Keras Tuner.

        Returns:
            keras.models.Sequential: The compiled model.

        """        
        seed1 = keras.initializers.glorot_uniform(seed=1)

        model = keras.Sequential()
        
        model.add(keras.layers.Input(self.input_shape))
        
        model.add(keras.layers.Conv2D(
            filters=hp.Int(
                    'filters_1',
                    min_value=2,
                    max_value=32,#64,
                    default=8
            ), 
            kernel_size=(3,3), activation='relu', kernel_initializer=seed1))
        
        model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))
        
        model.add(keras.layers.Dropout(
                rate=hp.Float(
                    'dropout_1',
                    min_value=0.1,
                    max_value=0.5,
                    default=0.2,
                    step=0.05
                )
            )
        )
        
        model.add(keras.layers.Conv2D(
            filters=hp.Int(
                    'filters_2',
                    min_value=2,
                    max_value=64,
                    default=16
            ),
            kernel_size=(3,3), activation='relu', kernel_initializer=seed1))
        
        model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))
        
        model.add(keras.layers.Dropout(
                rate=hp.Float(
                    'dropout_2',
                    min_value=0.0,
                    max_value=0.5,
                    default=0.2,
                    step=0.05
                )
            )
        )
        
        model.add(keras.layers.Flatten())
        
        model.add(
            keras.layers.Dense(
                units=hp.Int(
                    'units',
                    min_value=16,
                    max_value=2048,
                    step=64,
                    default=128
                ),
                activation='relu',
                kernel_initializer=seed1
            )
        )
        
        model.add(
            keras.layers.Dropout(
                rate=hp.Float(
                    'dropout_3',
                    min_value=0.0,
                    max_value=0.5,
                    default=0.3,
                    step=0.05
                )
            )
        )
        
        model.add(keras.layers.Dense(self.num_classes, activation='softmax',kernel_initializer=seed1))

        hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
        
        
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])
        
        return model


    def Entrainement_bayes(self):
        early_stop = EarlyStopping(monitor='accuracy', patience=5, verbose=1, mode='max', restore_best_weights=True)

        if self.trial_directory is not None:
            if not os.path.exists(self.trial_directory):
                os.makedirs(self.trial_directory)

            tuner_B = BayesianOptimization(
                self.build,
                objective='val_accuracy',
                executions_per_trial=self.executions_per_trial,
                overwrite=True,
                max_trials=self.max_trials,
                directory=self.trial_directory)
            print(tuner_B.summary())
        else:
            tuner_B = BayesianOptimization(
                self.build,
                objective='val_accuracy',
                executions_per_trial=self.executions_per_trial,
                overwrite=True,
                max_trials=self.max_trials)

        tuner_B.search(self.x_train, self.y_train, 
                       epochs=self.max_epochs,
                       validation_data=(self.x_test, self.y_test), 
                       callbacks=[early_stop],
                       batch_size=self.batch_size,
                       verbose=1)

        # Iterate through all trials and retrain to capture histories
        for trial in tuner_B.oracle.get_best_trials(num_trials=self.max_trials):
            hp = trial.hyperparameters
            model = self.build(hp)
            history = model.fit(self.x_train, self.y_train, 
                                epochs=self.max_epochs, 
                                batch_size=self.batch_size,
                                validation_data=(self.x_test, self.y_test),
                                verbose=1)
            # Save the history of this trial
            self.history.append(history.history)

        best_model = tuner_B.get_best_models(num_models=1)[0]
        loss, accuracy = best_model.evaluate(self.x_test, self.y_test)

        val_acc_per_epoch = history.history['val_accuracy']
        best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
        history_retrain = best_model.fit(self.x_train, self.y_train, 
                                         epochs=best_epoch, 
                                         batch_size=self.batch_size, 
                                         validation_data=(self.x_test, self.y_test),
                                         validation_split=0.2,
                                         verbose=0)


        return best_model, self.history