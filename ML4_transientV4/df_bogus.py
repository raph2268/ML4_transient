import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys,os
from importlib import reload

from sklearn.mixture import GaussianMixture
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay

from keras_tuner import HyperModel
from keras_tuner.tuners import BayesianOptimization

import pandas as pd
pd.set_option('display.max_columns', None)

from mpl_toolkits.axes_grid1 import make_axes_locatable

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #Permet d'enlever les Warning tensorflow

np.random.seed(1) # NumPy
import random
random.seed(2) # Python
tf.random.set_seed(3) # Tensorflow 

sys.path.append('..')
from torch.utils.data import DataLoader, TensorDataset


from utils import datadir, rundir, embeddable_image
from CNN_model import CNNHyperModel
import torch
import pickle



def df_proba(df_test, x_test, y_test, model, path=None):
    '''
    Produces a table containing the prediction made by the model for each image
    Also calculates the number of errors and the detection efficiency of the model
        
    df_test : dataframe containing the various information associated with the images, in particular the image label
    x_test : matrix containing test data
    y_test : array containing the test labels associated with the test data
    model : model used to make the prediction
    path : path to model
        
    return :
        
    df_test = dataframe containing the information associated with the images as well as the predictions made by the model
        
    '''

    #new label for bogus/transcients
    df_test['true_class']=['Real'if (a==0) else 'Injected' for a in df_test['y_test']]
    print('Shape of the set:', x_test.shape)
    
    #loading model
    if path is not None:
        load_model = tf.keras.models.load_model(f'{path}/{model}.h5')
        
    else:
        
        load_model = tf.keras.models.load_model(f'models/{model}.h5')
        
    score = load_model.evaluate(x_test, y_test, verbose=0, batch_size=None, steps=None)

    #print(f'Test loss     : {score[0]:4.4f}')
    print(f'Test accuracy : {score[1]:4.4f}')

    #model prediction 
    
    y_sigmoid = load_model.predict(x_test) #probability table y_sigmoid[:,0] = prob bogus y_sigmoid[:,1] = prob transcient 
    #Probability of the model belonging to transcient class
    df_test[f'proba_transient_{model}'] = y_sigmoid[:,1]
    df_test[f'proba_transient_perc_{model}'] = (y_sigmoid[:,1]*100).astype(int)

    #Probability of the model belonging to bogus class
    df_test[f'proba_bogus_{model}'] = y_sigmoid[:,0]
    df_test[f'proba_bogus_perc_{model}'] = (y_sigmoid[:,0]*100).astype(int)


    df_test[f'y_pred_{model}'] = np.argmax(y_sigmoid, axis=-1) #Retrieve the class it belongs to

    df_test[f'bogus_transient_pred_{model}']=['bogus' if (a==0) else 'transient' for a in df_test[f'y_pred_{model}']]
    
    
    #Compute the amount of error
    error_bogus=df_test[f'y_pred_{model}'].loc[ (df_test[f'y_pred_{model}']==1) & (df_test.y_test==0)].index.to_list() 
    error_transient=df_test[f'y_pred_{model}'].loc[ (df_test[f'y_pred_{model}']==0) & (df_test.y_test==1)].index.to_list() 
    print('potential real transcient',len(error_bogus))
    
    df_test['false_positive'] = ((df_test[f'y_pred_{model}'] == 1) & (df_test['y_test'] == 0)).astype(int)
    df_test['false_negative'] = ((df_test[f'y_pred_{model}'] == 0) & (df_test['y_test'] == 1)).astype(int)
    df_test['true_positive'] = ((df_test[f'y_pred_{model}'] == 1) & (df_test['y_test'] == 1)).astype(int)
    df_test['true_negative'] = ((df_test[f'y_pred_{model}'] == 0) & (df_test['y_test'] == 0)).astype(int)
    err =  len(error_bogus)+len(error_transient) #Compute the amount of error
    eff = (len(x_test)-err)/len(x_test) #Compute the efficiency
    
    print(f'Test efficiency : {eff:4.4f}')
    
    
    #probabilities of belonging to the class 1 when the true class is 0 (False positive)
    df_test[f'class_proba_bogus_{model}'] = df_test.y_test.mask(((df_test.y_test==0) & (df_test[f'proba_transient_perc_{model}']>90)), other='>90%')
    
    
    
    df_test[f'class_proba_bogus_{model}'] =df_test[f'class_proba_bogus_{model}'].mask(((df_test[f'class_proba_bogus_{model}']==0)& (df_test[f'proba_transient_perc_{model}']>80)), other='>80%')
    
    df_test[f'class_proba_bogus_{model}'] = df_test[f'class_proba_bogus_{model}'].mask(((df_test[f'class_proba_bogus_{model}']==0) & (df_test[f'proba_transient_perc_{model}']>70)), other='>70%')
    
    df_test[f'class_proba_bogus_{model}'] = df_test[f'class_proba_bogus_{model}'].mask(((df_test[f'class_proba_bogus_{model}']==0) & (df_test[f'proba_transient_perc_{model}']>60)), other='>60%')
    
    df_test[f'class_proba_bogus_{model}'] = df_test[f'class_proba_bogus_{model}'].mask(((df_test[f'class_proba_bogus_{model}']==0) & (df_test[f'proba_transient_perc_{model}']>50)), other='>50%')
    
    df_test[f'class_proba_bogus_{model}'] = df_test[f'class_proba_bogus_{model}'].mask(((df_test[f'class_proba_bogus_{model}']==0) & (df_test[f'proba_transient_perc_{model}']>40)), other='>40%')
    
    df_test[f'class_proba_bogus_{model}'] = df_test[f'class_proba_bogus_{model}'].mask(((df_test[f'class_proba_bogus_{model}']==0) & (df_test[f'proba_transient_perc_{model}']>30)), other='>30%')
    
    df_test[f'class_proba_bogus_{model}'] = df_test[f'class_proba_bogus_{model}'].mask(((df_test[f'class_proba_bogus_{model}']==0) & (df_test[f'proba_transient_perc_{model}']>20)), other='>20%')
    
    df_test[f'class_proba_bogus_{model}'] = df_test[f'class_proba_bogus_{model}'].mask(((df_test[f'class_proba_bogus_{model}']==0) & (df_test[f'proba_transient_perc_{model}']>10)), other='>10%')
    
    df_test[f'class_proba_bogus_{model}'] = df_test[f'class_proba_bogus_{model}'].mask((df_test[f'class_proba_bogus_{model}']==1) , other='Injection')
    
    df_test[f'class_proba_bogus_{model}'] = df_test[f'class_proba_bogus_{model}'].mask(((df_test[f'class_proba_bogus_{model}']==0) & (df_test[f'proba_transient_perc_{model}']<=10)), other='<10%')
    
    
    #probabilities of belonging to the class 1 when the true class is 1 (True positive)
    df_test[f'class_proba_transient_{model}'] = df_test.y_test.mask(((df_test.y_test==1) & (df_test[f'proba_bogus_perc_{model}']>90)), other='<90%')
    df_test[f'class_proba_transient_{model}'] = df_test[f'class_proba_transient_{model}'].mask(((df_test[f'class_proba_transient_{model}']==1) & (df_test[f'proba_bogus_perc_{model}']>80)), other='<80%')
    
    df_test[f'class_proba_transient_{model}'] = df_test[f'class_proba_transient_{model}'].mask(((df_test[f'class_proba_transient_{model}']==1) & (df_test[f'proba_bogus_perc_{model}']>70)), other='<70%')
    
    df_test[f'class_proba_transient_{model}'] = df_test[f'class_proba_transient_{model}'].mask(((df_test[f'class_proba_transient_{model}']==1) & (df_test[f'proba_bogus_perc_{model}']>60)), other='<60%')
    
    df_test[f'class_proba_transient_{model}'] = df_test[f'class_proba_transient_{model}'].mask(((df_test[f'class_proba_transient_{model}']==1) & (df_test[f'proba_bogus_perc_{model}']>50)), other='<50%')
    
    df_test[f'class_proba_transient_{model}'] = df_test[f'class_proba_transient_{model}'].mask(((df_test[f'class_proba_transient_{model}']==1) & (df_test[f'proba_bogus_perc_{model}']>40)), other='<40%')
    
    df_test[f'class_proba_transient_{model}'] = df_test[f'class_proba_transient_{model}'].mask(((df_test[f'class_proba_transient_{model}']==1) & (df_test[f'proba_bogus_perc_{model}']>30)), other='<30%')
    
    df_test[f'class_proba_transient_{model}'] = df_test[f'class_proba_transient_{model}'].mask(((df_test[f'class_proba_transient_{model}']==1) & (df_test[f'proba_bogus_perc_{model}']>20)), other='<20%')
    
    df_test[f'class_proba_transient_{model}'] = df_test[f'class_proba_bogus_{model}'].mask((df_test[f'class_proba_transient_{model}']==0) , other='bogus')
    df_test[f'class_proba_transient_{model}'] = df_test[f'class_proba_bogus_{model}'].mask(((df_test[f'class_proba_transient_{model}']==1) & (df_test[f'proba_bogus_perc_{model}']<=20)), other='<10%')
    

    return df_test

def df_proba_pytorch(df_test, x_test, y_test, model, model_name, path=None, batch_size=None):
    '''
    Produces a table containing the prediction made by the model for each image.
    Also calculates the number of errors and the detection efficiency of the model.
        
    df_test : dataframe containing various information associated with the images, in particular the image label.
    x_test : matrix containing test data.
    y_test : array containing the test labels associated with the test data.
    model : model used to make the prediction.
    path : path to model.
        
    return :
    df_test = dataframe containing the information associated with the images as well as the predictions made by the model.
    '''

    print('Shape of the test set:', x_test.shape)

    model.eval()  # Set the model to evaluation mode

    # Convert x_test to the correct shape and then to a PyTorch tensor
    X_test = [np.transpose(i, (2, 0, 1)) for i in x_test]
    x_test_tensor = torch.tensor(np.array(X_test), dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    # Create a DataLoader for batch processing
    if batch_size : 
        test_dataset = TensorDataset(x_test_tensor, y_test_tensor)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    all_probabilities = []

    with torch.no_grad():  # Disable gradient calculation
        if batch_size : 
            for inputs, _ in test_loader:  # Only need input data (x), ignore labels for prediction
                inputs = inputs.to('cpu')  # Move to CPU (or 'cuda' if on GPU)

                # Forward pass through the model
                outputs = model(inputs).squeeze()

                # Apply sigmoid to get probabilities
                batch_probabilities = torch.sigmoid(outputs).cpu().numpy()
                all_probabilities.append(batch_probabilities)
        else : 
            # Make predictions
            outputs = model(x_test_tensor).squeeze()
            probabilities = torch.sigmoid(outputs).numpy()

    # Concatenate probabilities from all batches
    if batch_size : probabilities = np.concatenate(all_probabilities, axis=0)

    # Probability of the model belonging to transient class
    df_test['true_class'] = ['Real' if (a == 0) else 'Injected' for a in df_test['y_test']]
    df_test[f'proba_transient_{model_name}'] = probabilities
    df_test[f'proba_transient_perc_{model_name}'] = (probabilities * 100).astype(int)

    # Probability of the model belonging to bogus class
    df_test[f'proba_bogus_{model_name}'] = 1 - probabilities
    df_test[f'proba_bogus_perc_{model_name}'] = ((1 - probabilities) * 100).astype(int)

    # Retrieve the predicted class (bogus/transient)
    df_test[f'y_pred_{model_name}'] = (probabilities >= 0.5).astype(int)
    df_test[f'bogus_transient_pred_{model_name}'] = ['bogus' if (a == 0) else 'transient' for a in df_test[f'y_pred_{model_name}']]

    # Calculate number of errors
    error_bogus = df_test[f'y_pred_{model_name}'].loc[(df_test[f'y_pred_{model_name}'] == 1) & (df_test.y_test == 0)].index.to_list()
    error_transient = df_test[f'y_pred_{model_name}'].loc[(df_test[f'y_pred_{model_name}'] == 0) & (df_test.y_test == 1)].index.to_list()

    df_test['false_positive'] = ((df_test[f'y_pred_{model_name}'] == 1) & (df_test['y_test'] == 0)).astype(int)
    df_test['false_negative'] = ((df_test[f'y_pred_{model_name}'] == 0) & (df_test['y_test'] == 1)).astype(int)
    df_test['true_positive'] = ((df_test[f'y_pred_{model_name}'] == 1) & (df_test['y_test'] == 1)).astype(int)
    df_test['true_negative'] = ((df_test[f'y_pred_{model_name}'] == 0) & (df_test['y_test'] == 0)).astype(int)

    err = len(error_bogus) + len(error_transient)  # Calculate number of errors
    eff = (len(x_test) - err) / len(x_test)  # Calculate detection efficiency

    print(f'Test efficiency : {eff:4.4f}')

    # Classifying bogus probabilities based on percentage thresholds
    for thresh in [90, 80, 70, 60, 50, 40, 30, 20, 10]:
        df_test[f'class_proba_bogus_{model_name}'] = df_test.y_test.mask(
            (df_test.y_test == 0) & (df_test[f'proba_transient_perc_{model_name}'] > thresh), 
            other=f'>{thresh}%'
        )

    df_test[f'class_proba_bogus_{model_name}'] = df_test[f'class_proba_bogus_{model_name}'].mask(
        df_test[f'class_proba_bogus_{model_name}'] == 1, 
        other='Injection'
    )

    df_test[f'class_proba_bogus_{model_name}'] = df_test[f'class_proba_bogus_{model_name}'].mask(
        (df_test[f'class_proba_bogus_{model_name}'] == 0) & (df_test[f'proba_transient_perc_{model_name}'] <= 10), 
        other='<10%'
    )

    # Classifying transient probabilities based on percentage thresholds
    for thresh in [90, 80, 70, 60, 50, 40, 30, 20]:
        df_test[f'class_proba_transient_{model_name}'] = df_test.y_test.mask(
            (df_test.y_test == 1) & (df_test[f'proba_bogus_perc_{model_name}'] > thresh), 
            other=f'<{thresh}%'
        )

    df_test[f'class_proba_transient_{model_name}'] = df_test[f'class_proba_transient_{model_name}'].mask(
        (df_test[f'class_proba_transient_{model_name}'] == 1) & (df_test[f'proba_bogus_perc_{model_name}'] <= 20), 
        other='<10%'
    )

    df_test[f'class_proba_transient_{model_name}'] = df_test[f'class_proba_transient_{model_name}'].mask(
        df_test[f'class_proba_transient_{model_name}'] == 0, 
        other='bogus'
    )

    return df_test






def df_association(df, cutouts = None):
    
    with open(f'saved/full_dia_source_table_UDEEP_and_injected.pkl', "rb") as pickle_file:
        full_dia_source_table = pickle.load(pickle_file)

    df['original_order'] = range(len(df))

    df = df.set_index('diaSourceId')

    pd.set_option('display.float_format', lambda x: '%d' % x)

    df = df.join(full_dia_source_table['diaObjectId'].astype(str))
    df = df.join(full_dia_source_table['midpointMjdTai'])
    df['nDiaSources'] = df.groupby('diaObjectId')['diaObjectId'].transform('count')
    df.index = df.index.astype(str)
    df = df.reset_index()

    df = df.sort_values('original_order').drop(columns=['original_order'])

    df = df.reset_index(drop=True)
    
    if cutouts is not None:
        df['image_np'] = [None] * len(df)
        
        for idx in df.index:
            df.at[idx, 'image_np'] = cutouts[idx]
    return df

def retrieve_LC(diaObjectId, band=None):
    if band:
        with open(f'{rundir}/saved/df_asso_{band}.pkl', "rb") as pickle_file:
            df_asso = pickle.load(pickle_file)
        df_LC = df_asso[df_asso['diaObjectId']==str(diaObjectId)]

    else: 
        with open(f'{rundir}/saved/df_asso.pkl', "rb") as pickle_file:
            df_asso = pickle.load(pickle_file)
        df_LC = df_asso[df_asso['diaObjectId']==str(diaObjectId)]

    return df_LC