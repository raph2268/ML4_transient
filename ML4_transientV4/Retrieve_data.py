# -*- coding: utf-8 -*-
import numpy as np
import sys,os

from sklearn.model_selection import train_test_split

import pandas as pd
from IPython.display import display
import pickle
pd.set_option('display.max_columns', None)

np.random.seed(1) # NumPy
import random
random.seed(2) # Python

from utils import datadir, pddir



data_path ='/sps/lsst/groups/transients/HSC/fouchez/raphael'



class retrieve_data_gen3:
    def __init__(self, visits, rotate=False):
        ''' 
        Initialize the retrieve_data object.

        Parameters:
            visits (int or list): Visit number(s) used. It can be a single value or an array containing several visit numbers.
        '''
        self.visits = visits

        if isinstance(self.visits, list):
            cutout_list = []
            label_list = []
            src_id_list = []
            for number_visit in self.visits:
                if rotate:
                    # Load rotated cutouts and associated feature data using pickle
                    with open((f"{data_path}/Detected_obj_sources_rotated_visit_{number_visit}.pkl"), 'rb') as f:
                        cutout = pickle.load(f)
                    self.feature = pd.read_pickle(f'{data_path}/Detected_obj_sources_rotated_visit_{number_visit}_df.pkl')

                else:
                    # Load non-rotated cutouts and associated feature data using pickle
                    with open((f"{data_path}/Detected_obj_sources_visit_{number_visit}.pkl"), 'rb') as f:
                        cutout = pickle.load(f)
                    self.feature = pd.read_pickle(f'{data_path}/Detected_obj_sources_visit_{number_visit}_df.pkl')
                
                
                
                # Append features to the lists
                label_list.append(self.feature['is_injection'].astype(int).values)
                src_id_list.append(self.feature[['diaSourceId', 'visit', 'band', 'apFlux', 'apFluxErr', 'snr']])
                cutout_list.append(cutout)
            
            # Stack cutouts and concatenate labels and source IDs
            self.cutout = np.vstack(cutout_list)
            self.labels = np.concatenate(label_list)
            self.src_id = np.concatenate(src_id_list)
        
        else:
            if rotate:
                # Load rotated cutouts and associated feature data for a single visit
                with open((f"{data_path}//Detected_obj_sources_rotated_visit_{self.visits}.pkl"), 'rb') as f:
                    self.cutout = pickle.load(f)
                self.feature = pd.read_pickle(f'{data_path}/Detected_obj_sources_rotated_visit_{self.visits}_df.pkl')
            else:
                # Load non-rotated cutouts and associated feature data for a single visit
                with open((f"{data_path}/Detected_obj_sources_visit_{self.visits}.pkl"), 'rb') as f:
                    self.cutout = pickle.load(f)
                self.feature = pd.read_pickle(f'{data_path}/Detected_obj_sources_visit_{self.visits}_df.pkl')
            
            self.labels = self.feature['is_injection'].astype(int)
            self.src_id = self.feature[['diaSourceId', 'visit', 'band', 'apFlux', 'apFluxErr', 'snr']]

    def split(self, test_size):
        '''
        Split the dataset into training and test sets.
        
        Parameters:
            test_size (float): Fraction of data to use for testing.

        Returns:
            x_train, x_test: Numpy arrays of the cutout images for training and testing.
            feature_train, feature_test: Numpy arrays of source ID features for training and testing.
            y_train, y_test: Numpy arrays of labels for training and testing.
        '''
        x = self.cutout
        y = np.array(self.labels)
        feature = np.array(self.src_id)

        x_train, x_test, y_train, y_test, feature_train, feature_test = train_test_split(x, y, feature, test_size=test_size, random_state=42)
        x_train = np.array(x_train)
        x_test = np.array(x_test)
        x_train = x_train.reshape(-1, 30, 30, 1)
        x_test  = x_test.reshape(-1, 30, 30, 1)
        
        # Normalize data
        x_train = (x_train - x_train.min(axis=(1, 2)).reshape((-1, 1, 1, 1))) / (x_train.max(axis=(1, 2)).reshape((-1, 1, 1, 1)) - x_train.min(axis=(1, 2)).reshape((-1, 1, 1, 1)))
        x_test  = (x_test - x_test.min(axis=(1, 2)).reshape((-1, 1, 1, 1))) / (x_test.max(axis=(1, 2)).reshape((-1, 1, 1, 1)) - x_test.min(axis=(1, 2)).reshape((-1, 1, 1, 1)))
        
        return x_train, x_test, feature_train, y_train, y_test, feature_test

    def data(self, test_size):
        '''
        Produce DataFrames to store the data and return them along with numpy arrays for training/testing the model.

        Parameters:
            test_size (float): Fraction of data to use for testing.

        Returns:
            df_train, df_test: DataFrames containing information like labels and indices for training/testing.
            x_train, x_test: Numpy arrays of data for training/testing the model.
            y_train, y_test: Numpy arrays of labels for training/testing the model.
        '''
        x_train, x_test, feature_train, y_train, y_test, feature_test = self.split(test_size)
        
        # Prepare training DataFrame
        df_train = pd.DataFrame({'y_train': y_train.astype(int)})
        feature_df_train = pd.DataFrame(feature_train, columns=['diaSourceId', 'visit', 'band', 'apFlux', 'apFluxErr', 'snr'])
        df_train = pd.concat([df_train, feature_df_train], axis=1)

        # Prepare test DataFrame
        df_test = pd.DataFrame({'y_test': y_test.astype(int)})
        feature_df_test = pd.DataFrame(feature_test, columns=['diaSourceId', 'visit', 'band', 'apFlux', 'apFluxErr', 'snr'])
        df_test = pd.concat([df_test, feature_df_test], axis=1)

        return df_train, df_test, x_train, x_test, y_train, y_test

    
    



    
    
    
    
class retrieve_data_gen3_UDEEP:
    def __init__(self, visits, rotate=False):
        ''' 
        Initialize the retrieve_data object.

        Parameters:
            visits (int or list): Visit number(s) used.
        '''
        self.visits = visits

        if isinstance(self.visits, list):
            cutout_list = []
            label_list = []
            src_id_list = []
            for number_visit in self.visits:
                if rotate: 
                    with open((f"{data_path}/Detected_obj_sources_rotated_visit_{number_visit}_UDEEP.pkl"), 'rb') as f:
                        cutout = pickle.load(f)
                    self.feature = pd.read_pickle(f'{data_path}/Detected_obj_sources_rotated_visit_{number_visit}_UDEEP_df.pkl')

                else: 
                    with open((f"{data_path}/Detected_obj_sources_visit_{number_visit}_UDEEP.pkl"), 'rb') as f:
                        cutout = pickle.load(f)
                    self.feature = pd.read_pickle(f'{data_path}/Detected_obj_sources_visit_{number_visit}_UDEEP_df.pkl')
                    
                label_list.append(self.feature['is_injection'].astype(int).values)
                src_id_list.append(self.feature[['diaSourceId', 'visit', 'band', 'apFlux', 'apFluxErr', 'snr']])
                cutout_list.append(cutout)
                
            self.cutout = np.vstack((*cutout_list,))
            self.labels = np.concatenate(label_list)
            self.src_id = np.concatenate(src_id_list)
        
        else:
            if rotate: 
                with open((f"{data_path}/Detected_obj_sources_rotated_visit_{self.visits}_UDEEP.pkl"), 'rb') as f:
                    self.cutout = pickle.load(f)
                self.feature = pd.read_pickle(f'{data_path}/Detected_obj_sources_rotated_visit_{self.visits}_UDEEP_df.pkl')

            else:
                with open((f"{data_path}/Detected_obj_sources_visit_{self.visits}_UDEEP.pkl"), 'rb') as f:
                    self.cutout = pickle.load(f)
                self.feature = pd.read_pickle(f'{data_path}/Detected_obj_sources_visit_{self.visits}_UDEEP_df.pkl')

            self.labels = self.feature['is_injection'].astype(int)
            self.src_id = self.feature[['diaSourceId', 'visit', 'band', 'apFlux', 'apFluxErr', 'snr']]

    def split(self, test_size):
        x = self.cutout
        y = np.array(self.labels)
        feature = np.array(self.src_id)

        x_train, x_test, y_train, y_test, feature_train, feature_test = train_test_split(x, y, feature, test_size=test_size, random_state=42)
        x_train=np.array(x_train)
        x_test = np.array(x_test)

        x_train = x_train.reshape(-1, 30, 30, 1)
        x_test  = x_test.reshape(-1, 30, 30, 1)
        
        # Normalize data
        x_train = (x_train - x_train.min(axis=(1, 2)).reshape((-1, 1, 1, 1))) / (x_train.max(axis=(1, 2)).reshape((-1, 1, 1, 1)) - x_train.min(axis=(1, 2)).reshape((-1, 1, 1, 1)))
        x_test  = (x_test - x_test.min(axis=(1, 2)).reshape((-1, 1, 1, 1))) / (x_test.max(axis=(1, 2)).reshape((-1, 1, 1, 1)) - x_test.min(axis=(1, 2)).reshape((-1, 1, 1, 1)))
        
        return x_train, x_test, feature_train, y_train, y_test, feature_test

    def data(self, test_size):
        x_train, x_test, feature_train, y_train, y_test, feature_test = self.split(test_size)
        
        df_train = pd.DataFrame({'y_train': y_train.astype(int)})
        feature_df_train = pd.DataFrame(feature_train, columns=['diaSourceId', 'visit', 'band', 'apFlux', 'apFluxErr', 'snr'])
        df_train = pd.concat([df_train, feature_df_train], axis=1)
        
        df_test = pd.DataFrame({'y_test': y_test.astype(int)})
        feature_df_test = pd.DataFrame(feature_test, columns=['diaSourceId', 'visit', 'band', 'apFlux', 'apFluxErr', 'snr'])
        df_test = pd.concat([df_test, feature_df_test], axis=1)

        return df_train, df_test, x_train, x_test, y_train, y_test

    
    
    
    
    
    

class retrieve_data_gen3_all_col:
    def __init__(self, visits):
        ''' 
        Initialize the retrieve_data object.

        Parameters:
            visits (int or list): Visit number(s) used. It can be a single value or an array containing several visit numbers.
            nbre_data (int): Number of data used (optional).

        '''
        self.datadir = '/sps/lsst/users/rbonnetguerrini/ML4_transientV3/saved'     
        self.visits = visits
        self.feature_columns = None  # Attribute to store feature column names

        if isinstance(self.visits, list):
            cutout_list = []
            label_list = []
            src_id_list = []
            for number_visit in self.visits:
                cutout = np.load(os.path.join(self.datadir, f"cutouts/Detected_obj_sources_visit_{number_visit}.npy"))
                self.feature = pd.read_csv(f'/sps/lsst/users/rbonnetguerrini/ML4_transientV3/saved/csv/Detected_obj_sources_visit_{number_visit}.csv')
                if self.feature_columns is None:
                    self.feature_columns = self.feature.columns.tolist()
                
                label_list.append(self.feature['is_injection'].astype(int).values)
                src_id_list.append(self.feature)
                cutout_list.append(cutout)
                
            self.cutout = np.vstack((*cutout_list,))
            self.labels = np.concatenate(label_list)
            self.src_id = np.concatenate(src_id_list)
        else : 
            self.cutout = np.load(os.path.join(self.datadir, f"cutouts/Detected_obj_sources_visit_{self.visits}.npy"))
            self.feature = pd.read_csv(f'{data_path}/csv/Detected_obj_sources_visit_{self.visits}.csv')
            self.feature_columns = self.feature.columns.tolist()

            self.labels = self.feature['is_injection'].astype(int)
            self.src_id = self.feature 
  
    def split(self, test_size):
        #x = np.concatenate((obj, simu))
        x = self.cutout
        y = np.array(self.labels)  # np.concatenate((np.zeros(len(obj)) , np.ones(len(simu)))).astype(np.int32)
        feature = np.array(self.src_id)
        x_train, x_test, y_train, y_test, feature_train, feature_test = train_test_split(x, y, feature, test_size=test_size, random_state=42)
        
        x_train = x_train.reshape(-1,30,30,1)
        x_test  = x_test.reshape(-1,30,30,1)
            
        #Normalizes data
        x_train = (x_train- x_train.min(axis=(1,2)).reshape((-1,1,1,1)) )/ (x_train.max(axis=(1,2)).reshape((-1,1,1,1))- x_train.min(axis=(1,2)).reshape((-1,1,1,1)))
        x_test  = (x_test - x_test.min(axis=(1,2)).reshape((-1,1,1,1)))/ (x_test.max(axis=(1,2)).reshape((-1,1,1,1)) - x_test.min(axis=(1,2)).reshape((-1,1,1,1)))  
        y_train = y_train
        y_test = y_test      
        
        return x_train, x_test, feature_train, y_train, y_test, feature_test

            
    def data(self, test_size): 
        '''
        Produce DataFrames to store the data and return them along with numpy arrays for training/testing the model.

        Parameters:
            test_size (float): Percentage of the total dataset that will be attributed to test datasets.

        Returns:
            df_train, df_test: DataFrames containing information like labels and indices associated with data for training/testing the model.
            x_train, x_test: Numpy arrays of data for training/testing the model.
            y_train, y_test: Numpy arrays of labels associated with data for training/testing the model.
        '''
       
        x_train, x_test, feature_train, y_train, y_test, feature_test = self.split(test_size)
        
        df_train = pd.DataFrame() 
        df_train['y_train'] = y_train.astype(int)
        df_train['y_train'] = df_train['y_train'].to_frame('y_train')
        feature_df_train = pd.DataFrame(feature_train, columns=self.feature_columns)
        df_train = pd.concat([df_train, feature_df_train], axis=1)         # Merge by concatenating along columns (axis=1)

        
        
        df_test = pd.DataFrame()
        df_test['y_test'] = y_test.astype(int)
        df_test['y_test'] = df_test['y_test'].to_frame('y_test')
        feature_df_test = pd.DataFrame(feature_test, columns=self.feature_columns)
        df_test = pd.concat([df_test, feature_df_test], axis=1)         # Merge by concatenating along columns (axis=1)

        return df_train, df_test, x_train, x_test, y_train, y_test
