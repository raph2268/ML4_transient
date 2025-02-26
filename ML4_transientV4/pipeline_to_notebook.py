from lsst.daf.butler import Butler
import lsst.afw.display as afwDisplay
import lsst.geom
from astropy.nddata import Cutout2D
import matplotlib.pyplot as plt
from astropy.visualization import simple_norm
from astropy.visualization import (MinMaxInterval, SqrtStretch,
                                   ImageNormalize)
import pandas as pd
import numpy as np
from scipy.ndimage import rotate


import os,glob
import pylab as plt
repo = "/sps/lsst/users/bsanchez/hsc_processing/pgrc2subset_2/butler.yaml"


def extract_all_single_visit(visit_id, collection):
    '''
    Query all the datasets of a single visit, 
    
    Parameters:
        visit_id (int) : visit id as refered in the dia source

    Returns:
        dataIds (list) : list of goodSeeingDiff_differenceExp.dataId
    '''
    dataIds = []
    butler = Butler(repo,collections=collection)
    registry = butler.registry
    
    result = registry.queryDatasets(datasetType='injected_goodSeeingDiff_differenceExp', 
                                    collections=collection, 
                                    where= f"visit = {visit_id}")
    

    for ref in result:
        dataIds.append(ref.dataId)
        
    return dataIds


def save_cutout(collection, coadd_science= False, return_file = False, save_file = False, rotate_data = False): 
    '''
    From a given collection, perform cutout and save them visit by visit.
    
    Parameters:
        collection (str) : path to collection
    
    Returns: 
    
    '''
    all_cutout_in_collection = []
    all_features_in_collection = []
    
    butler = Butler(repo,collections=collection)
    registry = butler.registry

    #filter the dataset
    datasetRefs = registry.queryDatasets(datasetType='goodSeeingDiff_differenceExp',collections=collection)
    datasetRefs = registry.queryDatasets(datasetType='injected_goodSeeingDiff_differenceExp',collections=collection)

    # Make sure we are having an array listing all the visits (must appeared once)
    visits = []
    for ref in datasetRefs:
        visits.append(ref.dataId['visit'])
    visits = np.unique(visits)
    nbr_cutout =0
    nbr_visit = 0
    nbr_exposure = 0
    rotation_angles = [90, 180, 270]

    # Going through the list of unique visit and then call all the ccd/detector, so we can save all object of the same visit together
    for visit in visits: 
        print(visit)
        nbr_visit = nbr_visit+1
        # Extract the IDs for a given visit (each ID correspond to one ccd/detector)
        all_cutout_in_visit = []
        all_features_in_visit = []
        all_cutout_coadd_in_visit = []
        all_cutout_science_in_visit = []
        new_rows = []  
        rotated_cutouts = []
        for ref in extract_all_single_visit(visit, collection): 
            nbr_exposure =  nbr_exposure + 1

            # Extract all the objects of the given ccd/detector
            diff_array = butler.get('injected_goodSeeingDiff_differenceExp', dataId = ref).getImage().array # extract the image from the butler exposure 
            if coadd_science : 
                coadd_array = butler.get('injected_goodSeeingDiff_templateExp', dataId = ref).getImage().array # extract the image from the butler COADD 
                science_array = butler.get('injected_calexp', dataId = ref).getImage().array # extract the image from the butler CALEXP 

            full_dia_source_table = butler.get('injected_goodSeeingDiff_diaSrcTable', dataId = ref)  # coordinates of the different object in the diff
            match_dia_source_table = butler.get('injected_goodSeeingDiff_matchDiaSrc', dataId = ref) #table including the matching

            #Create the label
            full_dia_source_table['is_injection'] = full_dia_source_table.diaSourceId.isin(match_dia_source_table.diaSourceId)
            coord = full_dia_source_table

            num_detections=len(coord['x'])
            coord['rotation_angle'] = None

            for detect in range(num_detections):
                cutout = Cutout2D(diff_array, (coord['x'][detect], coord['y'][detect]), 30)  # Proceed to the cutouts using the astropy tool 
                if coadd_science: 
                    cutout_coadd = Cutout2D(coadd_array, (coord['x'][detect], coord['y'][detect]), 30)  # Proceed to the cutouts using the astropy tool 
                    cutout_science = Cutout2D(science_array, (coord['x'][detect], coord['y'][detect]), 30)  # Proceed to the cutouts using the astropy tool 

                # Check the cutout data format
                if cutout.data.shape == (30, 30) and not np.isnan(cutout.data).any():
                    # Store the original cutout (no rotation, angle = 0°)
                    if rotate_data:
                        original_cutout = cutout.data

                        # Create a new row in the catalog for the original cutout
                        new_row = coord.iloc[[detect]].copy() 
                        # Apply rotations and store the rotated versions
                        for angle in rotation_angles:  
                            rotated_cutout = rotate(cutout.data, angle, reshape=False)  # Reshape=False to keep the same dimensions

                            # Create a new row by copying the existing one and modifying the rotation angle
                            rotated_row = new_row.copy()  # Copy the original row
                            rotated_row['rotation_angle'] = angle  # Update the rotation angle
                            new_rows.append(rotated_row)  # Append the rotated row to new rows
                            rotated_cutouts.append(rotated_cutout)
                            nbr_cutout = nbr_cutout+1
                        
                            
                    all_cutout_in_visit.append(cutout.data)                 
                    all_features_in_visit.append(coord.iloc[detect])
                    nbr_cutout = nbr_cutout+1

                    if coadd_science:
                        all_cutout_coadd_in_visit.append(cutout_coadd.data)
                        all_cutout_science_in_visit.append(cutout_science.data) 

        
        

        all_features_in_visit = pd.DataFrame(all_features_in_visit)
        new_rows_df = pd.concat(new_rows, ignore_index=True)  # Concatenate rotated rows into a single DataFrame
        all_cutout_in_visit = np.concatenate((all_cutout_in_visit, rotated_cutouts))
        all_features_in_visit = pd.concat([all_features_in_visit, new_rows_df], ignore_index=True)


        if save_file == True:
            if rotate_data :
                np.save(f'saved/cutouts/Detected_obj_sources_rotated_visit_{visit}.npy', all_cutout_in_visit)
                all_features_in_visit.to_csv(f'saved/csv/Detected_obj_sources_rotated_visit_{visit}.csv')
                if coadd_science :
                    np.save(f'saved/cutouts_coadd/Coadd_detected_obj_sources_visit_{visit}.npy', all_cutout_coadd_in_visit)
                    np.save(f'saved/cutouts_science/Science_detected_obj_sources_visit_{visit}.npy', all_cutout_science_in_visit)

            else : 
                np.save(f'saved/cutouts/Detected_obj_sources_visit_{visit}.npy', all_cutout_in_visit)
                all_features_in_visit.to_csv(f'saved/csv/Detected_obj_sources_visit_{visit}.csv')
                if coadd_science :
                    np.save(f'saved/cutouts_coadd/Coadd_detected_obj_sources_visit_{visit}.npy', all_cutout_coadd_in_visit)
                    np.save(f'saved/cutouts_science/Science_detected_obj_sources_visit_{visit}.npy', all_cutout_science_in_visit)


        if return_file == True:
            all_cutout_in_collection.append(all_cutout_in_visit)
            all_features_in_collection.append(all_features_in_visit)

    print(f'visit analysed : {nbr_visit}, \nexposure explored : {nbr_exposure}, \ncutout realised : {nbr_cutout}')

    if return_file == True :    
        return all_cutout_in_collection, all_features_in_collection

def call_visit(visit): 
    '''
    Load from the save file both cutouts and features
    
    Parameters:
        visit_id (int) : visit id as refered in the dia source

    Returns:
        im (ndarray) : shape (number of cutout per visit (all filter, detector), 30, 30), cutout, raw
        feature (pandas.DataFrame) : features of each object of the visit 
    '''
    im = np.load(f'saved/cutouts/Detected_obj_sources_visit_{visit}.npy')
    feature = pd.read_csv(f'saved/csv/Detected_obj_sources_visit_{visit}.csv')
    
    return im, feature 


def plot_from_source(visit, max_cutouts = 1000): 
    '''
    Plot for a single visit the amount of cutout
    
    Parameters:
        visit_id (int) : visit id as refered in the dia source
        max_cutouts (int) :  set a limit of cutout to plot (stop the loop after this amount)
        
    '''
    
    # Create interval object
    interval = MinMaxInterval()
    cutout_visit = call_visit(visit)[0]
    feature_visit = call_visit(visit)[1]
    
    # Display the images
    i = 0

    num_cutout = len(cutout_visit)
        
    if num_cutout > max_cutouts: 
        num_cutout = max_cutouts
    cols = 10  # nbr columns wanted
    rows = num_cutout // cols + (num_cutout % cols > 0)

    fig, axs = plt.subplots(rows, cols, figsize=(1.8 * cols, 2.4 * rows))
    axs = axs.flatten()  # Flatten the array of axes for easy iteration

    iddet = 0

    for detect in range(num_cutout):

        # Image processing steps
        vmin, vmax = interval.get_limits(cutout_visit[detect])
        
        if iddet != feature_visit['detector'][detect]:
            iddet = feature_visit['detector'][detect]
            idvis = feature_visit['visit'][detect]
            print(f'visit nbr : {idvis}, detector : {iddet}')
        norm = ImageNormalize(vmin=vmin, vmax=vmax, stretch=SqrtStretch())
        axs[detect].imshow(cutout_visit[detect], origin='lower', norm=norm)
        axs[detect].axis('off')  # Hide the axis for a cleaner look

        # Add label to each subplot
        label = feature_visit['is_injection'][detect]
        if label == True :  axs[detect].set_title(f'Injection', fontsize=16) 
        else: axs[detect].set_title(f'Real', fontsize=16)
            
    # Hide any remaining empty subplots
    for ax in axs[num_cutout:]:
        ax.axis('off')

    plt.tight_layout()
    idvis = feature_visit['visit']
    iddet = feature_visit['detector']
    plt.show()
