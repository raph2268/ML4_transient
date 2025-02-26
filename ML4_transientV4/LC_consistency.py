import pandas as pd
import numpy as np
from collections import Counter

from bokeh.io import output_notebook, push_notebook, show, curdoc, reset_output
from bokeh.plotting import figure, output_file, save
from bokeh.models import (ColumnDataSource, Slider, HoverTool, CategoricalColorMapper, Legend, Div,
                          Title, Tabs, Whisker, Circle, LegendItem, BoxZoomTool, ColorBar, LinearAxis, Range1d)
from bokeh.layouts import column, gridplot, row
from bokeh.palettes import Spectral10, Viridis, Spectral, Category10, brewer, tol, Turbo256
from bokeh.transform import factor_cmap, factor_mark

import pickle
from bokeh.transform import linear_cmap

from bokeh.transform import factor_cmap, factor_mark, linear_cmap

from matplotlib import cm

from astropy.time import Time
from datetime import datetime

from PIL import ImageOps
from PIL import Image
import io
import base64
import glob
from astropy import units as u
import sys, os

from astropy.coordinates import SkyCoord
import astropy.units as u
import glob
from astropy.time import Time

from Retrieve_data import retrieve_data

from utils import ImageDisplayer, datadir, rundir, pddir, embeddable_image, time_dict, filt_dict, All_visits

from df_bogus import df_proba
from astropy.coordinates import SkyCoord
from astropy import units as u
import pickle
import warnings
warnings.filterwarnings("ignore", message="out of range integer may result in loss of precision", category=UserWarning)


rundir = '/sps/lsst/users/rbonnetguerrini/ML4_transientV2'

    
    
#Building the network consistency metric
def NetCons(object_dict, model):
    """
    Calculates the Network Consistency metric for a set of light curves and classifies them.

    Parameters:
    - object_dict (dict): A dictionary where keys are object IDs and values are dictionaries containing:
        - 'cutouts': List of numpy arrays representing cutouts of light curves.
        - 'visits': List of visit times corresponding to the cutouts.
        - 'bandpasses': List of bandpasses corresponding to the cutouts.
    - model (object): A trained ML model that returns prediction probabilities.

    Returns:
    - NetCons (float): The overall network consistency metric.
    - LCCs (list): List of Light Curve Consistency metrics for each light curve.
    - LC_classes (list): List of predicted class labels for each light curve.
    - LC_len (list): List of the number of cutouts in each light curve.
    - cutout_classes (list): List of predictions, visits, and bandpasses for each cutout in each light curve.
    """

    total_cutouts = sum(len(data['cutouts']) for data in object_dict.values())

    LCCs = [] 
    LC_len = []
    LC_classes = []
    NetCons = 0
    cutout_classes = []
    count_nobj = 0
    count_sum_nobj = 0
    for id_, data in object_dict.items():
        cutouts_in_LC = data['cutouts']
        # Perform prediction
        LC_predict = model.predict(cutouts_in_LC, verbose = 0)
        classified = np.argmax(LC_predict, axis=-1)
        #print('classified : ', classified)
        preds = []
        for i in range(len(LC_predict)): 
            preds.append([np.argmax(LC_predict[i], axis=-1), data['visits'][i], data['bandpasses'][i]])
        
        cutout_classes.append(preds)

        LC_class = round(np.mean(classified))
        LC_classes.append(LC_class)
        LCC = max(1-np.mean(classified),np.mean(classified)) # Light curve consistency

        #print(LCC)
        
        LCCs.append(LCC)
        nobj = len(cutouts_in_LC)
        sum_nobj = (total_cutouts)
        LC_len.append(nobj)
        NetCons = NetCons + nobj/sum_nobj *  LCC
        count_nobj = count_nobj + nobj
        
    return NetCons, LCCs, LC_classes, LC_len, cutout_classes

def encode_image_to_base64(data):
    """
    Converts a numpy array image to a base64-encoded PNG string.

    Parameters:
    - data (numpy array): Image data to be encoded.

    Returns:
    - str: Base64-encoded PNG string.
    """
    
    # Ensure data is a numpy array and squeeze it
    data = np.squeeze(data)
    
    # Normalize the data to the range [0, 1]
    data_min = np.min(data)
    data_max = np.max(data)
    if data_max > data_min:
        normalized_data = (data - data_min) / (data_max - data_min)
    else:
        normalized_data = np.zeros_like(data)
    
    # Apply the colormap
    colormap = cm.get_cmap('viridis')
    colored_image = colormap(normalized_data)
    
    # Convert the colormap result to an 8-bit image
    image = Image.fromarray(np.uint8(colored_image * 255))

    # Save the image to a buffer
    buffer = io.BytesIO()
    image.save(buffer, format='png')
    
    # Encode the image in base64
    for_encoding = buffer.getvalue()
    return 'data:image/png;base64,' + base64.b64encode(for_encoding).decode()


def plot_consistency(NetCons, LCCs, LC_classes, LC_len, cutout_images, cutout_classes):
    """
    Creates and displays a plot showing the consistency of light curves and their classification.

    Parameters:
    - NetCons (float): The overall network consistency metric.
    - LCCs (list): List of Light Curve Consistency metrics.
    - LC_classes (list): List of predicted class labels for each light curve.
    - LC_len (list): List of the number of cutouts in each light curve.
    - cutout_images (dict): Dictionary with keys as object IDs and values containing 'cutouts' which are lists of numpy arrays.
    - cutout_classes (list): List of predictions, visits, and bandpasses for each cutout in each light curve.
    
    Displays:
    - A scatter plot showing the relationship between the number of objects in the light curve and the light curve prediction consistency.
    - A bar plot showing the percentage of transient objects per band.
    """
    curdoc().theme = 'dark_minimal'

    # Identify indices for transient (class 1) and bogus (class 0) light curves
    transient_indices = [i for i in range(len(LC_classes)) if LC_classes[i] == 1]
    bogus_indices = [i for i in range(len(LC_classes)) if LC_classes[i] == 0]

    # Extract data for transient and bogus light curves
    transient_lens = [LC_len[i] for i in transient_indices]
    transient_cons = [LCCs[i] for i in transient_indices]
    transient_classes_visits = [cutout_classes[i] for i in transient_indices]
    
    # Extract individual details for transient light curves
    transient_classes = [[sub[0] for sub in inner_list] for inner_list in transient_classes_visits]
    transient_visit = [[sub[1] for sub in inner_list] for inner_list in transient_classes_visits]
    transient_bdp = [[sub[2] for sub in inner_list] for inner_list in transient_classes_visits]
    
    bogus_lens = [LC_len[i] for i in bogus_indices]
    bogus_cons = [LCCs[i] for i in bogus_indices]
    bogus_classes_visits = [cutout_classes[i] for i in bogus_indices]
    
    # Extract individual details for bogus light curves
    bogus_classes = [[sub[0] for sub in inner_list] for inner_list in bogus_classes_visits]
    bogus_visit = [[sub[1] for sub in inner_list] for inner_list in bogus_classes_visits]
    bogus_bdp = [[sub[2] for sub in inner_list] for inner_list in bogus_classes_visits]

    # Convert cutout images to base64 strings for embedding in HTML tooltips
    all_cutouts_base64 = []
    for idx_, data in cutout_images.items():
        cutouts = data['cutouts']
        cutouts_base64 = [encode_image_to_base64(cutout) for cutout in cutouts]
        all_cutouts_base64.append(cutouts_base64)
    
    # Separate cutout images for transient and bogus light curves

    transient_cutout_base64 = [all_cutouts_base64[i] for i in transient_indices]
    bogus_cutout_base64 = [all_cutouts_base64[i] for i in bogus_indices]

    # Prepare HTML tooltips for images
    def create_tooltip(images, text1, text2, text3):
        tooltip = "<div style='white-space: nowrap;'>"
        for j in range(len(images)):
            # Wrap each image and its text in a div
            tooltip += f"""
            <div style='display: inline-block; text-align: center; margin-right: 1px;'>
                <div style='font-size: 8px; color: #333;'>pred : {str(text1[j])}, band : {str(text3[j])}</div>
                <img src='{images[j]}' height='50' width='50' style='display: block; margin: 2px auto;'></img>
            </div>
            """
        tooltip += "</div>"
        return tooltip
    
    # Generate tooltips for transient and bogus light curves
    transient_cutouts = [create_tooltip(transient_cutout_base64[i], transient_classes[i], transient_visit[i], transient_bdp[i]) for i in range(len(transient_classes))]
    bogus_cutouts = [create_tooltip(bogus_cutout_base64[i], bogus_classes[i], bogus_visit[i], bogus_bdp[i]) for i in range(len(bogus_classes))]

    # Create ColumnDataSource objects for Bokeh scatter plots
    transient_source = ColumnDataSource(data=dict(x=transient_lens, y=transient_cons, index=transient_indices, image=transient_cutouts))
    bogus_source = ColumnDataSource(data=dict(x=bogus_lens, y=bogus_cons, index=bogus_indices, image=bogus_cutouts))

    # Create a scatter plot to show the relationship between number of objects and consistency
    scatter_plot = figure(title=f'NetCons = {NetCons}, ratio transient/bogus : {len(transient_indices)/len(bogus_indices)}',
                          x_axis_label='Number of objects in the light curve',
                          y_axis_label='Light Curve prediction consistency',
                          tools='pan,wheel_zoom,box_zoom,reset,hover')

    scatter_plot.scatter('x', 'y', source=transient_source, color='red', marker='x', size=10, legend_label='Class 1')
    scatter_plot.scatter('x', 'y', source=bogus_source, color='blue', marker='circle', size=10, legend_label='Class 0')

    # Add HoverTool to display images and additional information on hover
    hover = HoverTool()
    hover.tooltips = [("LC element", "@x"), ("Consistency", "@y"), ("Cutouts", "@image{safe}")]
    scatter_plot.add_tools(hover)
    scatter_plot.legend.title = 'Prediction Percentage for Real Data'

    # Data for the bar plot
    band_counts = Counter()
    label_1_counts = Counter()
    
    # Count occurrences of each band and class label
    for sublist in cutout_classes:
        for entry in sublist:
            label = entry[0]
            band = entry[2]
            
            band_counts[band] += 1
            if label == 1:
                label_1_counts[band] += 1
                
    # Calculate percentage of transients for each band and overall
    percentages = {band: (label_1_counts[band] / band_counts[band]) * 100 for band in band_counts}
    total_label_1 = sum(label_1_counts.values())
    label_1_counts['all_bands'] = total_label_1
    total_count = sum(band_counts.values())
    overall_percentage = (total_label_1 / total_count) * 100
    percentages['all_bands'] = overall_percentage
    

    bands = list(percentages.keys())
    values = list(percentages.values())

    bar_source = ColumnDataSource(data=dict(
        bands=bands,
        values=values,
        counts=[label_1_counts[band] for band in bands],
        colors=['green', 'red', 'darkred', 'pink', 'blue']  # Add the colors list here
    ))
    # Create the bar plot
    bar_plot = figure(x_range=bands, height=400, title="Percentage of transient per Band",
                      toolbar_location=None, tools="")

    bar_plot.vbar(x='bands', top='values', width=0.9, source=bar_source, color='colors')

    # Annotate bars with counts
    bar_plot.text(x='bands', y='values', source=bar_source, text='counts', text_font_size='10pt',
                  text_color='black', text_align='center', text_baseline='bottom')
    
    # Customize bar plot appearance
    bar_plot.y_range.start = 0
    bar_plot.xgrid.grid_line_color = None
    bar_plot.axis.minor_tick_line_color = None
    bar_plot.outline_line_color = None
    bar_plot.xaxis.axis_label = "Band"
    bar_plot.yaxis.axis_label = "Percentage of transient(%)"
    bar_plot.title.align = 'center'
    bar_plot.add_layout(bar_plot.title, 'above')

    # Add a horizontal line for the overall percentage
    bar_plot.line([0, 10], [overall_percentage, overall_percentage], color='gray', line_width=2, line_dash='dashed')
    bar_plot.add_layout(bar_plot.title)

    # Arrange plots side by side
    grid = gridplot([[scatter_plot, bar_plot]])

    # Show the plot
    output_notebook()  # Use this line bcs running in a Jupyter notebook
    show(grid, notebook_handle=True)
    
    return grid 





#################################################################################################################################
#################################################################################################################################
# Light curve single object plot


def get_id_mag(object_ids):
    """
    get the calibrated magnitude for a given source index
    """
    visit_id, det_id=get_id_from_src(object_ids)

    real_cat_filename=glob.glob(f'/sps/lsst/users/bracine/ForMariam/20210320_Fake_and_real_cutouts/cutouts_with_calexp/Cat_real_sources_visit_{visit_id}*.csv')[0]
    Catalog_detected_real=pd.read_csv(real_cat_filename)
    dia_srcs=Catalog_detected_real[Catalog_detected_real['id']==object_ids]
    
    cal_flux, cal_flux_err, cal_mag, cal_mag_err = get_mag_values("base_CircularApertureFlux_6_0_instFlux",dia_srcs)
    return cal_flux, cal_flux_err, cal_mag, cal_mag_err

def get_mag_values(fluxname, source_row):
    """
    Retrieves calibrated magnitude and flux values using the DM tools for that
    particular image for a given flux measurement.

    This way of converting fluxes to magnitudes is aware of the zero points
    of the image as well as the correct calculation.
    
    We can obtain then, magnitudes and fluxes in NJy.
    
    Parameters:
        fluxname (str): The name of the flux measurement.
        source_row (dict): A dictionary containing the source information, including the flux and flux error values.
        photcal (PhotCalibration): An instance of the PhotCalibration class that provides calibration functions.

    Returns:
        cal_mag (float): Calibrated magnitude value.
        cal_flux (float): Calibrated flux value in nanojanskys.
    """
    cal_flux = source_row[fluxname]
    cal_flux_err = source_row[fluxname+'Err'] 

    cal_mag = (np.array(source_row[fluxname])* u.nJy).to(u.ABmag).value
    cal_mag_err = (np.array(source_row[fluxname+'Err'])* u.nJy).to(u.ABmag).value
    
    return cal_flux, cal_flux_err, cal_mag, cal_mag_err

def get_id_from_src(src_id):
    """
    this function is decoding the id, found that out from a DC2 notebook from Bruno, and adapted to HSC
    """
    visit_id = (src_id>>32)//200
    det_id = (src_id>>32)%(200*visit_id)
    return visit_id, det_id

def encode_image_to_base64(data):
    """
    Converts a numpy array image to a base64-encoded PNG string.

    Parameters:
    - data (numpy array): Image data to be encoded.

    Returns:
    - str: Base64-encoded PNG string.
    """
    
    # Ensure data is a numpy array and squeeze it
    data = np.squeeze(data)
    
    # Normalize the data to the range [0, 1]
    data_min = np.min(data)
    data_max = np.max(data)
    if data_max > data_min:
        normalized_data = (data - data_min) / (data_max - data_min)
    else:
        normalized_data = np.zeros_like(data)
    
    # Apply the colormap
    colormap = cm.get_cmap('viridis')
    colored_image = colormap(normalized_data)
    
    # Convert the colormap result to an 8-bit image
    image = Image.fromarray(np.uint8(colored_image * 255))

    # Save the image to a buffer
    buffer = io.BytesIO()
    image.save(buffer, format='png')
    
    # Encode the image in base64
    for_encoding = buffer.getvalue()
    return 'data:image/png;base64,' + base64.b64encode(for_encoding).decode()

def retrieve_LC(object_id, ref_visit_id, visit_ids, filt_dict, time_dict):
    radius=0.2
    # Initialize lists to store results
    matched_tables = []
    matched_cutouts = []
    visit_numbers = []
    object_ids = [] 
    # Load reference catalog and select the object of interest
    ref_cat = pd.read_csv(os.path.join(pddir, f"Cat_real_sources_visit_{ref_visit_id}.csv"))
    ref_object = ref_cat.loc[ref_cat['id'] == object_id]
    
    # Ensure the object_id exists in the reference visit catalog
    if ref_object.empty:
        print(f"Object ID {object_id} not found in reference visit {ref_visit_id}.")
        return {
            'cutouts': np.array([]),
            'visits': np.array([]),
            'bandpasses': [],
            'object_ids': [], 
        }
    
    # Get reference object's coordinates and create a SkyCoord object
    ref_coord = SkyCoord([ref_object["coord_ra"].values[0]] * u.radian, [ref_object["coord_dec"].values[0]] * u.radian)
    
    # Loop over the other visit IDs
    for visit_id in visit_ids:
        if visit_id == ref_visit_id:
            continue  # Skip the reference visit
        
        # Load the data for the current visit
        real_data = np.load(os.path.join(datadir, f"Detected_real_sources_visit_{visit_id}.npy"))
        cat_real = pd.read_csv(os.path.join(pddir, f"Cat_real_sources_visit_{visit_id}.csv"))
        
        # Convert current catalog coordinates to SkyCoord
        new_cat_coord = SkyCoord(cat_real["coord_ra"].values * u.radian, cat_real["coord_dec"].values * u.radian)
        
        # Match current visit catalog to the reference object
        idx, d2d, _ = new_cat_coord.match_to_catalog_sky(ref_coord)
        
        # Mask for matches within the specified radius
        mask_match = d2d.arcsec < radius
        
        if mask_match.any():  # If any matches are found
            matched_idx = np.where(mask_match)[0]
            
            # Extract matched entries from the current catalog
            matched_table = cat_real.iloc[matched_idx].copy()
            matched_table['visit_id'] = visit_id  # Track the visit ID
            matched_table['object_id'] = object_id  # Use the original object ID for consistency
            
            # Extract corresponding cutouts from real_data
            matched_cutout = real_data[matched_idx]
            object_ids.extend(matched_table['id'].tolist())
            matched_tables.append(matched_table)
            matched_cutouts.append(matched_cutout)
            visit_numbers.append(np.full(len(matched_cutout), visit_id))  # Add visit number array
    
    # Combine all matched tables into a single DataFrame
    if matched_tables:
        result_table = pd.concat(matched_tables, ignore_index=True)
    else:
        result_table = pd.DataFrame()  # Return an empty DataFrame if no matches found
    
    # Combine all matched cutouts into a single array
    if matched_cutouts:
        result_cutouts = np.concatenate(matched_cutouts)
        result_visits = np.concatenate(visit_numbers)  # Combine visit numbers
    else:
        result_cutouts = np.array([])  # Return an empty array if no matches found
        result_visits = np.array([])

    # Gather the data for the specified object_id
    object_light_curve = {
        'cutouts': result_cutouts,
        'visits': result_visits,
        'object_ids': object_ids,
        'bandpasses': [filt_dict.get(visit_id, 'unknown') for visit_id in result_visits],
        'time':[time_dict.get(visit_id, 'unknown') for visit_id in result_visits]
    }

    return object_light_curve

def plot_LC (object_id, ref_visit_id, model):
    load_model = tf.keras.models.load_model(f'{rundir}/Compare_visit/models/{model}.h5')
    LC =  retrieve_LC(object_id, ref_visit_id, All_visits, filt_dict, time_dict)
    if LC['visits'].size > 0:
        # Extract data
        times = LC['time']
        magnitudes = [get_id_mag(obj_id)[2] for obj_id in LC['object_ids']]  # Extract magnitudes
        times = [Time(i, format='mjd').datetime for i in LC['time']]
        bandpasses = LC['bandpasses']
        cutouts = LC['cutouts']
        LC_predicts = load_model.predict(cutouts, verbose = 0)
        transient_probs = (LC_predicts[:,1]*100).astype(int)
        cutouts = [encode_image_to_base64(img) for img in LC['cutouts']]

        df = pd.DataFrame({
            'cutouts': cutouts,
            'mag' : magnitudes,
            'visits':  LC['visits'],
            'object_ids': LC['object_ids'],
            'bandpasses': LC['bandpasses'] ,
            'time': times, 
            'pred': transient_probs
        })

    databogus = ColumnDataSource(df.loc[df.pred<50])
    datatransient = ColumnDataSource(df.loc[df.pred>50])

    # Prepare data for Bokeh
    GROUP2 = ['g', 'r', 'i' ,'z'] 

    colors = ['green','red','darkred','deeppink']
    # Create the plot
    p = figure(title="Light Curve", x_axis_label='Time', y_axis_label='Magnitude', x_axis_type='datetime')
    p.square('time', 'mag', size=8,color=factor_cmap('bandpasses', colors, factors=GROUP2), alpha=0.5, source=databogus)
    p.star('time', 'mag', size=8,color=factor_cmap('bandpasses', colors, factors=GROUP2), alpha=0.5, source=datatransient)

    p.add_tools(HoverTool(tooltips="""
        <div>
            <div>
                <span style='font-size: 14px; color: #224499'> filter : @bandpasses</span>
                <br>
                <span style='font-size: 14px; color: #224499'> pred : @pred</span>
                <img src='@cutouts' height="60" width="60" style='float: left; margin: 5px 5px 5px 5px'></img>
            </div>
        </div>
        """))
    # Customize plot
    p.yaxis.axis_label = "Magnitude"
    p.xaxis.axis_label = "Time"
    p.y_range.flipped = True  # Magnitudes are usually plotted inversely
    curdoc().theme = 'dark_minimal'

    # Show plot
    output_notebook()
    show(p, notebook_handle =True)
    return p





def plot_LC_gen3(df, model, model_name = None): 
    cutouts_array = np.array(df['image_np'].tolist())

    if model_name is None: 
        df = df_proba(df, cutouts_array, df['y_test'].values , model)

        model_name = model
        
    else : 
        y_test_tensor  = torch.tensor(df['y_test'].values, dtype=torch.long)
        df = df_proba_pytorch(df, cutouts_array, y_test_tensor , model, model_name)
    
    for idx, row in df.iterrows(): 
        visit = row['visit']
        features = pd.read_csv(f'{rundir}/saved/csv/Detected_obj_sources_visit_{visit}.csv')
        idx_img = features[features['diaSourceId'].astype(str)==row['diaSourceId']].index
        coadd = np.load(f"{rundir}/saved/cutouts_coadd/Coadd_detected_obj_sources_visit_{visit}.npy")
        science = np.load(f"{rundir}/saved/cutouts_science/Science_detected_obj_sources_visit_{visit}.npy")
        coadd = coadd.reshape(-1,30,30,1)
        coadd = (coadd- coadd.min(axis=(1,2)).reshape((-1,1,1,1)) )/ (coadd.max(axis=(1,2)).reshape((-1,1,1,1))- coadd.min(axis=(1,2)).reshape((-1,1,1,1)))
        science = science.reshape(-1,30,30,1)
        science = (science- science.min(axis=(1,2)).reshape((-1,1,1,1)) )/ (science.max(axis=(1,2)).reshape((-1,1,1,1))- science.min(axis=(1,2)).reshape((-1,1,1,1)))

        df.at[idx, 'image_coadd'] = embeddable_image(coadd[1537])
        df.at[idx, 'image_science'] = embeddable_image(science[idx_img])
    
    df['image'] = [embeddable_image(data) for data in cutouts_array]    
    
    df['cal_mag'] = (np.array(df['apFlux'])* u.nJy).to(u.ABmag).value
    df['cal_mag_err'] = (np.array(df['apFluxErr'])* u.nJy).to(u.ABmag).value
    df['day_obs'] = Time(df['midpointMjdTai'], format='mjd').datetime64
    dataTP = ColumnDataSource(df.loc[df.true_positive == 1])
    dataTN = ColumnDataSource(df.loc[df.true_negative == 1])
    dataFP= ColumnDataSource(df.loc[df.false_positive == 1])
    dataFN = ColumnDataSource(df.loc[df.false_negative == 1])
    if model_name is None: 
        model_name = model
    # Prepare data for Bokeh
    GROUP2 = ['g', 'r', 'i' ,'z'] 

    colors = ['green','red','darkred','deeppink']
    # Create the plot
    p = figure(title="Light Curve", x_axis_label='Time', y_axis_label='Magnitude', x_axis_type='datetime')
    
    TP_glyph = p.x('day_obs', 'cal_mag', color='#FFC300',  source=dataTP, line_alpha=0.6, fill_alpha=0.6, size=8, legend_label = "True Positive")
    FP_glyph = p.star('day_obs', 'cal_mag', color='#FF5733',  source=dataFP, line_alpha=0.6, fill_alpha=0.6, size=8, legend_label = "False Positive")
    TN_glyph = p.circle('day_obs', 'cal_mag', color='#C70039',  source=dataTN, line_alpha=0.6, fill_alpha=0.6, size=8, legend_label = "True Negative")
    FN_glyph = p.square('day_obs', 'cal_mag', color='#b3b6b7',  source=dataFN, line_alpha=0.6, fill_alpha=0.6, size=8, legend_label = "False Negative")

    p.add_tools(HoverTool(tooltips=f"""
        <div>
            <div>
                <span style='font-size: 14px; color: #224499'> Real class : @true_class</span>
                <br>                
                <span style='font-size: 14px; color: #224499'> filter : @band</span>
                <br>
                <span style='font-size: 14px; color: #224499'> pred : @proba_transient_{model_name}</span>
                <img src='@image_coadd' height="60" width="60" style='float: left; margin: 5px 5px 5px 5px'></img>
                <img src='@image_science' height="60" width="60" style='float: left; margin: 5px 5px 5px 5px'></img>
                <img src='@image' height="60" width="60" style='float: left; margin: 5px 5px 5px 5px'></img>
            </div>
        </div>
        """))
    # Customize plot
    p.yaxis.axis_label = "Magnitude"
    p.xaxis.axis_label = "Time"
    p.y_range.flipped = True  # Magnitudes are usually plotted inversely
    curdoc().theme = 'dark_minimal'
    p.legend.location = "top_right"
    p.legend.title = "Classification"
    p.legend.click_policy = "hide"

    # Show plot
    output_notebook()
    show(p, notebook_handle =True)
    return p