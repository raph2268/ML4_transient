import pandas as pd
import numpy as np
import pickle
from astropy.time import Time
from astropy import units as u
from bokeh.io import output_notebook, show, curdoc
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, HoverTool

from utils import embeddable_image
from df_bogus import df_proba

data_path ='/sps/lsst/groups/transients/HSC/fouchez/raphael'
with open("saved/df_asso.pkl", "rb") as pickle_file:
    df_asso= pickle.load(pickle_file)

def retrieve_LC(diaObjectId):

    df_LC = df_asso[df_asso['diaObjectId']==str(diaObjectId)]

    return df_LC

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
from astropy import units as u
from astropy.time import Time
data_path ='/sps/lsst/groups/transients/HSC/fouchez/raphael'

def plot_LC(df, model, model_name = None): 
    cutouts_array = []

    for idx, row in df.iterrows(): 
        visit = row['visit']
        features = pd.read_pickle(f'{data_path}/Detected_obj_sources_visit_{visit}_UDEEP_df.pkl')
        idx_img = features[features['diaSourceId'].astype(str)==row['diaSourceId']].index
        # Load coadd cutouts using pickle
        with open(f"{data_path}/Detected_obj_sources_visit_{visit}_UDEEP.pkl", 'rb') as f:
            diff = pickle.load(f)
        with open(f"{data_path}/Coadd_detected_obj_sources_visit_{visit}_UDEEP.pkl", 'rb') as f:
            coadd = pickle.load(f)

        # Load science cutouts using pickle
        with open(f"{data_path}/Science_detected_obj_sources_visit_{visit}_UDEEP.pkl", 'rb') as f:
            science = pickle.load(f)
        diff = np.array(diff)
        coadd = np.array(coadd)
        science = np.array(science)
    
        diff = diff.reshape(-1,30,30,1)
        diff = (diff- diff.min(axis=(1,2)).reshape((-1,1,1,1)) )/ (diff.max(axis=(1,2)).reshape((-1,1,1,1))- diff.min(axis=(1,2)).reshape((-1,1,1,1)))
        coadd = coadd.reshape(-1,30,30,1)
        coadd = (coadd- coadd.min(axis=(1,2)).reshape((-1,1,1,1)) )/ (coadd.max(axis=(1,2)).reshape((-1,1,1,1))- coadd.min(axis=(1,2)).reshape((-1,1,1,1)))
        science = science.reshape(-1,30,30,1)
        science = (science- science.min(axis=(1,2)).reshape((-1,1,1,1)) )/ (science.max(axis=(1,2)).reshape((-1,1,1,1))- science.min(axis=(1,2)).reshape((-1,1,1,1)))
        cutouts_array.append(diff[idx_img]) 
        df.at[idx, 'image'] = embeddable_image(diff[idx_img])
        df.at[idx, 'image_coadd'] = embeddable_image(coadd[idx_img])
        df.at[idx, 'image_science'] = embeddable_image(science[idx_img])
    
    cutouts_array = np.array(np.vstack((*cutouts_array,)))
    if model_name is None: 
        df = df_proba(df, cutouts_array, df['y_test'].values , model)

        model_name = model
        
    else : 
        y_test_tensor  = torch.tensor(df['y_test'].values, dtype=torch.long)
        df = df_proba_pytorch(df, cutouts_array, y_test_tensor , model, model_name)
    
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