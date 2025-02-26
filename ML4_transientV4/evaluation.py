import tensorflow as tf
from tensorflow import keras

import torch
import torch.nn as nn

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys,os
from matplotlib import cm
from sklearn.mixture import GaussianMixture
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV

from PIL import ImageOps
from PIL import Image
import io
import base64

import h5py

import umap

import pandas as pd

from mpl_toolkits.axes_grid1 import make_axes_locatable

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #Permet d'enlever les Warning tensorflow


np.random.seed(1) # NumPy
import random
random.seed(2) # Python
tf.random.set_seed(3) # Tensorflow 

#sys.path.append('..')
#sys.path.append('/sps/lsst/users/mguy/bogus_detection/bogus_detection_v8/df_model')

from utils import datadir, rundir, pddir, embeddable_image
from Retrieve_data import retrieve_data_gen3
from CNN_model import CNNHyperModel
from df_bogus import df_proba, df_proba_pytorch,  retrieve_LC, df_association

from bokeh.plotting import figure, output_file, save, curdoc
from bokeh.models import HoverTool, ColumnDataSource, CategoricalColorMapper, Legend, Div, Title, Tabs, Button, CustomJS, ColorBar, LinearAxis, Range1d, LegendItem
from bokeh.palettes import Spectral10, Viridis, Spectral, Category10, brewer, tol, Turbo256
from bokeh.transform import factor_cmap, factor_mark
from bokeh.layouts import column, gridplot, row
from bokeh.io import output_notebook, show
from bokeh.transform import linear_cmap
import colorcet as cc

import pickle

import warnings
warnings.filterwarnings("ignore", message="out of range integer may result in loss of precision", category=UserWarning)

class evaluating_model_gen3:
    def __init__(self, model, visit, visual_class=None):
        """
        Initializes the evaluating_model class.

        Parameters:
        model (str): Name of the model to be evaluated.
        visit (int): Visit number for data retrieval.
        visual_class (optional): Class for visualizations (if any).
        """
        curdoc().theme = 'dark_minimal'

        self.test_size = .99
        self.model = model
        self.load_model = tf.keras.models.load_model(f'models/{model}.h5')
        # Retrieve and prepare data
        self.visit = visit
        self.data = retrieve_data_gen3(visits = visit)
        
        # Split data into training and testing sets
        self.df_train, df_test, self.x_train, self.x_test, self.y_train, self.y_test = self.data.data(test_size=self.test_size)
        df_test = df_proba(df_test, self.x_test, self.y_test, model)
        
        self.df_test = df_association(df_test)
        # Extract features from the model
        self.features_test = keras.Model(inputs=self.load_model.inputs,
                                        outputs=[layer.output for layer in self.load_model.layers])(self.x_test)
        

        # Apply UMAP for dimensionality reduction on the second-to-last layer
        self.reducer_umapDNN = umap.UMAP(random_state=1)
        self.umapDNN_data = self.reducer_umapDNN.fit_transform(self.features_test[8])
        dim1, dim2 = np.hsplit(self.umapDNN_data, 2)
        self.df_test['umapDNN_Dim_1'], self.df_test['umapDNN_Dim_2'] = dim1, dim2
        self.df_test_full = self.df_test
        # Set plotting parameters
        self.width = 800
        self.height = 550
        self.nbre_pts = 1500
        self.df_test['image'] = [embeddable_image(data) for data in self.x_test]

        self.data_glob_umap = ColumnDataSource(self.df_test[0:self.nbre_pts])
   
        self.dataTP_umap_CNN = ColumnDataSource(self.df_test.loc[self.df_test.true_positive == 1][0:self.nbre_pts])
        self.dataTN_umap_CNN = ColumnDataSource(self.df_test.loc[self.df_test.true_negative == 1][0:self.nbre_pts])
        self.dataFP_umap_CNN = ColumnDataSource(self.df_test.loc[self.df_test.false_positive == 1][0:self.nbre_pts])
        self.dataFN_umap_CNN = ColumnDataSource(self.df_test.loc[self.df_test.false_negative == 1][0:self.nbre_pts])

    def create_umap_data(self, inspected_LC): 
        cutouts_array = np.array(inspected_LC['image_np'].tolist())
        inspected_LC['image'] = [embeddable_image(data) for data in cutouts_array]
        inspected_LC = df_proba(inspected_LC, cutouts_array, inspected_LC['y_test'], self.model)
        LC_features = keras.Model(inputs=self.load_model.inputs,
                                outputs=[layer.output for layer in self.load_model.layers])(cutouts_array)
        LC_umap = self.reducer_umapDNN.transform(LC_features[8])
        new_dim1, new_dim2 = np.hsplit(LC_umap, 2)
        # Add the new UMAP dimensions to a new DataFrame or append to existing df_test
        inspected_LC['umapDNN_Dim_1'], inspected_LC['umapDNN_Dim_2'] = new_dim1, new_dim2
        
        self.dataTP_umap_LC = ColumnDataSource(inspected_LC.loc[inspected_LC.true_positive == 1][0:self.nbre_pts])
        self.dataTN_umap_LC = ColumnDataSource(inspected_LC.loc[inspected_LC.true_negative == 1][0:self.nbre_pts])
        self.dataFP_umap_LC = ColumnDataSource(inspected_LC.loc[inspected_LC.false_positive == 1][0:self.nbre_pts])
        self.dataFN_umap_LC = ColumnDataSource(inspected_LC.loc[inspected_LC.false_negative == 1][0:self.nbre_pts])
    
    
    def plot_confusion_matrix(self, plot=True, SNR_cut = None):
        
        if SNR_cut is not None:
            df = self.df_test[(self.df_test['snr']>= SNR_cut) |
                (self.df_test['snr'] <= -SNR_cut) ]
        else : df = self.df_test
           
        # Calculate the total number of points for percentages
        total_points = len(df)

        # Calculate counts for TP, TN, FP, FN
        count_tp = len(df.loc[df.true_positive == 1])
        count_tn = len(df.loc[df.true_negative == 1])
        count_fp = len(df.loc[df.false_positive == 1])
        count_fn = len(df.loc[df.false_negative == 1])

        # Calculate sensitivity, specificity, and precision
        sensitivity = count_tp / (count_tp + count_fn) if (count_tp + count_fn) > 0 else 0
        specificity = count_tn / (count_tn + count_fp) if (count_tn + count_fp) > 0 else 0
        precision = count_tp / (count_tp + count_fp) if (count_tp + count_fp) > 0 else 0
        neg_predictive_value = count_tn / (count_tn + count_fn) if (count_tn + count_fn) > 0 else 0
        accuracy = (count_tn + count_tp)/ (count_tn + count_fn + count_tp + count_fp) if (count_tn + count_fn + count_tp + count_fp) > 0 else 0

        # Calculate percentages
        percentage_tp = "TP :"+str(round((count_tp / total_points) * 100, 2))+"%"
        percentage_tn = "TN :"+str(round((count_tn / total_points) * 100, 2))+"%"
        percentage_fp = "FP :"+str(round((count_fp / total_points) * 100, 2))+"%"
        percentage_fn = "FN :"+str(round((count_fn / total_points) * 100, 2))+"%"
        percentage_sensitivity = "Sensitivity :\n"+str(round(sensitivity * 100, 2))+"%"
        percentage_specificity = "Specificity :\n"+str(round(specificity * 100, 2))+"%"
        percentage_precision = "Precision :\n"+str(round(precision * 100, 2))+"%"
        percentage_neg_predictive = "Negative\npredictive :\n"+str(round(neg_predictive_value * 100, 2))+"%"
        percentage_accuracy = "Accuracy:\n"+str(round(accuracy * 100, 2))+"%"

        # Data for the confusion matrix plot
        labels = ['True Positive', 'False Negative', 'False Positive', 'True Negative', 'Sensitivity', 'Specificity', 'Precision', 'Negative Predictive Value', 'Accuracy']
        counts = [count_tp, count_fn, count_fp, count_tn, None, None, None, None, None]
        percentages = [percentage_tp, percentage_fn, percentage_fp, percentage_tn, percentage_sensitivity, percentage_specificity, percentage_precision, percentage_neg_predictive, percentage_accuracy]
        colors = ['#FFC300', '#b3b6b7',  '#FF5733', '#C70039', '#ECEDED', '#ECEDED', '#ECEDED', '#ECEDED', '#ECEDED']  # Adjust colors as needed

        # Prepare the ColumnDataSource for Bokeh
        confusion_data = ColumnDataSource(data=dict(
            labels=labels,
            counts=counts,
            percentages=percentages,
            x=[0, 1, 0, 1, 2, 2, 0, 1, 2],  # Adjust x positions
            y=[2, 2, 1, 1, 1, 2, 0, 0, 0],  # Adjust y positions
            colors=colors
        ))

        # Create a figure for confusion matrix
        p = figure(title="Confusion Matrix", x_range=(-0.5, 2.5), y_range=(-0.5, 2.5),
                   width=400, height=400, tools="")

        # Add rectangles for each class (TP, TN, FP, FN, Sensitivity, Specificity, Precision)
        p.rect(x='x', y='y', width=1, height=1, source=confusion_data,
               color='colors', alpha=0.9, line_color="colors", line_width=2)

        # Add text for percentage in the center of each square
        p.text(x='x', y='y', text='percentages', source=confusion_data,text_font_size="16pt", text_align="center", text_baseline="middle", alpha=0.75)

        # Add hover tool to display counts and percentages
        hover = HoverTool(tooltips=[('Class', '@labels'), ('Count', '@counts'), ('Percentage', '@percentages')])
        p.add_tools(hover)

        # Remove grid lines and axis labels for cleaner look
        p.xgrid.grid_line_color = None
        p.ygrid.grid_line_color = None
        p.axis.visible = False

        if plot: 
            output_notebook()  # Use this line if running in a Jupyter notebook
            show(p, notebook_handle=True)

        return p

    def injected_pred_distrib(self, SNR_cut=None):
        # Apply SNR cut if specified
        if SNR_cut is not None:
            df = self.df_test[(self.df_test['snr']>= SNR_cut) |
                (self.df_test['snr'] <= -SNR_cut) ]
        else : df = self.df_test

        # Filter data
        idx_transient = df[f'proba_transient_{self.model}'].loc[df['y_test'] == 1]
        bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

        # Compute histograms
        counts_t, bins_t = np.histogram(df[f'proba_transient_{self.model}'], bins=bins)
        counts_1, bins_1 = np.histogram(idx_transient, bins=bins)

        # Prepare data for Bokeh
        source_all = ColumnDataSource(data=dict(x=bins[:-1], top=counts_t, count=counts_t))
        source_injected = ColumnDataSource(data=dict(x=bins[:-1], top=counts_1, count=counts_1))

        # Create a new Bokeh figure
        p = figure(
            title='All data vs injection probability',
            x_axis_label='Probability output',
            y_axis_label='Count',
            height=400,
            width=400)

        # Add bars for all data and injected data
        p.vbar(x='x', top='top', width=0.08, color='lightgray', source=source_all, legend_label='All data')
        p.vbar(x='x', top='top', width=0.08, color='#5b75cd', source=source_injected, legend_label='Injected data')
        
        # Add hover tools
        hover_all = HoverTool(
            renderers=[p.renderers[0]],  # Specify the renderer (vbar for all data)
            tooltips=[("Probability bin", "@x"), ("Count", "@count")]
        )
        hover_injected = HoverTool(
            renderers=[p.renderers[1]],  # Specify the renderer (vbar for injected data)
            tooltips=[("Probability bin", "@x"), ("Injected Count", "@count")]
        )
        p.add_tools(hover_all, hover_injected)
        
        # Customize plot
        p.legend.click_policy="hide"
        p.legend.location = 'top_left'
        p.legend.label_text_color = 'white'
        p.xgrid.grid_line_color = None
        p.ygrid.grid_line_color = None
        p.axis.axis_label_text_color = 'white'
        p.axis.major_label_text_color = 'white'
        p.title.text_color = 'white'

        # Show plot (you can also return p instead)
        output_notebook()  # Use this line if running in a Jupyter notebook
        show(p, notebook_handle=True)

        return p
        
    # Plot UMAP with true labels
    def Umap_true_label(self, plot = True, SNR_cut = None, inspected_LC = None):
        if SNR_cut is not None:
            self.dataTP_umap_CNN = self.filter_ColumnDataSource(self.dataTP_umap_CNN, SNR_cut)
            self.dataFP_umap_CNN = self.filter_ColumnDataSource(self.dataFP_umap_CNN, SNR_cut)
            self.dataTN_umap_CNN = self.filter_ColumnDataSource(self.dataTN_umap_CNN, SNR_cut)
            self.dataFN_umap_CNN = self.filter_ColumnDataSource(self.dataFN_umap_CNN, SNR_cut)
           
        u8 = figure(title='Results of umap after CNN', width=self.width, height=self.height, tools=(['tap', 'lasso_select', 'pan', 'wheel_zoom', 'reset']))

        title=f"Distribution of true label, after the CNN, {self.model}"

        u8.add_layout(Title(text=title, text_font_style="normal"), "above")

        u8.add_tools(HoverTool(tooltips=f"""
        <div>
            <div>
                <img 
                    src='@image' height="60" width="60" style='float: left; margin: 5px 5px 5px 5px'
                    ></img>
            <div>
                <span style='font-size: 14px; color: #224499'>Predicted class:</span>
                <span style='font-size: 14px'>@bogus_transient_pred_{self.model}</span><br>
                <span style='font-size: 14px; color: #224499'>Obj id:</span>
                <span style='font-size: 14px'>@diaObjectId</span><br>
                <span style='font-size: 14px; color: #224499'>Nbr src in LC:</span>
                <span style='font-size: 14px'>@nDiaSources</span><br>                
            </div>
        </div>
        """))
        #color_mapper = linear_cmap(field_name=f'y_pred_{self.model}', palette=Category10[4], low=0, high=1)
        
        u8.x('umapDNN_Dim_1', 'umapDNN_Dim_2', color='#FFC300',  source=self.dataTP_umap_CNN, line_alpha=0.6, fill_alpha=0.6, size=8, legend_label = "True Positive")
        u8.star('umapDNN_Dim_1', 'umapDNN_Dim_2', color='#FF5733',  source=self.dataFP_umap_CNN, line_alpha=0.6, fill_alpha=0.6, size=8, legend_label = "False Positive")
        u8.circle('umapDNN_Dim_1', 'umapDNN_Dim_2', color='#C70039',  source=self.dataTN_umap_CNN, line_alpha=0.6, fill_alpha=0.6, size=8, legend_label = "True Negative")
        u8.square('umapDNN_Dim_1', 'umapDNN_Dim_2', color='#b3b6b7',  source=self.dataFN_umap_CNN, line_alpha=0.6, fill_alpha=0.6, size=8, legend_label = "False Negative")
        
        if inspected_LC is not None:
            self.create_umap_data(inspected_LC)
            u8.x('umapDNN_Dim_1', 'umapDNN_Dim_2', color='#FFC300',  source=self.dataTP_umap_LC, line_alpha=0.6, fill_alpha=1, size=18, legend_label = "LC True Positive")
            u8.star('umapDNN_Dim_1', 'umapDNN_Dim_2', color='#FF5733',  source=self.dataFP_umap_LC, line_alpha=0.6, fill_alpha=1, size=18, legend_label = "LC False Positive")
            u8.circle('umapDNN_Dim_1', 'umapDNN_Dim_2', color='#C70039',  source=self.dataTN_umap_LC, line_alpha=0.6, fill_alpha=1, size=18, legend_label = "LC True Negative")
            u8.square('umapDNN_Dim_1', 'umapDNN_Dim_2', color='#b3b6b7',  source=self.dataFN_umap_LC, line_alpha=0.6, fill_alpha=1, size=18, legend_label = "LC False Negative")
        
        u8.add_layout(u8.legend[0])
        
        u8.legend.title = 'True label'
        u8.legend.click_policy="hide"


        # Create a Div widget for displaying selected sources
        div = Div(width=400, height=u8.height, height_policy="fixed", text="Select points on the plot")
        src = self.dataFP_umap_CNN
        # CustomJS callback to update the Div with selected fruit names
        cb = CustomJS(args=dict(src=src, div=div), code='''
            var sel_inds = src.selected.indices;
            var objId = src.data['diaObjectId'];
            var text = "Selected obj id :<br>";

            for (var i = 0; i < sel_inds.length; i++) {
                var sel_i = sel_inds[i];
                text += objId[sel_i] + "<br>";
            }

            div.text = text;
        ''')

        # Attach the callback to the selected indices change event
        src.selected.js_on_change('indices', cb)
        layout = row(u8, div)

        if plot: 
            output_notebook()  # Use this line if running in a Jupyter notebook
            show(layout, notebook_handle=True)
        
        if SNR_cut is not None:
            self.dataTP_umap_CNN = ColumnDataSource(self.df_test.loc[self.df_test.true_positive == 1][0:self.nbre_pts])
            self.dataTN_umap_CNN = ColumnDataSource(self.df_test.loc[self.df_test.true_negative == 1][0:self.nbre_pts])
            self.dataFP_umap_CNN = ColumnDataSource(self.df_test.loc[self.df_test.false_positive == 1][0:self.nbre_pts])
            self.dataFN_umap_CNN = ColumnDataSource(self.df_test.loc[self.df_test.false_negative == 1][0:self.nbre_pts])
        
        return u8
   

    def Umap_SNR(self, plot=True, snr_range=(-20, 20), SNR_cut = None, inspected_LC = None):
        if SNR_cut is not None:
            self.dataTP_umap_CNN = self.filter_ColumnDataSource(self.dataTP_umap_CNN, SNR_cut)
            self.dataFP_umap_CNN = self.filter_ColumnDataSource(self.dataFP_umap_CNN, SNR_cut)
            self.dataTN_umap_CNN = self.filter_ColumnDataSource(self.dataTN_umap_CNN, SNR_cut)
            self.dataFN_umap_CNN = self.filter_ColumnDataSource(self.dataFN_umap_CNN, SNR_cut)
            
        # Main UMAP plot
        u10 = figure(title='Results of UMAP after CNN', width=self.width, height=self.height, tools=('pan, wheel_zoom, reset'))

        title = f"Distribution of SNR, after the CNN, {self.model}"
        u10.add_layout(Title(text=title, text_font_style="normal"), "above")
        self.df_test['clipped_SNR'] = self.df_test['snr'].clip(*snr_range)

        u10.add_tools(HoverTool(tooltips=f"""
        <div>
            <div>
                <img src='@image' height="60" width="60" style='float: left; margin: 5px 5px 5px 5px'></img>
            <div>
                <span style='font-size: 14px; color: #224499'>Predicted class:</span>
                <span style='font-size: 14px'>@bogus_transient_pred_{self.model}</span><br>
                <span style='font-size: 14px; color: #224499'>SNR:</span>
                <span style='font-size: 14px'>@snr</span><br>
            </div>
        </div>
        """))

        # Create a linear color mapper based on SNR values
        color_mapper = linear_cmap(field_name='snr', palette=cc.kbc, low=self.df_test['clipped_SNR'].min(), high=self.df_test['clipped_SNR'].max())

        u10.x('umapDNN_Dim_1', 'umapDNN_Dim_2', color=color_mapper,  source=self.dataTP_umap_CNN, line_alpha=0.6, fill_alpha=0.6, size=8, legend_label = "True Positive")
        u10.star('umapDNN_Dim_1', 'umapDNN_Dim_2', color=color_mapper,  source=self.dataFP_umap_CNN, line_alpha=0.6, fill_alpha=0.6, size=8, legend_label = "False Positive")
        u10.circle('umapDNN_Dim_1', 'umapDNN_Dim_2', color=color_mapper,  source=self.dataTN_umap_CNN, line_alpha=0.6, fill_alpha=0.6, size=8, legend_label = "True Negative")
        u10.square('umapDNN_Dim_1', 'umapDNN_Dim_2', color=color_mapper,  source=self.dataFN_umap_CNN, line_alpha=0.6, fill_alpha=0.6, size=8, legend_label = "False Negative")
        
        if inspected_LC is not None:
            self.create_umap_data(inspected_LC)
            u10.x('umapDNN_Dim_1', 'umapDNN_Dim_2', color=color_mapper,  source=self.dataTP_umap_LC, line_alpha=0.6, fill_alpha=1, size=18, legend_label = "LC True Positive")
            u10.star('umapDNN_Dim_1', 'umapDNN_Dim_2', color=color_mapper,  source=self.dataFP_umap_LC, line_alpha=0.6, fill_alpha=1, size=18, legend_label = "LC False Positive")
            u10.circle('umapDNN_Dim_1', 'umapDNN_Dim_2', color=color_mapper,  source=self.dataTN_umap_LC, line_alpha=0.6, fill_alpha=1, size=18, legend_label = "LC True Negative")
            u10.square('umapDNN_Dim_1', 'umapDNN_Dim_2', color=color_mapper,  source=self.dataFN_umap_LC, line_alpha=0.6, fill_alpha=1, size=18, legend_label = "LC False Negative")
        

        # Add a color bar to represent the SNR values
        color_bar = ColorBar(color_mapper=color_mapper['transform'], width=8, location=(0, 0))
        u10.add_layout(color_bar, 'right')

        u10.add_layout(u10.legend[0])
        u10.legend.title = 'True label'
        u10.legend.click_policy = "hide"

        # Handle NaN values by dropping them
        valid_snr = self.df_test['snr'].dropna()

        # Create a vertical histogram of the SNR values
        hist, edges = np.histogram(valid_snr, bins=50, range = (self.df_test['clipped_SNR'].min(), self.df_test['clipped_SNR'].max()))

        bin_midpoints = (edges[:-1] + edges[1:]) / 2

        # Create a new ColumnDataSource for the histogram, including bin midpoints
        hist_source = ColumnDataSource(data=dict(
            top=hist,
            bottom=np.zeros_like(hist),
            left=edges[:-1],
            right=edges[1:],
            snr_mid=bin_midpoints  # Midpoints of the bins for color mapping
        ))
        snr_mapper = linear_cmap(field_name='snr_mid', palette=cc.kbc, low=self.df_test['clipped_SNR'].min(), high=self.df_test['clipped_SNR'].max())

        # Apply color mapping based on the midpoints of the SNR bins

        snr_plot = figure(width=200, height=self.height, title="SNR Distribution", tools="")

        # Use the color mapper for the histogram
        snr_plot.quad(top='top', bottom='bottom', left='left', right='right', source=hist_source, fill_color=snr_mapper, line_color="white")

        # Combine the plots
        layout = gridplot([[u10, snr_plot]], toolbar_location="right", merge_tools=True)

        if plot:
            output_notebook()
            show(layout, notebook_handle = True)
        if SNR_cut is not None:
            self.dataTP_umap_CNN = ColumnDataSource(self.df_test.loc[self.df_test.true_positive == 1][0:self.nbre_pts])
            self.dataTN_umap_CNN = ColumnDataSource(self.df_test.loc[self.df_test.true_negative == 1][0:self.nbre_pts])
            self.dataFP_umap_CNN = ColumnDataSource(self.df_test.loc[self.df_test.false_positive == 1][0:self.nbre_pts])
            self.dataFN_umap_CNN = ColumnDataSource(self.df_test.loc[self.df_test.false_negative == 1][0:self.nbre_pts])
        
        return layout
    
    
    def filter_ColumnDataSource(self, CDS, SNR_cut):
        try:
            # Convert ColumnDataSource to DataFrame
            data = CDS.data
            df_from_cds = pd.DataFrame(data)

            # Drop unnecessary columns
            columns_to_drop = ['level_0', 'index']  # Add any other columns that are not needed
            columns_to_drop = [col for col in columns_to_drop if col in df_from_cds.columns]

            if columns_to_drop:
                df_from_cds = df_from_cds.drop(columns=columns_to_drop)

            # Ensure the SNR column exists and is numeric
            snr_column = 'snr'
            if snr_column not in df_from_cds.columns:
                raise KeyError(f"Column '{snr_column}' not found in DataFrame")

            df_from_cds[snr_column] = pd.to_numeric(df_from_cds[snr_column], errors='coerce')

            # Filter the DataFrame
            filtered_df = df_from_cds[
                (df_from_cds[snr_column] >= SNR_cut) |
                (df_from_cds[snr_column] <= -SNR_cut)
            ]

            # Reset index and avoid adding a new column
            filtered_df = filtered_df.reset_index(drop=True)

            # Create a new ColumnDataSource
            new_ColumnDataSource = ColumnDataSource(filtered_df)

            return new_ColumnDataSource

        except Exception as e:
            print(f"An error occurred: {e}")
            return None

    
    
    
    
    
    
    
    
    
    
    
    
    
        
        
        
        
        
       
    
    
    
class evaluating_model_pytorch_gen3:
    def __init__(self, model, visit, model_name, visual_class=None):
        """
        Initializes the EvaluatingModel class.

        Parameters:
        model_path (str): Path to the PyTorch model to be evaluated.
        visit (int): Visit number for data retrieval.
        visual_class (optional): Class for visualizations (if any).
        """
        curdoc().theme = 'dark_minimal'

        self.test_size = .99
        self.model = model
        self.model_name = model_name
        # Retrieve and prepare data
        self.visit = visit
        self.data = retrieve_data_gen3(visits=visit)
        
        # Split data into training and testing sets
        self.df_train, df_test, self.x_train, x_test, self.y_train, y_test = self.data.data(test_size=self.test_size)
        self.df_test_spy, self.x_test, self.y_test = spy(df_test, x_test, y_test)
        self.df_test = df_proba_pytorch(self.df_test_spy, self.x_test, self.y_test, self.model, self.model_name)
        self.df_test = df_association(self.df_test)

        # Extract features from the model
        self.features_test = self.extract_features(self.x_test)

        # Add image data for visualization
        self.df_test['image'] = [embeddable_image(data) for data in self.x_test]
        
        # Apply UMAP for dimensionality reduction on the second-to-last layer
        self.reducer_umapDNN = umap.UMAP(random_state=1)
        self.umapDNN_data = self.reducer_umapDNN.fit_transform(self.features_test)
        dim1, dim2 = np.hsplit(self.umapDNN_data, 2)
        self.df_test['umapDNN_Dim_1'], self.df_test['umapDNN_Dim_2'] = dim1, dim2
        self.df_test_full = self.df_test

        # Set plotting parameters
        self.width = 800
        self.height = 550
        self.nbre_pts = 3000
        self.data_glob_umap = ColumnDataSource(self.df_test[0:self.nbre_pts])
        self.dataTP_umap_CNN = ColumnDataSource(self.df_test.loc[self.df_test.true_positive == 1][0:self.nbre_pts])
        self.dataTN_umap_CNN = ColumnDataSource(self.df_test.loc[self.df_test.true_negative == 1][0:self.nbre_pts])
        self.dataFP_umap_CNN = ColumnDataSource(self.df_test.loc[self.df_test.false_positive == 1][0:self.nbre_pts])
        self.dataFN_umap_CNN = ColumnDataSource(self.df_test.loc[self.df_test.false_negative == 1][0:self.nbre_pts])

    def extract_features(self, x_test):
        """
        Extract features from the model.

        Parameters:
        x_test (torch.Tensor): The input data for feature extraction.

        Returns:
        np.ndarray: The features extracted from the model.
        """
        with torch.no_grad():
            X_test = [np.transpose(i, (2, 0, 1)) for i in x_test]

            x_test_tensor = torch.tensor(np.array(X_test), dtype=torch.float32)
            features = self.model(x_test_tensor)

        return features.numpy()
    
    def plot_confusion_matrix(self, plot=True, SNR_cut = None):
        
        if SNR_cut is not None:
            df = self.df_test[(self.df_test['snr']>= SNR_cut) |
                (self.df_test['snr'] <= -SNR_cut) ]
        else : df = self.df_test
           
        # Calculate the total number of points for percentages
        total_points = len(df)

        # Calculate counts for TP, TN, FP, FN
        count_tp = len(df.loc[df.true_positive == 1])
        count_tn = len(df.loc[df.true_negative == 1])
        count_fp = len(df.loc[df.false_positive == 1])
        count_fn = len(df.loc[df.false_negative == 1])

        # Calculate sensitivity, specificity, and precision
        sensitivity = count_tp / (count_tp + count_fn) if (count_tp + count_fn) > 0 else 0
        specificity = count_tn / (count_tn + count_fp) if (count_tn + count_fp) > 0 else 0
        precision = count_tp / (count_tp + count_fp) if (count_tp + count_fp) > 0 else 0
        neg_predictive_value = count_tn / (count_tn + count_fn) if (count_tn + count_fn) > 0 else 0
        accuracy = (count_tn + count_tp)/ (count_tn + count_fn + count_tp + count_fp) if (count_tn + count_fn + count_tp + count_fp) > 0 else 0

        # Calculate percentages
        percentage_tp = "TP :"+str(round((count_tp / total_points) * 100, 2))+"%"
        percentage_tn = "TN :"+str(round((count_tn / total_points) * 100, 2))+"%"
        percentage_fp = "FP :"+str(round((count_fp / total_points) * 100, 2))+"%"
        percentage_fn = "FN :"+str(round((count_fn / total_points) * 100, 2))+"%"
        percentage_sensitivity = "Sensitivity :\n"+str(round(sensitivity * 100, 2))+"%"
        percentage_specificity = "Specificity :\n"+str(round(specificity * 100, 2))+"%"
        percentage_precision = "Precision :\n"+str(round(precision * 100, 2))+"%"
        percentage_neg_predictive = "Negative\npredictive :\n"+str(round(neg_predictive_value * 100, 2))+"%"
        percentage_accuracy = "Accuracy:\n"+str(round(accuracy * 100, 2))+"%"

        # Data for the confusion matrix plot
        labels = ['True Positive', 'False Negative', 'False Positive', 'True Negative', 'Sensitivity', 'Specificity', 'Precision', 'Negative Predictive Value', 'Accuracy']
        counts = [count_tp, count_fn, count_fp, count_tn, None, None, None, None, None]
        percentages = [percentage_tp, percentage_fn, percentage_fp, percentage_tn, percentage_sensitivity, percentage_specificity, percentage_precision, percentage_neg_predictive, percentage_accuracy]
        colors = ['#FFC300', '#b3b6b7',  '#FF5733', '#C70039', '#ECEDED', '#ECEDED', '#ECEDED', '#ECEDED', '#ECEDED']  # Adjust colors as needed

        # Prepare the ColumnDataSource for Bokeh
        confusion_data = ColumnDataSource(data=dict(
            labels=labels,
            counts=counts,
            percentages=percentages,
            x=[0, 1, 0, 1, 2, 2, 0, 1, 2],  # Adjust x positions
            y=[2, 2, 1, 1, 1, 2, 0, 0, 0],  # Adjust y positions
            colors=colors
        ))

        # Create a figure for confusion matrix
        p = figure(title="Confusion Matrix", x_range=(-0.5, 2.5), y_range=(-0.5, 2.5),
                   width=400, height=400, tools="")

        # Add rectangles for each class (TP, TN, FP, FN, Sensitivity, Specificity, Precision)
        p.rect(x='x', y='y', width=1, height=1, source=confusion_data,
               color='colors', alpha=0.9, line_color="colors", line_width=2)

        # Add text for percentage in the center of each square
        p.text(x='x', y='y', text='percentages', source=confusion_data,text_font_size="16pt", text_align="center", text_baseline="middle", alpha=0.75)

        # Add hover tool to display counts and percentages
        hover = HoverTool(tooltips=[('Class', '@labels'), ('Count', '@counts'), ('Percentage', '@percentages')])
        p.add_tools(hover)

        # Remove grid lines and axis labels for cleaner look
        p.xgrid.grid_line_color = None
        p.ygrid.grid_line_color = None
        p.axis.visible = False

        if plot: 
            output_notebook()  # Use this line if running in a Jupyter notebook
            show(p, notebook_handle=True)

        return p
    def injected_pred_distrib(self, SNR_cut=None):
        # Apply SNR cut if specified
        if SNR_cut is not None:
            df = self.df_test[(self.df_test['snr']>= SNR_cut) |
                (self.df_test['snr'] <= -SNR_cut) ]
        else : df = self.df_test

        # Filter data
        idx_transient = df[f'proba_transient_{self.model_name}'].loc[df['y_test'] == 1]
        bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

        # Compute histograms
        counts_t, bins_t = np.histogram(df[f'proba_transient_{self.model_name}'], bins=bins)
        counts_1, bins_1 = np.histogram(idx_transient, bins=bins)

        # Prepare data for Bokeh
        source_all = ColumnDataSource(data=dict(x=bins[:-1], top=counts_t, count=counts_t))
        source_injected = ColumnDataSource(data=dict(x=bins[:-1], top=counts_1, count=counts_1))

        # Create a new Bokeh figure
        p = figure(
            title='All data vs injection probability',
            x_axis_label='Probability output',
            y_axis_label='Count',
            height=400,
            width=400        )

        # Add bars for all data and injected data
        p.vbar(x='x', top='top', width=0.08, color='lightgray', source=source_all, legend_label='All data')
        p.vbar(x='x', top='top', width=0.08, color='#5b75cd', source=source_injected, legend_label='Injected data')
        
        # Add hover tools
        hover_all = HoverTool(
            renderers=[p.renderers[0]],  # Specify the renderer (vbar for all data)
            tooltips=[("Probability bin", "@x"), ("Count", "@count")]
        )
        hover_injected = HoverTool(
            renderers=[p.renderers[1]],  # Specify the renderer (vbar for injected data)
            tooltips=[("Probability bin", "@x"), ("Injected Count", "@count")]
        )
        p.add_tools(hover_all, hover_injected)
        
        # Customize plot
        p.legend.click_policy="hide"
        p.legend.location = 'top_left'
        p.legend.label_text_color = 'white'
        p.xgrid.grid_line_color = None
        p.ygrid.grid_line_color = None
        p.axis.axis_label_text_color = 'white'
        p.axis.major_label_text_color = 'white'
        p.title.text_color = 'white'

        # Show plot (you can also return p instead)
        output_notebook()  # Use this line if running in a Jupyter notebook
        show(p, notebook_handle=True)

        return p
    
    def create_umap_data(self, inspected_LC): 
        cutouts_array = np.array(inspected_LC['image_np'].tolist())
        inspected_LC['image'] = [embeddable_image(data) for data in cutouts_array]
        y_test_tensor  = torch.tensor(inspected_LC['y_test'].values, dtype=torch.long)
        inspected_LC = df_proba_pytorch(inspected_LC, cutouts_array, y_test_tensor , self.model, self.model_name)
        cutouts_array = [np.transpose(i, (2, 0, 1)) for i in cutouts_array]
        with torch.no_grad():
            cutouts_tensor = torch.tensor(cutouts_array).float()
            LC_features = self.model.forward(cutouts_tensor)
            
        LC_umap = self.reducer_umapDNN.transform(LC_features)
        new_dim1, new_dim2 = np.hsplit(LC_umap, 2)
        # Add the new UMAP dimensions to a new DataFrame or append to existing df_test
        inspected_LC['umapDNN_Dim_1'], inspected_LC['umapDNN_Dim_2'] = new_dim1, new_dim2
        
        self.dataTP_umap_LC = ColumnDataSource(inspected_LC.loc[inspected_LC.true_positive == 1][0:self.nbre_pts])
        self.dataTN_umap_LC = ColumnDataSource(inspected_LC.loc[inspected_LC.true_negative == 1][0:self.nbre_pts])
        self.dataFP_umap_LC = ColumnDataSource(inspected_LC.loc[inspected_LC.false_positive == 1][0:self.nbre_pts])
        self.dataFN_umap_LC = ColumnDataSource(inspected_LC.loc[inspected_LC.false_negative == 1][0:self.nbre_pts])

        # Plot UMAP with true labels
    def Umap_true_label(self, plot = True, SNR_cut = None, inspected_LC = None):
        if SNR_cut is not None:
            self.dataTP_umap_CNN = self.filter_ColumnDataSource(self.dataTP_umap_CNN, SNR_cut)
            self.dataFP_umap_CNN = self.filter_ColumnDataSource(self.dataFP_umap_CNN, SNR_cut)
            self.dataTN_umap_CNN = self.filter_ColumnDataSource(self.dataTN_umap_CNN, SNR_cut)
            self.dataFN_umap_CNN = self.filter_ColumnDataSource(self.dataFN_umap_CNN, SNR_cut)
        u8 = figure(title='Results of umap after CNN', width=self.width, height=self.height, tools=(['tap', 'lasso_select', 'pan', 'wheel_zoom', 'reset']))

        title=f"Distribution of true label, after the CNN"

        u8.add_layout(Title(text=title, text_font_style="normal"), "above")

        u8.add_tools(HoverTool(tooltips=f"""
        <div>
            <div>
                <img 
                    src='@image' height="60" width="60" style='float: left; margin: 5px 5px 5px 5px'
                    ></img>
            <div>
                <span style='font-size: 14px; color: #224499'>Pred class:</span>
                <span style='font-size: 14px'>@bogus_transient_pred_{self.model_name}</span><br>
                <span style='font-size: 14px; color: #224499'>Obj id:</span>
                <span style='font-size: 14px'>@diaObjectId</span><br>
                <span style='font-size: 14px; color: #224499'>Nbr src in LC:</span>
                <span style='font-size: 14px'>@nDiaSources</span><br>
            </div>
        </div>
        """))
        #color_mapper = linear_cmap(field_name=f'y_pred_{self.model_name}', palette=Category10[4], low=0, high=1)

        u8.x('umapDNN_Dim_1', 'umapDNN_Dim_2', color='#FFC300',  source=self.dataTP_umap_CNN, line_alpha=0.6, fill_alpha=0.6, size=8, legend_label = "True Positive")
        u8.star('umapDNN_Dim_1', 'umapDNN_Dim_2', color='#FF5733',  source=self.dataFP_umap_CNN, line_alpha=0.6, fill_alpha=0.6, size=8, legend_label = "False Positive")
        u8.circle('umapDNN_Dim_1', 'umapDNN_Dim_2', color='#C70039',  source=self.dataTN_umap_CNN, line_alpha=0.6, fill_alpha=0.6, size=8, legend_label = "True Negative")
        u8.square('umapDNN_Dim_1', 'umapDNN_Dim_2', color='#b3b6b7',  source=self.dataFN_umap_CNN, line_alpha=0.6, fill_alpha=0.6, size=8, legend_label = "False Negative")
        
        if inspected_LC is not None:
            self.create_umap_data(inspected_LC)
            u8.x('umapDNN_Dim_1', 'umapDNN_Dim_2', color='#FFC300',  source=self.dataTP_umap_LC, line_alpha=0.6, fill_alpha=1, size=18, legend_label = "LC True Positive")
            u8.star('umapDNN_Dim_1', 'umapDNN_Dim_2', color='#FF5733',  source=self.dataFP_umap_LC, line_alpha=0.6, fill_alpha=1, size=18, legend_label = "LC False Positive")
            u8.circle('umapDNN_Dim_1', 'umapDNN_Dim_2', color='#C70039',  source=self.dataTN_umap_LC, line_alpha=0.6, fill_alpha=1, size=18, legend_label = "LC True Negative")
            u8.square('umapDNN_Dim_1', 'umapDNN_Dim_2', color='#b3b6b7',  source=self.dataFN_umap_LC, line_alpha=0.6, fill_alpha=1, size=18, legend_label = "LC False Negative")
        
        u8.add_layout(u8.legend[0])
        u8.legend.title = 'True label'
        u8.legend.click_policy="hide"

        # Create a Div widget for displaying selected fruit names
        div = Div(width=400, height=u8.height, height_policy="fixed", text="Select points on the plot")
        src = self.dataFP_umap_CNN
        # CustomJS callback to update the Div with selected fruit names
        cb = CustomJS(args=dict(src=src, div=div), code='''
            var sel_inds = src.selected.indices;
            var objId = src.data['diaObjectId'];
            var text = "Selected obj id :<br>";

            for (var i = 0; i < sel_inds.length; i++) {
                var sel_i = sel_inds[i];
                text += objId[sel_i] + "<br>";
            }

            div.text = text;
        ''')

        # Attach the callback to the selected indices change event
        src.selected.js_on_change('indices', cb)
        layout = row(u8, div)

        if plot: 
            output_notebook()  # Use this line if running in a Jupyter notebook
            show(layout, notebook_handle=True)
        
        if SNR_cut is not None:
            self.dataTP_umap_CNN = ColumnDataSource(self.df_test.loc[self.df_test.true_positive == 1][0:self.nbre_pts])
            self.dataTN_umap_CNN = ColumnDataSource(self.df_test.loc[self.df_test.true_negative == 1][0:self.nbre_pts])
            self.dataFP_umap_CNN = ColumnDataSource(self.df_test.loc[self.df_test.false_positive == 1][0:self.nbre_pts])
            self.dataFN_umap_CNN = ColumnDataSource(self.df_test.loc[self.df_test.false_negative == 1][0:self.nbre_pts])
        
        return u8
    
    
    def Umap_SNR(self, plot=True, snr_range=(-20, 20), SNR_cut = None, inspected_LC = None):
        if SNR_cut is not None:
            self.dataTP_umap_CNN = self.filter_ColumnDataSource(self.dataTP_umap_CNN, SNR_cut)
            self.dataFP_umap_CNN = self.filter_ColumnDataSource(self.dataFP_umap_CNN, SNR_cut)
            self.dataTN_umap_CNN = self.filter_ColumnDataSource(self.dataTN_umap_CNN, SNR_cut)
            self.dataFN_umap_CNN = self.filter_ColumnDataSource(self.dataFN_umap_CNN, SNR_cut)
            
        # Main UMAP plot
        u10 = figure(title='Results of UMAP after CNN', width=self.width, height=self.height, tools=('pan, wheel_zoom, reset'))

        title = f"Distribution of SNR, after the CNN, {self.model_name}"
        u10.add_layout(Title(text=title, text_font_style="normal"), "above")
        self.df_test['clipped_SNR'] = self.df_test['snr'].clip(*snr_range)

        u10.add_tools(HoverTool(tooltips=f"""
        <div>
            <div>
                <img src='@image' height="60" width="60" style='float: left; margin: 5px 5px 5px 5px'></img>
            <div>
                <span style='font-size: 14px; color: #224499'>Predicted class:</span>
                <span style='font-size: 14px'>@bogus_transient_pred_{self.model_name}</span><br>
                <span style='font-size: 14px; color: #224499'>SNR:</span>
                <span style='font-size: 14px'>@snr</span><br>
            </div>
        </div>
        """))

        # Create a linear color mapper based on SNR values
        color_mapper = linear_cmap(field_name='snr', palette=cc.kbc, low=self.df_test['clipped_SNR'].min(), high=self.df_test['clipped_SNR'].max())

        u10.x('umapDNN_Dim_1', 'umapDNN_Dim_2', color=color_mapper,  source=self.dataTP_umap_CNN, line_alpha=0.6, fill_alpha=0.6, size=8, legend_label = "True Positive")
        u10.star('umapDNN_Dim_1', 'umapDNN_Dim_2', color=color_mapper,  source=self.dataFP_umap_CNN, line_alpha=0.6, fill_alpha=0.6, size=8, legend_label = "False Positive")
        u10.circle('umapDNN_Dim_1', 'umapDNN_Dim_2', color=color_mapper,  source=self.dataTN_umap_CNN, line_alpha=0.6, fill_alpha=0.6, size=8, legend_label = "True Negative")
        u10.square('umapDNN_Dim_1', 'umapDNN_Dim_2', color=color_mapper,  source=self.dataFN_umap_CNN, line_alpha=0.6, fill_alpha=0.6, size=8, legend_label = "False Negative")
        
        if inspected_LC is not None:
            self.create_umap_data(inspected_LC)
            u10.x('umapDNN_Dim_1', 'umapDNN_Dim_2', color=color_mapper,  source=self.dataTP_umap_LC, line_alpha=0.6, fill_alpha=1, size=18, legend_label = "LC True Positive")
            u10.star('umapDNN_Dim_1', 'umapDNN_Dim_2', color=color_mapper,  source=self.dataFP_umap_LC, line_alpha=0.6, fill_alpha=1, size=18, legend_label = "LC False Positive")
            u10.circle('umapDNN_Dim_1', 'umapDNN_Dim_2', color=color_mapper,  source=self.dataTN_umap_LC, line_alpha=0.6, fill_alpha=1, size=18, legend_label = "LC True Negative")
            u10.square('umapDNN_Dim_1', 'umapDNN_Dim_2', color=color_mapper,  source=self.dataFN_umap_LC, line_alpha=0.6, fill_alpha=1, size=18, legend_label = "LC False Negative")
        

        # Add a color bar to represent the SNR values
        color_bar = ColorBar(color_mapper=color_mapper['transform'], width=8, location=(0, 0))
        u10.add_layout(color_bar, 'right')

        u10.add_layout(u10.legend[0])
        u10.legend.title = 'True label'
        u10.legend.click_policy = "hide"

        # Handle NaN values by dropping them
        valid_snr = self.df_test['snr'].dropna()

        # Create a vertical histogram of the SNR values
        hist, edges = np.histogram(valid_snr, bins=50, range = (self.df_test['clipped_SNR'].min(), self.df_test['clipped_SNR'].max()))

        bin_midpoints = (edges[:-1] + edges[1:]) / 2

        # Create a new ColumnDataSource for the histogram, including bin midpoints
        hist_source = ColumnDataSource(data=dict(
            top=hist,
            bottom=np.zeros_like(hist),
            left=edges[:-1],
            right=edges[1:],
            snr_mid=bin_midpoints  # Midpoints of the bins for color mapping
        ))

        # Apply color mapping based on the midpoints of the SNR bins
        snr_mapper = linear_cmap(field_name='snr_mid', palette=cc.kbc, low=self.df_test['clipped_SNR'].min(), high=self.df_test['clipped_SNR'].max())

        snr_plot = figure(width=200, height=self.height, title="SNR Distribution", tools="")

        # Use the color mapper for the histogram
        snr_plot.quad(top='top', bottom='bottom', left='left', right='right', source=hist_source,
                      fill_color=snr_mapper, line_color="white")

        # Combine the plots
        layout = gridplot([[u10, snr_plot]], toolbar_location="right", merge_tools=True)

        if plot:
            output_notebook()
            show(layout, notebook_handle = True)
        if SNR_cut is not None:
            self.dataTP_umap_CNN = ColumnDataSource(self.df_test.loc[self.df_test.true_positive == 1][0:self.nbre_pts])
            self.dataTN_umap_CNN = ColumnDataSource(self.df_test.loc[self.df_test.true_negative == 1][0:self.nbre_pts])
            self.dataFP_umap_CNN = ColumnDataSource(self.df_test.loc[self.df_test.false_positive == 1][0:self.nbre_pts])
            self.dataFN_umap_CNN = ColumnDataSource(self.df_test.loc[self.df_test.false_negative == 1][0:self.nbre_pts])
        
        return layout
    def Umap_prob_injection(self, plot = True, SNR_cut = None):
        if SNR_cut is not None:
            self.data_glob_umap = self.filter_ColumnDataSource(self.data_glob_umap, SNR_cut)
            
        column_name = f'class_proba_bogus_{self.model_name}'
        # Retrieve sorted unique values
        GROUP = sorted(self.df_test[f'class_proba_bogus_{self.model_name}'].unique())

        u3 = figure(title='Distribution of probability', width=self.width, height=self.height, tools=('pan, wheel_zoom, reset'))

        title=f"Probability distribution predicted by the model that the object labeled as bogus is a transient, model {self.model_name}"

        u3.add_layout(Title(text=title, text_font_style="normal"), "above")

        u3.add_tools(HoverTool(tooltips=f"""
        <div>
            <div>
                <img 
                    src='@image' height="60" width="60" style='float: left; margin: 5px 5px 5px 5px'
                    ></img>
            <div>
                <span style='font-size: 14px; color: #224499'>Real class:</span>
                <span style='font-size: 14px'>@y_test</span><br>
                <span style='font-size: 14px; color: #224499'>model pred transient:</span>
                <span style='font-size: 14px'>@proba_transient_perc_{self.model_name} %</span>
            </div>
        </div>
        """))

        u3.circle('umapDNN_Dim_1', 'umapDNN_Dim_2', source=self.data_glob_umap, color=factor_cmap(f'class_proba_bogus_{self.model_name}', palette=Spectral[11], factors=GROUP), line_alpha=0.6, 
                   size=8, legend_group=f'class_proba_bogus_{self.model_name}')

        u3.add_layout(u3.legend[0], 'right')
        u3.legend.title = 'output proba:'
        u3.legend.click_policy="hide"

        if plot:
            output_notebook()  # Use this line if running in a Jupyter notebook
            show(u3, notebook_handle=True)
        if SNR_cut is not None:        
            self.data_glob_umap = ColumnDataSource(self.df_test[0:self.nbre_pts])
        return u3

    def umap_visual_classes(self, plot = True, SNR_cut = None):
        
        pred0 = self.tune_pred_gmm()
        self.df_test['label_clustering_umap'] = pred0.astype(str)

        num_dipole = self.df_test.label_clustering_umap.loc['Dipole']
        num_CR = self.df_test.label_clustering_umap.loc['CR']
        num_Negative = self.df_test.label_clustering_umap.loc['Negative']
        num_Interpolation = self.df_test.label_clustering_umap.loc['Interpolation']
        num_Simutransient = self.df_test.label_clustering_umap.loc['Simulated transient']
        num_Transient = self.df_test.label_clustering_umap.loc['Transient']
        num_LowSigtonoise = self.df_test.label_clustering_umap.loc['Low signal to noise']

        #classe visuel umap
        self.df_test['classvisuel_umap'] = self.df_test.label_clustering_umap.mask((self.df_test.label_clustering_umap==num_dipole), other='Dipole')
        self.df_test['classvisuel_umap'] = self.df_test.classvisuel_umap.mask((self.df_test.label_clustering_umap==num_CR), other='Cosmic ray')
        self.df_test['classvisuel_umap'] = self.df_test.classvisuel_umap.mask((self.df_test.label_clustering_umap==num_Negative), other='Negative transient')
        self.df_test['classvisuel_umap'] = self.df_test.classvisuel_umap.mask((self.df_test.label_clustering_umap==num_Interpolation), other='Interpolation')
        self.df_test['classvisuel_umap'] = self.df_test.classvisuel_umap.mask((self.df_test.label_clustering_umap==num_Simutransient), other='Simulated transient')
        self.df_test['classvisuel_umap'] = self.df_test.classvisuel_umap.mask((self.df_test.label_clustering_umap==num_Transient), other='Transient')
        self.df_test['classvisuel_umap'] = self.df_test.classvisuel_umap.mask((self.df_test.label_clustering_umap==num_LowSigtonoise), other='Low signal to noise')
        
        data_glob_umap = ColumnDataSource(self.df_test)
        if SNR_cut is not None:
            data_glob_umap = self.filter_ColumnDataSource(data_glob_umap, SNR_cut)

        GROUP = sorted(self.df_test.classvisuel_umap.unique())

        u9 = figure(title=f'Visual class obtained by clustering, {self.model_name}', width=self.width, height=self.height, tools=('pan, wheel_zoom, reset'))

        u9.add_tools(HoverTool(tooltips=f"""
        <div>
            <div>
                <img 
                    src='@image' height="60" width="60" style='float: left; margin: 5px 5px 5px 5px'
                    ></img>
            <div>
                <span style='font-size: 14px; color: #224499'>Real class:</span>
                <span style='font-size: 14px'>@y_test</span><br>
                <span style='font-size: 14px; color: #224499'>Proba transient:</span>
                <span style='font-size: 14px'>@proba_transient_perc_{self.model_name} %</span>
            </div>
        </div>
        """))

        u9.circle('umapDNN_Dim_1', 'umapDNN_Dim_2', source=data_glob_umap, color=factor_cmap('classvisuel_umap', palette=Category10[10], factors=GROUP), line_alpha=0.6, 
                  fill_alpha=0.6, size=8, legend_group='classvisuel_umap')

        u9.add_layout(u9.legend[0], 'right')
        if plot: 
            output_notebook()  # Use this line if running in a Jupyter notebook
            show(u9, notebook_handle=True)

        return u9

    def filter_ColumnDataSource(self, CDS, SNR_cut):
        try:
            # Convert ColumnDataSource to DataFrame
            data = CDS.data
            df_from_cds = pd.DataFrame(data)

            # Drop unnecessary columns
            columns_to_drop = ['level_0', 'index']  # Add any other columns that are not needed
            columns_to_drop = [col for col in columns_to_drop if col in df_from_cds.columns]

            if columns_to_drop:
                df_from_cds = df_from_cds.drop(columns=columns_to_drop)

            # Ensure the SNR column exists and is numeric
            snr_column = 'SNR_base_CircularApertureFlux_6_0'
            if snr_column not in df_from_cds.columns:
                raise KeyError(f"Column '{snr_column}' not found in DataFrame")

            df_from_cds[snr_column] = pd.to_numeric(df_from_cds[snr_column], errors='coerce')

            # Filter the DataFrame
            filtered_df = df_from_cds[
                (df_from_cds[snr_column] >= SNR_cut) |
                (df_from_cds[snr_column] <= -SNR_cut)
            ]

            # Reset index and avoid adding a new column
            filtered_df = filtered_df.reset_index(drop=True)

            # Create a new ColumnDataSource
            new_ColumnDataSource = ColumnDataSource(filtered_df)

            return new_ColumnDataSource
        
        except Exception as e:
            print(f"An error occurred: {e}")
            return None
        
    # Tools for the plots
    def gmm_bic_score(self, estimator, X):
        """Callable to pass to GridSearchCV that will use the BIC score."""
        # Make it negative since GridSearchCV expects a score to maximize
        return -estimator.bic(X)
    
    def tune_pred_gmm(self):
                
        param_grid = {
            "n_components": range(7, 10),
            "covariance_type": ["spherical", "tied", "diag", "full"],
        }
        grid_search = GridSearchCV(
            GaussianMixture(), param_grid=param_grid, scoring=self.gmm_bic_score
        )
        grid_search.fit(self.umapDNN_data)

        df_BIC = pd.DataFrame(grid_search.cv_results_)[
            ["param_n_components", "mean_test_score"] #,"param_covariance_type"]
        ]

        df_BIC["mean_test_score"] = -df_BIC["mean_test_score"]
        df_BIC = df_BIC.rename(
            columns={
                "param_n_components": "Number of components",
                "param_covariance_type": "Type of covariance",
                "mean_test_score": "BIC score",
            }
        )

        # Create gaussian Mixture object with optimized parameter
        best_gmm = GaussianMixture(n_components=grid_search.best_params_["n_components"],
                                   covariance_type=grid_search.best_params_["covariance_type"]
                                  )

        # adjust on data model
        best_gmm.fit(self.umapDNN_data)

        # retrieve tags
        pred0 = best_gmm.predict(self.umapDNN_data)
        
        return pred0