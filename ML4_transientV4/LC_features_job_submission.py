
from astropy import units as u
from astropy.time import Time

import light_curve as lc
from utils import datadir, rundir, pddir, embeddable_image

import warnings; warnings.simplefilter('ignore')
import pickle
import numpy as np
import pandas as pd

with open("saved/full_dia_source_table_UDEEP.pkl", "rb") as pickle_file:
    full_dia_source_table_UDEEP = pickle.load(pickle_file)
    
with open("saved/pred_dia_obj_table.pkl", "rb") as pickle_file:
    clean_dia_obj_table = pickle.load(pickle_file) 

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

def compute_LC_features(df_LC, LC):
    df_LC = df_LC.sort_values(['midpointMjdTai'],ascending=True)
    t = df_LC['midpointMjdTai'].values
    err = df_LC['cal_mag_err'].values
    bands = df_LC['band'].values
    m = df_LC['cal_mag'].values
    f = df_LC['apFlux'].values
    amplitude = lc.Amplitude()
    adn = lc.AndersonDarlingNormal()
    bazin_fit = lc.BazinFit('mcmc')
    beyond_n_std = lc.BeyondNStd()
    cusum = lc.Cusum()
    duration = lc.Duration()
    eta = lc.Eta()
    eta_e = lc.EtaE()
    excess_variance = lc.ExcessVariance()
    flux_n_not_det_before_fd = lc.FluxNNotDetBeforeFd()
    inter_percentile_range = lc.InterPercentileRange()
    kurtosis = lc.Kurtosis()
    linear_fit = lc.LinearFit()
    linear_trend = lc.LinearTrend()
    linexp_fit = lc.LinexpFit('mcmc')
    magnitude_percentage_ratio = lc.MagnitudePercentageRatio()
    maximum_slope = lc.MaximumSlope()
    maximum_time_interval = lc.MaximumTimeInterval()
    mean = lc.Mean()
    mean_variance = lc.MeanVariance()
    median = lc.Median()
    median_absolute_deviation = lc.MedianAbsoluteDeviation()
    median_buffer_range_percentage = lc.MedianBufferRangePercentage()
    minimum_time_interval = lc.MinimumTimeInterval()
    observation_count = lc.ObservationCount()
    otsu_split = lc.OtsuSplit()
    percent_amplitude = lc.PercentAmplitude()
    percent_difference_magnitude_percentile = lc.PercentDifferenceMagnitudePercentile()
    periodogram = lc.Periodogram()
    reduced_chi2 = lc.ReducedChi2()
    roms = lc.Roms()
    skew = lc.Skew()
    standard_deviation = lc.StandardDeviation()
    stetson_k = lc.StetsonK()
    time_mean = lc.TimeMean()
    time_standard_deviation = lc.TimeStandardDeviation()
    villar_fit = lc.VillarFit('mcmc')
    weighted_mean = lc.WeightedMean()

    extractor_mag = lc.Extractor(
        amplitude,
        adn,
        beyond_n_std,
        cusum,
        duration,
        eta,
        eta_e,
        inter_percentile_range,
        linear_fit,
        linear_trend,
        linexp_fit,
        magnitude_percentage_ratio,
        maximum_slope,
        maximum_time_interval,
        mean,
        median,
        median_absolute_deviation,
        median_buffer_range_percentage,
        minimum_time_interval,
        observation_count,
        otsu_split,
        percent_amplitude,
        periodogram,
        reduced_chi2,
        roms,
        skew,
        standard_deviation,
        stetson_k,
        time_mean,
        time_standard_deviation,
        weighted_mean
    )
    print(df_LC['diaObjectId'].values[0])
    result_mag = extractor_mag(t, m, err, sorted=True, check=False)
    mag_feature_names = extractor_mag.names 
    for name, value in zip(mag_feature_names, result_mag):
        LC[f"{name}"] = value
    
    extractor_flux = lc.Extractor(
        bazin_fit, 
        mean_variance, 
        percent_difference_magnitude_percentile,
        villar_fit
    )
    result_flux = extractor_flux(f, m, err, sorted=True, check=False)
    
    flux_feature_names = extractor_flux.names
    for name, value in zip(flux_feature_names, result_flux):
        LC[f"{name}"] = value

    return LC

LC_features_z = pd.DataFrame()
for diaObjetId in clean_dia_obj_table.index:  # Iterate over the indices
    row = clean_dia_obj_table.loc[[diaObjetId]]  # Retrieve the entire row as a DataFrame
    df_LC = retrieve_LC(str(diaObjetId), 'z')
    value_counts_m = df_LC['cal_mag'].value_counts()

    if len(df_LC) > 8 and not (df_LC['apFlux'] < 0).all():
        try:
            LC_features = compute_LC_features(df_LC, row)  # Now row is a DataFrame
            # Concatenate with LC_features_z and reset index for stacking rows
            LC_features_z = pd.concat([LC_features_z, LC_features])  # Reset index
        except ValueError as e:
            print(f"Skipping due to error: {e}, at object {diaObjetId}")


lc_features_list = []
for diaObjetId, row in clean_dia_obj_table.iterrows():
    df_LC = retrieve_LC(str(diaObjetId), 'z')
    value_counts_m = df_LC['cal_mag'].value_counts()
    if len(df_LC) > 8 and not (df_LC['apFlux'] < 0).all():
        try:
            LC_features = compute_LC_features(df_LC, row)
            lc_features_list.append(LC_features)
        except ValueError as e:
            print(f"Skipping due to error: {e}, at object {diaObjetId}")
LC_features_z = pd.concat(lc_features_list, ignore_index=True)

with open(f"saved/full_dia_obj_table_z.pkl", "wb") as pickle_file:
    pickle.dump(LC_features_z, pickle_file)