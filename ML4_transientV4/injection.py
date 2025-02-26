import matplotlib.pyplot as plt

import numpy as np
import pandas as pd


import astropy.units as u

import lsst.afw.display as afwDisplay
import lsst.geom
from astropy.table import Table

from lsst.geom import Point2D, Box2I, Point2I, Extent2I
import random
from scipy.stats import norm 
from astropy.table import Table

from lsst.source.injection import VisitInjectConfig,VisitInjectTask



import pickle
import scipy.stats as stats

import warnings


warnings.filterwarnings("ignore", category=RuntimeWarning)  # Ignore runtime warnings
warnings.filterwarnings("ignore", category=FutureWarning)


def sample_from_distribution(band, num_samples=1):
    with open(f'saved/mag_distrib.pkl', "rb") as pickle_file:
        mag_distrib = pickle.load(pickle_file)
    if band in mag_distrib:
        best_distribution = mag_distrib[band]
        dist_name, params = list(best_distribution.items())[0]  # Get the best distribution name and parameters
        dist = getattr(stats, dist_name)  # Get the distribution object from scipy.stats
        
        # Sample from the distribution
        samples = dist.rvs(*params.values(), size=num_samples)
        return samples
    else:
        print(f"Band {band} not found in best fits.")
        return None
    

def calculate_orientation_and_extent(Ixx, Ixy, Iyy):
    '''
    Calculates the orientation and extent of an object based on its second moments.

    Parameters:
    Ixx (float): Second moment of the object along the x-axis. Often in degree²
    Ixy (float): Second moment of the object along the x and y-axes. Often in degree²
    Iyy (float): Second moment of the object along the y-axis. Often in degree²
        
    Returns: 
    tuple: A tuple containing:
        - theta (float): The orientation angle of the object in radians.
        - a (float): The semi-major axis length of the object.
        - b (float): The semi-minor axis length of the object.
    '''
    # Calculate position angle (orientation)
    theta = 0.5 * np.arctan2(2 * Ixy, Ixx - Iyy)
    
    # Calculate eigenvalues of the moment matrix
    term1 = (Ixx + Iyy) / 2
    term2 = np.sqrt(((Ixx - Iyy) / 2) ** 2 + Ixy ** 2)
    lambda1 = term1 + term2
    lambda2 = term1 - term2

    a = np.sqrt(lambda1) 
    b = np.sqrt(lambda2)
    
    return theta, a, b




def generate_fakes_pos_mag(single_detec, band): 
    '''
    Generates fake positions and magnitudes of a fake supernova given galaxy second moment.

    Parameters:
    single_detec (dict): A dictionary containing the detection parameters.
                         Expected keys are 'ixx', 'ixy', 'iyy', and a flux key (default is 'calibFlux').
    band (string): band of in which the source will be injected

    Returns:
    tuple: A tuple containing:
        - r (float): The randomly generated radial distance from the center based on the semi-major axis length.
        - theta (float): The randomly generated orientation angle in degrees.
        - mag (float): The randomly generated magnitude in AB magnitudes.
        - orientation (float): The orientation angle of the object in radians.
    '''
    orientation, a, b = calculate_orientation_and_extent(single_detec['ixx'], single_detec['ixy'], single_detec['iyy'])
    
    # sample r ~ N(0,a) and sample theta randomly
    r = np.random.normal(0, a, size=1)
    theta = random.uniform(0, 180)
    
    # compute magnitude
    flux_type = 'calibFlux'   # can be change 
    
    flux = (np.array(single_detec[flux_type]))
    mag = (flux * u.nJy).to(u.ABmag)

    # dealing with infinite magnitude
    if np.isfinite(mag.value): 
        mag  = random.uniform(mag.value -1, mag.value + 3)
    else: 
        mag = sample_from_distribution(band)[0]
    
    return r, theta, mag, orientation


def plot_fakes_pos_tohost(catalog_of_galaxies, band): 
    '''
    Plots the positions of fake sources relative to their own host galaxies' centers.

    Parameters:
    catalog_of_galaxies (DataFrame): DataFrame containing the catalog of galaxies with their properties.
    band (string): band of in which the source will be injected

    Returns:
    None'''
    
    rads = []
    thetas = []
    thetas4plot = []
    mags = []
    # compute and store r and theta
    for index, detec in catalog_of_galaxies.iterrows():
        r, theta, mag, orientation = generate_fakes_pos_mag(detec, band)

        rads.append(r)
        
        # theta is sampled from 0 to 180 since r can be negative, fixing this here
        if r<0: 
            thetas4plot.append(theta+180)

        else : 
            thetas4plot.append(theta)
        mags.append(mag)

    # plotting 
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    sc = ax.scatter(thetas4plot, np.abs(rads), c=np.array(mags), cmap='YlOrRd', alpha=0.75)
    ax.set_rlabel_position(-22.5)  # Move radial labels away from plotted line
    ax.grid(True)
    cbar = fig.colorbar(sc, ax=ax, label='Magnitude')
    ax.set_title("Fakes positions relative to their own galaxies centers", va='bottom')
    ax.set_xlabel('radius in pixels')
    plt.show()
    
    
def random_hex_color():
    '''
    Generates a random hexadecimal color code.

    Returns:
    str: A string representing a random hexadecimal color code in the format #RRGGBB.
    '''
    hex_chars = '0123456789ABCDEF'
    color_code = '#' + ''.join(random.choice(hex_chars) for _ in range(6))
    return color_code



def pos_mag_4catalog(catalog_of_galaxies, image, band, plot = False, additional_host_data = False): 
    '''
    Generates positions and magnitudes for a catalog of galaxies, and optionally plots them on the image. 
    The plot is composed of circles, centered on the host galaxies, with crosses centered the injected object, host and injection have matching colors. The stars represents the hostless injection. They represents around 5% of the injection.

    Parameters:
    catalog_of_galaxies (DataFrame): DataFrame containing the catalog of galaxies with their properties (source table).
    image (Image): The image object containing the WCS information, calexp for instance.
    band (string): band of in which the source will be injected
    plot (bool): If True, plot the galaxies on the image. Default is False.
    additional_host_data (bool): If True, include additional host data in the output. Default is False.

    Returns:
    tuple: A tuple containing:
        - inject_ra (list): List of RA coordinates for the injected fake sources.
        - inject_dec (list): List of Dec coordinates for the injected fake sources.
        - mags (list): List of magnitudes for the injected fake sources.
        - host_data (DataFrame, optional): DataFrame containing additional host data if requested.
    '''
    
    
    inject_ra = []
    inject_dec = []
    new_positions = []
    new_rand_positions = []
    mags = []
    host_data = pd.DataFrame()
    flux_type = 'calibFlux'   # can be change 

    # Retrieve sky origin and pixel origin from wcs
    wcs = image.getWcs()

    sky_origin = wcs.pixelToSky(Point2D(0, 0))  # Convert origin pixel to sky coordinates

    sky_origin_ra = sky_origin.getRa().asDegrees()
    sky_origin_dec = sky_origin.getDec().asDegrees()

    pixel_origin_x = wcs.getPixelOrigin().getX()
    pixel_origin_y = wcs.getPixelOrigin().getY()

    pixel_scale = wcs.getPixelScale().asArcseconds()

    
    for index, detec in catalog_of_galaxies.iterrows():
        r, theta, mag, orientation = generate_fakes_pos_mag(detec, band)
        mags.append(mag)
        
    
        # Convert radial positions to RaDec
        ra = detec['ra']
        dec = detec['dec']
    
        ## Convert polar coordinates (r, theta) to Cartesian coordinates (x, y)
        x = r * np.cos(np.radians(theta))
        y = r * np.sin(np.radians(theta))
        
        ## Retrieve the right position removing the galaxy orientation
        x_rot = x * np.cos(orientation) - y * np.sin(orientation)
        y_rot = x * np.sin(orientation) + y * np.cos(orientation)
        
        ## compute the difference between the host and the injection
        delta_ra = (x_rot/np.cos(np.radians(dec)))*0.168432/3600 # need to convert from pixel/arcesc to degrees
        delta_dec = y_rot*0.168432/3600
        
        ## retrieving the global ra dec position of the injection
        ra_i = ra - delta_ra 
        dec_i = dec - delta_dec
        
        # store the ra dec in list 
        inject_ra.append(float(ra_i))
        inject_dec.append(float(dec_i))
        new_positions.append((ra_i, dec_i, ra, dec))
        
        # optionnally store host and visit information for each injection
        if additional_host_data == True:    
            flux = (np.array(detec[flux_type]))
            host_mag = (flux * u.nJy).to(u.ABmag)
            single_host_data = pd.DataFrame({"ra" : ra_i, 
                                            "dec" : dec_i, 
                                            "mag" : mag, 
                                            "host_magnitude" : host_mag, 
                                            "visit" : detec['visit'], 
                                            "detector" : detec['detector'], 
                                            "parent_index" : index, 
                                            "band" : detec['band']
                                            })
            
            host_data = pd.concat([host_data, single_host_data], ignore_index=True)
        
    
    # generate hostless injection
    nbr_hostless = len(catalog_of_galaxies)//20    # ~5% of the injection
    mag_hostless = [sample_from_distribution(band)[0] for i in range(nbr_hostless)]
    mags.extend(mag_hostless)
    # pick a random position on the ccd
    x = np.random.uniform(0, 2100 , size = nbr_hostless)
    y = np.random.uniform(0, 4200, size = nbr_hostless)
    
    # convert those in ra dec
    ra, dec = image.wcs.pixelToSkyArray(x,y, degrees = True)

    inject_ra.extend(list(ra))
    inject_dec.extend(list(dec))
    new_rand_positions = [ (r, d, xx, yy)  for r, d, xx, yy in zip(ra, dec, x, y)]
    
                                                
    # add optionnal data to the hostless injection (None for host infos)

    if additional_host_data == True:    
        single_host_data = pd.DataFrame({"ra" : list(ra), 
                                            "dec" : list(dec), 
                                            "mag" : mag_hostless, 
                                            "host_magnitude": [None] * len(ra),  # Defaulting to None
                                            "visit" : detec['visit'], 
                                            "detector" : detec['detector'], 
                                            "parent_index": [None] * len(ra),  # Defaulting to None
                                            "band" : detec['band']
                                            })        
        
        host_data = pd.concat([host_data, single_host_data], ignore_index=True)



    # Plot ! 
    if plot == True : 

        # Set up the display
        afwDisplay.setDefaultBackend('matplotlib')
        fig = plt.figure(figsize=(10, 8))
        afw_display = afwDisplay.Display(1)
        afw_display.scale('asinh', 'zscale')
        afw_display.mtv(image)

        print(wcs)
        # Plot elliptical apertures on each galaxy
        with afw_display.Buffering():
            for index, detec in catalog_of_galaxies.iterrows():
                if detec['extendedness'] == 1:
                    # Extract the necessary columns
                    ixx = detec['ixx']
                    iyy = detec['iyy']
                    ixy = detec['ixy']
                    position = Point2D(detec['x'], detec['y'])  
                    theta, a, b = calculate_orientation_and_extent(ixx, ixy, iyy)

                    # Create and plot the elliptical aperture
                    ellipse = Ellipse((position.getX(), position.getY()), width=2*a, height=2*b, angle=np.degrees(theta), edgecolor='#d62728', facecolor='none', linewidth=2)
                    plt.gca().add_patch(ellipse)

        with afw_display.Buffering():
            for index, (ra_i, dec_i, host_ra, host_dec) in enumerate(new_positions):
                # injection center
                color = random_hex_color()
                pixel_pos = wcs.skyToPixel(lsst.geom.SpherePoint(ra_i, dec_i, lsst.geom.degrees))
                afw_display.dot('x', pixel_pos.getX(), pixel_pos.getY(), size=100, ctype=color)
                
                # host center 
                center_pos = wcs.skyToPixel(lsst.geom.SpherePoint(host_ra, host_dec, lsst.geom.degrees))
                afw_display.dot('o', center_pos.getX(), center_pos.getY(), size=100, ctype=color)

        with afw_display.Buffering():
            # hostless
            for ra_i, dec_i, x, y  in new_rand_positions:
                color = random_hex_color()
                afw_display.dot('*', x, y, size=100, ctype=color)
        plt.show()  
        
    if additional_host_data == True : 
        return inject_ra, inject_dec, mags, host_data

    return inject_ra, inject_dec, mags


def zoomed_pos_mag_4catalog(catalog_of_galaxies, image,  x_center, y_center, half_size, band): 
    '''
    Generates positions and magnitudes for a catalog of galaxies within a zoomed-in region of an image,
    and plots them on the zoomed-in image.
    Warning, even though the catalog is given in the input, the actual position of the injections (host and hostless) are sampled in the function, so they'll not match other plots.

    Parameters:
    catalog_of_galaxies (DataFrame): DataFrame containing the catalog of galaxies with their properties.
    image (Image): The image object containing the WCS information.
    x_center (int): X-coordinate of the center of the zoomed-in region.
    y_center (int): Y-coordinate of the center of the zoomed-in region.
    half_size (int): Half the size of the zoomed-in region (i.e., the full size will be 2 * half_size).
    band (string): band of in which the source will be injected


    Returns:
    None
    '''
    inject_ra = []
    inject_dec = []
    new_positions = []
    mags = []

    for index, detec in catalog_of_galaxies.iterrows():
        r, theta, mag, orientation = generate_fakes_pos_mag(detec, band)

        mags.append(mag)
        
         # Convert radial positions to RaDec
        ra = detec['ra']
        dec = detec['dec']
    
        ## Convert polar coordinates (r, theta) to Cartesian coordinates (x, y)
        x = r * np.cos(np.radians(theta))

        y = r * np.sin(np.radians(theta))

        x_rot = x * np.cos(orientation) - y * np.sin(orientation)

        y_rot = x * np.sin(orientation) + y * np.cos(orientation)

        delta_ra = (x_rot/np.cos(np.radians(dec)))*0.168432/3600

        delta_dec = y_rot*0.168432/3600

        ra_i = ra - delta_ra

        dec_i = dec - delta_dec

        inject_ra.append(ra_i)
        inject_dec.append(dec_i)
        new_positions.append((ra_i, dec_i, ra, dec))

    bbox = Box2I(Point2I(x_center-half_size, y_center-half_size),
             Extent2I(2 * half_size, 2 * half_size))
    subimage = image.Factory(image, bbox)
    
    
    nbr_hostless = len(catalog_of_galaxies)//20    
    mag_hostless = [random.uniform(17, 23) for i in range(nbr_hostless)]
    mags.extend(mag_hostless)
    x = np.random.uniform(0, 2100 , size = nbr_hostless)
    y = np.random.uniform(0, 4200, size = nbr_hostless)

    ra, dec = image.wcs.pixelToSkyArray(x,y, degrees = True)

    inject_ra.extend(list(ra))
    inject_dec.extend(list(dec))
    new_rand_positions = [ (r, d, xx, yy)  for r, d, xx, yy in zip(ra, dec, x, y)]

    
    # Set up the display
    afwDisplay.setDefaultBackend('matplotlib')
    fig = plt.figure(figsize=(10, 8))
    afw_display = afwDisplay.Display(1)
    afw_display.setMaskTransparency(100, 'DETECTED')
    afw_display.setMaskTransparency(100, 'STREAK')

    #afw_display.setMaskTransparency(100)
    afw_display.scale('asinh', 'zscale')
    afw_display.mtv(subimage)
    wcs = subimage.getWcs()
    print(wcs)
    # Plot elliptical apertures on each galaxy
    with afw_display.Buffering():
        for index, detec in catalog_of_galaxies.iterrows():
            if detec['extendedness'] == 1:
                # Extract the necessary columns
                ixx = detec['ixx']
                iyy = detec['iyy']
                ixy = detec['ixy']
                x = detec['x']
                y = detec['y']

                if (x_center - half_size < x < x_center + half_size) and (y_center - half_size < y < y_center + half_size):
                    position = Point2D(x, y)
                    theta, a, b = calculate_orientation_and_extent(ixx, ixy, iyy)

                    # Create and plot the elliptical aperture
                    ellipse = Ellipse((position.getX(), position.getY()),
                                      width= a, height=b, angle=np.degrees(theta),
                                      edgecolor='red', facecolor='none', linewidth=2)
                    plt.gca().add_patch(ellipse)

    with afw_display.Buffering():
        for index, (ra_i, dec_i, host_ra, host_dec) in enumerate(new_positions):
            color = random_hex_color()
            pixel_pos = wcs.skyToPixel(lsst.geom.SpherePoint(ra_i, dec_i, lsst.geom.degrees))
            if (x_center - half_size < pixel_pos.getX() < x_center + half_size) and (y_center - half_size < pixel_pos.getY() < y_center + half_size):

                afw_display.dot('x', pixel_pos.getX(), pixel_pos.getY(), size=10, ctype=color)

                # Draw line connecting cross to the center of the ellipse
                center_pos = wcs.skyToPixel(lsst.geom.SpherePoint(host_ra, host_dec, lsst.geom.degrees))
                afw_display.dot('o', center_pos.getX(), center_pos.getY(), size=10, ctype=color)
    
    
    with afw_display.Buffering():
        for ra_i, dec_i, x, y  in new_rand_positions:
            if (x_center - half_size < x < x_center + half_size) and (y_center - half_size < y < y_center + half_size):
                color = random_hex_color()
                afw_display.dot('*', x, y, size=10, ctype=color)

    plt.show()
    

def create_catalog(dict_data, host_data = False):
    '''
    Creates a catalog from dictionary data, optionally including host galaxy data. The short catalog fits the injection pipeline standard. 

    Parameters:
    dict_data (dict): Dictionary containing 'ra', 'dec', 'mag', and optionally 'host_magnitude', 'visit', 'detector', 'parent_index', 'band'.
    host_data (bool, optional): Flag indicating whether to include host galaxy data. Defaults to False.

    Returns:
    astropy.table.Table: Catalog table containing injection_id, ra, dec, source_type, mag, and optionally host_magnitude, visit, detector, parent_index, band.
    '''
    # reduce catalog, fitting injection pipeline standard
    if host_data is False:
        df = pd.DataFrame({ "injection_id" : np.arange(len(dict_data["ra"])), 
                           "ra" : dict_data["ra"], 
                           "dec" : dict_data["dec"],
                           "source_type" : "Star",
                           "mag" : dict_data["mag"]})
        fancy_catalog = Table.from_pandas(df)
        return fancy_catalog
    
    # reduce catalog, adding information on host and visit
    else : 
        df = pd.DataFrame({ "injection_id" : np.arange(len(dict_data["ra"])), 
                           "ra" : dict_data["ra"], 
                           "dec" : dict_data["dec"],
                           "source_type" : "Star",
                           "mag" : dict_data["mag"],
                          "host_magnitude" : dict_data["host_magnitude"],
                          "visit" : dict_data["visit"],
                          "detector" : dict_data["detector"],
                          "parent_index" : dict_data["parent_index"],
                          "band" : dict_data["band"],
                          })
        fancy_catalog = Table.from_pandas(df)
        return fancy_catalog
    
    
    
def inject_sources(input_exp, visit_summary, catalog_of_injection, plot = False):

    '''
    Injects sources into an input exposure using the provided catalog of injection sources and visit summary data.

    Parameters:
    -----------
    input_exp : Exposure
        The input exposure where sources will be injected.
    visit_summary : VisitSummary
        Summary of the visit, containing detector-specific information like PSF, photometric calibration, and WCS. Obtained running 
        visit_summary = butler.get("finalVisitSummary", dataId=ref, collections=collection,).find(ref["detector"])
    catalog_of_injection : dict
        Catalog of sources to be injected, typically containing positions, magnitudes, etc.
    plot : bool, optional
        If True, displays a plot of the injected sources on the exposure (default is False).

    Returns:
    --------
    tuple
        A tuple containing the injected exposure and the catalog of injected sources.
        - injected_exposure (Exposure): The exposure object after sources have been injected.
        - injected_catalog (Catalog): The catalog of injected sources with updated properties.
    '''    
    input_exposure = input_exp
  

    # NOTE: Visit-level injections should instead use the visit summary table.
    detector_summary = visit_summary
    psf = detector_summary.getPsf()
    photo_calib = detector_summary.getPhotoCalib()
    wcs = detector_summary.getWcs()


    # Instantiate the injection classes.
    inject_config = VisitInjectConfig(process_all_data_ids  = False)
    inject_task = VisitInjectTask(config=inject_config)

    # Run the source injection task.
    injected_output = inject_task.run(
        injection_catalogs = catalog_of_injection,
        input_exposure = input_exposure.clone(),
        psf = psf,
        photo_calib = photo_calib,
        wcs = wcs,
    )
    
    injected_exposure=injected_output.output_exposure
    injected_catalog=injected_output.output_catalog
    
    if (plot == True) : 
        afwDisplay.setDefaultBackend('matplotlib')
        fig = plt.figure(figsize=(10, 8))
        afw_display = afwDisplay.Display(1)
        afw_display.setMaskTransparency(100)
        afw_display.setMaskTransparency(50, name="INJECTED")
        afw_display.scale('asinh', 'zscale')
        afw_display.mtv(injected_exposure)
        plt.show()
    
    return injected_exposure, injected_catalog




def create_catalog_for_all_ccd(datasetRefs, butler, band, csv = False, save_filename = None):
    all_catalog = pd.DataFrame()

    for i,reference in enumerate(datasetRefs):
        ref = reference.dataId
        if ref['band'] == band : 

            detecs = butler.get('sourceTable', dataId=ref)
            calexp = butler.get('calexp', dataId=ref)

            filtered_detecs = detecs[detecs['extendedness'] == 1]
            # Sample n rows from the filtered DataFrame    
            nbr_fake = int(len(filtered_detecs)/20)
            if nbr_fake == 0:
                print(ref)
                print('nbr of galaxy: %d    nbr of injection: %d' % (len(filtered_detecs), nbr_fake))
                print('Cannot proceed, no fake detections to sample. Skipping to next reference.')
                continue
            sampled_detecs = filtered_detecs.sample(n=nbr_fake, random_state=42)
            finite_mag_per = sum(np.isfinite((flux * u.nJy).to(u.ABmag).value) for flux in sampled_detecs['calibFlux'])/len(sampled_detecs)*100
            if finite_mag_per<=0:
                
                print(ref)
                print('nbr of galaxy : %d    nbr of injection : %d    percentage of real mag : %d' % (len(filtered_detecs), nbr_fake, finite_mag_per))
                print('cannot be stored, no flux')
            
            else :
                print(ref)
                print('nbr of galaxy : %d    nbr of injection : %d    percentage of real mag : %d' % (len(filtered_detecs), nbr_fake, finite_mag_per))
                
                ra, dec, mags, df = pos_mag_4catalog(sampled_detecs, calexp, band, False, True)

                fancy_catalog = create_catalog(df, True).to_pandas()
                all_catalog = pd.concat([all_catalog, fancy_catalog], ignore_index=True)

    if csv : 
        all_catalog.to_csv(f'saved/{save_filename}.csv')


    return create_catalog_for_all_ccd