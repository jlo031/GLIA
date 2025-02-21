# ---- This is <batch_classification.py> ----

import os
import pathlib

from loguru import logger

import numpy as np
import matplotlib.pyplot as plt
from osgeo import gdal

import GLIA_classifier.gaussian_linear_IA_classifier as glia
import GLIA_classifier.classification as glass

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

# define data directories
base_dir = pathlib.Path('./test_data').resolve()

# build further directory structure from base_dir
feature_dir = base_dir / 'features'
results_dir = base_dir / 'results'

# define classifier model
clf_model_path = pathlib.Path('../src/GLIA_classifier/clf_models/belgica_bank_ice_types_2022.pickle').resolve()

# -------------------------------------------------------------------------- #

# set parameters for classification

use_valid_mask = True
estimate_uncertainties = True
uncertainty_params_dict = []
overwrite = True
loglevel = 'INFO'

# -------------------------------------------------------------------------- #

# list all folders in feature_dir
# this folder should only contain feature folders of individual images
# if needed, you 
img_list = os.listdir(feature_dir)

logger.info(f'Found {len(img_list)} imager folders in feature_dir')

for i,img in enumerate(img_list):

    logger.info(f'Processing image {i+1} of {len(img_list)}')

    current_feature_dir = feature_dir / img

    # create and save valid.img if needed
    # you can add invalid swaths, data points, etc, as needed
    if use_valid_mask == True and not (current_feature_dir/'valid.img').is_file():
        land_mask = gdal.Open((current_feature_dir/'landmask.img').as_posix()).ReadAsArray()
        HH = gdal.Open((current_feature_dir/'Sigma0_HH_db.img').as_posix()).ReadAsArray()
        valid_mask = np.ones(land_mask.shape)
        valid_mask[land_mask==1] = 0
        valid_mask[HH==0] = 0
        # write labels
        out = gdal.GetDriverByName('Envi').Create((current_feature_dir/'valid.img').as_posix(), valid_mask.shape[1], valid_mask.shape[0], 1, gdal.GDT_Byte)
        out.GetRasterBand(1).WriteArray(valid_mask)
        out.FlushCache()

    glass.classify_image_from_feature_folder(
        current_feature_dir,
        results_dir,
        clf_model_path,
        use_valid_mask = use_valid_mask,
        estimate_uncertainties = estimate_uncertainties,
        uncertainty_params_dict = uncertainty_params_dict,
        overwrite = overwrite,
        loglevel = loglevel,
    )

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

# ---- End of <batch_classification.py> ----




