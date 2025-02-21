# ---- This is <classify_test_image.py> ----

import os
import pathlib

from loguru import logger

import numpy as np
import matplotlib.pyplot as plt
from osgeo import gdal

import GLIA_classifier.gaussian_linear_IA_classifier as glia

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

# define data directories
base_dir = pathlib.Path('./test_data').resolve()

# define image basename
# usually just the S1 name string, here the image is heavily subsampled to reduce file size in this demo
# in practice, you can of course loop over multiple basebames here
img_basename = 'S1A_EW_GRDM_1SDH_20220503T082621_20220503T082725_043044_0523D1_AF89_subsampled'

# define classifier model
clf_pickle_path = pathlib.Path('../src/GLIA_classifier/clf_models/belgica_bank_ice_types_2022.pickle').resolve()

# -------------------------------------------------------------------------- #

# build directory strucutre
feature_dir = base_dir / 'features' / f'{img_basename}'
results_dir = base_dir / 'classification_results'

# -------------------------------------------------------------------------- #

logger.info(f'Main data directory is: {base_dir}')
logger.info(f'Current feature directory is: {feature_dir}')
logger.info(f'Results will be written to: {results_dir}')
logger.info(f'Classifier model is: {clf_pickle_path}')

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

# load the classifier and build clf object
clf_params_dict = glia.read_classifier_dict_from_pickle(clf_pickle_path)
clf = glia.make_GLIA_clf_object_from_clf_params_dict(clf_params_dict)

# get some information required about the classifier
required_features = clf_params_dict['required_features']
clf_type = clf_params_dict['type']

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

# You can implement a variety of checks that ensure you are using the correct classifier with the correct data
# Here are some examples of how to implement these
# Further checks about dimensions, valid masks, etc should be implemented if necessary

# CHECK FEATURES

# get list of existing features in feat_folder
existing_features = sorted([f for f in os.listdir(feature_dir) if f.endswith('img')])


# check that all required_features exist
features_exist = True
for f in required_features:
    if f'{f}.img' not in existing_features:
        logger.error(f'Cannot find required feature: {f}')
        features_exist = False

if features_exist:
    logger.info('All required features found in feature_dir')

# -------------------------------------------------------------------------- #

# CHECK IA MASK

if clf_type == 'GLIA':
    logger.info('Classifier requires IA information')

    # check that IA.img exists
    if 'IA.img' not in existing_features:
        logger.error(f'Cannot find IA image: IA.img')

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

# load image files
HH = gdal.Open((feature_dir/'Sigma0_HH_db.img').as_posix()).ReadAsArray()
HV = gdal.Open((feature_dir/'Sigma0_HV_db.img').as_posix()).ReadAsArray()
IA = gdal.Open((feature_dir/'IA.img').as_posix()).ReadAsArray()

# get image dimensions
img_dims = HH.shape

# stack data to feature vector X
X_vec = np.stack((HH.ravel(), HV.ravel()),1)
IA_vec = IA.ravel()

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

# predict the labels
y_pred, prob = clf.predict(X_vec, IA_vec)

# and reshape to image dimension
y_img = y_pred.reshape(img_dims)

# crude application of a "valid_mask"
y_img[HH==0] = 0

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

# visualize for a qick check
fig, axes = plt.subplots(2,2,sharex=True,sharey=True)
axes = axes.ravel()
axes[0].imshow(HH, cmap='gray')
axes[1].imshow(HV, cmap='gray')
axes[2].imshow(IA, cmap='gray')
axes[3].imshow(y_img)

plt.show()

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

# save results to disk

# make sure results_dir exists
results_dir.mkdir(exist_ok=True)

# output path
outfile = results_dir/f'{img_basename}_labels.img'

# output dimensions
Nx, Ny = y_img.shape

# write
output = gdal.GetDriverByName('ENVI').Create(outfile.as_posix(), Ny, Nx, 1, gdal.GDT_Byte)
output.GetRasterBand(1).WriteArray(y_img)
output.FlushCache()
output = None

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

# ---- End of <classify_test_image.py> ----




