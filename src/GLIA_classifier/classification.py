# ---- This is <classification.py> ----

"""
Module for forward classification of satellite images
""" 

import argparse
import os
import sys
import pathlib
import shutil
import copy

from loguru import logger

import numpy as np

from osgeo import gdal

import GLIA_classifier.gaussian_linear_IA_classifier as glia
import GLIA_classifier.uncertainty_utils as glia_uncertainty_utils

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

def classify_image_from_feature_folder(
    feature_folder,
    result_folder,
    clf_model_path,
    valid_mask = True,
    estimate_uncertainties = False,
    uncertainty_params_dict = [],
    overwrite = False,
    loglevel = 'INFO',
):

    """Classify input image with classifier from clf_model_path and features from feature_folder 

    Parameters
    ----------
    feature_folder : path to input feature folder
    result_folder : path to result folder where labels file is placed
    clf_model_path : path to pickle file with classifier model dict
    estimate_uncertainties : estimate apost and mahal uncertainties (default True)
    uncertainty_params_dict : dictionary with parameters for uncertainty estimation
    valid_mask : use valid mask
    overwrite : overwrite existing files (default=False)
    loglevel : loglevel setting (default='INFO')
    """

    # remove default logger handler and add personal one
    logger.remove()
    logger.add(sys.stderr, level=loglevel)

    logger.info('Processing input image for classification')

# -------------------------------------------------------------------------- #

    # convert folder strings to paths
    feature_folder = pathlib.Path(feature_folder).resolve()
    result_folder  = pathlib.Path(result_folder).resolve()
    clf_model_path = pathlib.Path(clf_model_path).resolve()

    logger.debug(f'feature_folder: {feature_folder}')
    logger.debug(f'result_folder: {result_folder}')
    logger.debug(f'clf_model_path: {clf_model_path}')

    if not feature_folder.is_dir():
        logger.error(f'Cannot find feature_folder: {feature_folder}')
        return

    if not clf_model_path.is_file():
        logger.error(f'Cannot find clf_model_path: {clf_model_path}')
        return

# -------------------------------------------------------------------------- #

    # get input basename from feature_folder
    f_base = feature_folder.stem

    logger.debug(f'f_base: {f_base}')

    # define output file names and paths
    result_labels_path      = result_folder / f'{f_base}_labels.img'
    result_labels_path_hdr  = result_folder / f'{f_base}_labels.hdr'
    result_mahal_path       = result_folder / f'{f_base}_mahal_uncertainty.img'
    result_mahal_path_hdr   = result_folder / f'{f_base}_mahal_uncertainty.hdr'
    result_apost_path       = result_folder / f'{f_base}_apost_uncertainty.img'
    result_apost_path_hdr   = result_folder / f'{f_base}_apost_uncertainty.hdr'

    logger.debug(f'result_labels_path: {result_labels_path}')
    logger.debug(f'result_mahal_path:  {result_mahal_path}')
    logger.debug(f'result_apost_path:  {result_apost_path}')

    # check if main outfile already exists
    if result_labels_path.is_file() and not overwrite:
        logger.info('Output files already exist, use `-overwrite` to force')
        return
    elif result_labels_path.is_file() and overwrite:
        logger.info('Removing existing output file and classifying again')
        result_labels_path.unlink()
        result_labels_path_hdr.unlink()
        result_mahal_path.unlink(missing_ok=True)
        result_mahal_path_hdr.unlink(missing_ok=True)
        result_apost_path.unlink(missing_ok=True)
        result_apost_path_hdr.unlink(missing_ok=True)

# -------------------------------------------------------------------------- #

    # GET BASIC CLASSIFIER INFO

    # load classifier dictionary
    clf_params_dict = glia.read_classifier_dict_from_pickle(clf_model_path.as_posix())

    # check that it is a valid classifier dict with all required information
    valid_clf_params_dict = glia.check_clf_params_dict(clf_params_dict, loglevel=loglevel)

    if not valid_clf_params_dict:
        logger.error(f'Invalid clf_params_dict')
        return

    # get clf_type
    clf_type = clf_params_dict['type']

    # get list of required features
    required_features = sorted(clf_params_dict['required_features'])

    logger.info(f'clf_type: {clf_type}')
    logger.info(f'required_features: {required_features}')

# ---------------------------------- #

    # build clf object

    if clf_type == 'GLIA':
        clf = glia.make_GLIA_clf_object_from_clf_params_dict(clf_params_dict)

    elif clf_type =='gaussian':
        clf = gia.make_gaussian_clf_object_from_params_dict(clf_params_dict)

    else:
        logger.error('This clf type is not implemented in this library')
        return

# -------------------------------------------------------------------------- #

    # PREPARE UNCERTAINTY ESTIMATION

    if estimate_uncertainties:
        logger.info('Preparing uncertainty estimation')

        # extract clf parameters needed for uncertainties
        if clf_type == 'GLIA':
            mu_vec_all_classes  = clf_params_dict['mu']
            cov_mat_all_classes = clf_params_dict['Sigma']
            n_classes           = int(clf_params_dict['n_class'])
            n_features          = int(clf_params_dict['n_feat'])
            IA_0                = clf_params_dict['IA_0']
            IA_slope            = clf_params_dict['b']
        elif clf_type =='gaussian':
            mu_vec_all_classes   = clf_params_dict['mu']
            cov_vmat_all_classes = clf_params_dict['Sigma']
            n_classes            = int(clf_params_dict['n_class'])
            n_features           = int(clf_params_dict['n_feat'])
        else:
            logger.error('This clf type is not implemented in this library')
            return


        # set default values for uncertainty estimation
        uncertainty_params = dict()
        uncertainty_params['apost_uncertainty_measure'] = 'Entropy'
        uncertainty_params['DO_apost_uncertainty'] = True
        uncertainty_params['DO_mahal_uncertainty'] = True
        uncertainty_params['discrete_uncertainty'] = True
        uncertainty_params['mahal_thresh_min'] = 6
        uncertainty_params['mahal_thresh_max'] = 12
        uncertainty_params['mahal_discrete_thresholds'] = np.array([6, 8, 10, 12])
        uncertainty_params['apost_discrete_thresholds'] = ['default']

        valid_uncertainty_keys = uncertainty_params.keys()

        # user uncertainty_params_dict if given
        if uncertainty_params_dict == []:
            logger.info('Using default parameters for uncertainty estimation')
        elif type(uncertainty_params_dict) is not dict:
            logger.error(f'uncertainty_params_dict must be a dictionary, but data type was {type(uncertainty_params_dict)}.')
            return

        else:
            logger.info('Using parameters from uncertainty_params_dict for uncertainty estimation')

            uncertainty_keys = uncertainty_params_dict.keys()
            logger.debug(f'uncertainty_params_dict.keys(): {uncertainty_keys}')

            # overwrite uncertainty_params with correct values from uncertainty_params_dict
            for key in uncertainty_keys:
                if key in valid_uncertainty_keys:
                    logger.debug(f'overwriting default value for "{key}" with: {uncertainty_params_dict[key]}')
                    uncertainty_params[key] = uncertainty_params_dict[key]
                else:
                    logger.warning(f'uncertainty_params_dict key "{key}" is unknown and will not be used')

# -------------------------------------------------------------------------- #

    # CHECK ADN LOAD FEATURES

    # get list of existing features
    existing_features = sorted([f for f in os.listdir(feature_folder)])

    logger.debug(f'Found feature files in feature_folder: {existing_features}')
    logger.info('Loading required features')

    # features will be read into a dictionary
    feature_dict = dict()

    # loop through required features, check, and load
    for f in required_features:

        if f'{f}.img' not in existing_features:
            logger.error(f'Could not find required feature: {f}.img')
            return
        else:
            feature_dict[f] = gdal.Open((feature_folder/f'{f}.img').as_posix()).ReadAsArray()

    # get Nx and Ny from first required feature
    Ny, Nx = feature_dict[required_features[0]].shape
    shape  = (Ny, Nx)
    N      = Nx*Ny

    # check that all required features have same dimensions
    for key in feature_dict.keys():
        Ny_current, Nx_current = feature_dict[key].shape
        if not Nx == Nx_current or not Ny == Ny_current:
            logger.error(f'Image dimensions of required features do not match')
            return

# -------------------------------------------------------------------------- #

    # CHECK AND LOAD VALID MASK

    if valid_mask:

        if 'valid.img' not in existing_features:
            logger.error(f'Could not find valid mask: valid.img')
            return
        else:
            logger.info('Loading valid mask')
            valid_mask = gdal.Open((feature_folder/'valid.img').as_posix()).ReadAsArray()

        # check that valid_mask dimensions match feature dimensions
        if not valid_mask.shape[0]==Ny and valid_mask.shape[1]==Nx:
            logger.error(f'valid_mask dimensions do not match feature dimensions')
            return

    else:
        valid_mask = np.ones(shape)

# ---------------------------------- #

    # CHECK AND LOAD IA MASK

    if clf_type == 'GLIA':

        # check that IA.img exists
        if 'IA.img' not in existing_features:
            logger.error(f'Could not find IA mask: IA.img')
            return
        else:
            logger.info('Loading IA mask')
            IA = gdal.Open((feature_folder/'IA.img').as_posix()).ReadAsArray()

        # check that IA dimensions match feature dimensions
        if not IA.shape[0]==Ny and IA.shape[1]==Nx:
            logger.error(f'IA dimensions do not match featured imensions')
            return

# -------------------------------------------------------------------------- #

    # initialize labels and probabilities
    labels = np.zeros(N)

    if estimate_uncertainties:
        mahal = np.zeros((N,n_classes))
        mahal.fill(np.nan)
        probs = np.zeros((N,n_classes))
        probs.fill(np.nan)




    """
    # find number of blocks from block_size
    n_blocks   = int(np.ceil(N/block_size))

    # logger
    logger.info('Performing block-wise processing of memory-mapped data')
    logger.info(f'block-size: {block_size}')
    logger.info(f'Number of blocks: {n_blocks}')

    # for progress report at every 10%
    log_percs   = np.array([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9])
    perc_blocks = np.ceil(np.array(n_blocks)*log_percs).astype(int)
    """
# -------------------------------------------------------------------------- #

    # STACK FEATURES TO FEATURE, VALID, AND IA VECTORS

    # initialize stacked feature vector
    X = np.zeros((N, len(required_features)))

    for i,feature in enumerate(required_features):
        X[:,i] = feature_dict[feature].flatten()

    IA = IA.flatten()
    valid_mask = valid_mask.flatten()

    ##return feature_dict, IA, valid_mask, X

# -------------------------------------------------------------------------- #

    # CLASSIFY

    # predict labels where valid==1
    if clf_type == 'GLIA':

        if estimate_uncertainties:
            labels[valid_mask==1], probs[valid_mask==1] = clf.predict(X[valid_mask==1], IA[valid_mask==1])

            # for uncertainties
            logger.debug('Estimating mahal_img for current block')
            mahal[valid_mask==1] = glia_uncertainty_utils.get_mahalanobis_distance(X[valid_mask==1], mu_vec_all_classes, cov_mat_all_classes, IA_test=IA[valid_mask==1], IA_0=IA_0, IA_slope=IA_slope)
	
        else:
            labels[valid_mask==1], _ = clf.predict(X[valid_mask==1], IA[valid_mask==1])


    elif clf_type == 'gaussian':

        if estimate_uncertainties:
            labels[valid_mask==1],probs[valid_mask==1] = clf.predict(X[valid_mask==1])

            # for uncertainties
            logger.debug('Estimating mahal for current block')
            mahal[valid_mask==1] = glia_uncertainty_utils.get_mahalanobis_distance(X[valid_mask==1], mu_vec_all_classes, cov_mat_all_classes)

        else:
            labels[valid_mask==1], _ = clf.predict(X[valid_mask==1])

    else:
        logger.error('This clf type is not implemented in this library')
        return



    # set labels to 0 where valid==0
    labels[valid_mask==0] = 0

    if estimate_uncertainties:
        mahal[valid_mask==0] = 0

    logger.info('Finished classification')

# -------------------------------------------------------------------------- #

    if estimate_uncertainties:

        logger.info('Estimating apost and mahal uncertainties')

        uncertainty_apost, uncertainty_mahal = glia_uncertainty_utils.uncertainty(
            probs,
            mahal,
            n_features,
            apost_uncertainty_meausure = uncertainty_params['apost_uncertainty_measure'],
            DO_apost_uncertainty = uncertainty_params['DO_apost_uncertainty'],
            DO_mahalanobis_uncertainty = uncertainty_params['DO_mahal_uncertainty'],
            discrete_uncertainty = uncertainty_params['discrete_uncertainty'],
            mahal_thresh_min = uncertainty_params['mahal_thresh_min'],
            mahal_thresh_max = uncertainty_params['mahal_thresh_max'],
            mahal_discrete_thresholds = uncertainty_params['mahal_discrete_thresholds'],
            apost_discrete_thresholds = uncertainty_params['apost_discrete_thresholds']
        )

# -------------------------------------------------------------------------- #

    # reshape to image geometry
    labels = np.reshape(labels, shape)

    if estimate_uncertainties:
        if uncertainty_mahal is not False:
            uncertainty_mahal  = np.reshape(uncertainty_mahal, shape)
        if uncertainty_apost is not False:
            uncertainty_apost  = np.reshape(uncertainty_apost, shape)

        logger.info('Finished uncertainty estimation')

# -------------------------------------------------------------------------- #

    # create result_folder if needed
    result_folder.mkdir(parents=True, exist_ok=True)

    # write labels
    output_labels = gdal.GetDriverByName('Envi').Create(result_labels_path.as_posix(), Nx, Ny, 1, gdal.GDT_Byte)
    output_labels.GetRasterBand(1).WriteArray(labels)
    output_labels.FlushCache()


    # write uncertainties
    if estimate_uncertainties:
        if uncertainty_mahal is not False:
            output_mahal = gdal.GetDriverByName('Envi').Create(result_mahal_path.as_posix(), Nx, Ny, 1, gdal.GDT_Float32)
            output_mahal.GetRasterBand(1).WriteArray(uncertainty_mahal)
            output_mahal.FlushCache()
        if uncertainty_apost is not False:
            output_apost = gdal.GetDriverByName('Envi').Create(result_apost_path.as_posix(), Nx, Ny, 1, gdal.GDT_Float32)
            output_apost.GetRasterBand(1).WriteArray(uncertainty_apost)
            output_apost.FlushCache()

    logger.info(f'Result writtten to {result_labels_path}')

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

# ---- End of <classifcation.py> ----
