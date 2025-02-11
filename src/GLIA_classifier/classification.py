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

import ice_type_classification.uncertainty_utils as uncertainty_utils
import ice_type_classification.classification_utils as classification_utils

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

def classify_image_from_feature_folder(
    feature_folder,
    result_folder,
    classifier_model_path,
    valid_mask = True,
    uncertainties = False,
    uncertainty_params_dict = [],
    overwrite = False,
    loglevel = 'INFO',
):

    """Classify input image

    Parameters
    ----------
    feat_folder : path to input feature folder
    result_folder : path to result folder where labels file is placed
    classifier_model_path : path to pickle file with classifier model dict
    uncertainties : estimate apost and mahal uncertainties (default True)
    uncertainty_params_dict : dictionary with parameters for uncertainty estimation
    valid_mask : use valid mask
    overwrite : overwrite existing files (default=False)
    loglevel : loglevel setting (default='INFO')
    """

    # remove default logger handler and add personal one
    logger.remove()
    logger.add(sys.stderr, level=loglevel)

    logger.info('Classifying input image')

# -------------------------------------------------------------------------- #

    # convert folder strings to paths
    feat_folder           = pathlib.Path(feat_folder).resolve()
    result_folder         = pathlib.Path(result_folder).resolve()
    classifier_model_path = pathlib.Path(classifier_model_path).resolve()

    logger.debug(f'feat_folder: {feat_folder}')
    logger.debug(f'result_folder: {result_folder}')
    logger.debug(f'classifier_model_path: {classifier_model_path}')

    if not feat_folder.is_dir():
        logger.error(f'Cannot find feat_folder: {feat_folder}')
        return

    if not classifier_model_path.is_file():
        logger.error(f'Cannot find classifier_model_path: {classifier_model_path}')
        return

# -------------------------------------------------------------------------- #

    # get input basename from feat_folder
    f_base = feat_folder.stem

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
    clf_params_dict = glia.read_classifier_dict_from_pickle(classifier_model_path.as_posix())

    # check that it is a valid classifier dict with all required information
    valid_clf_params_dict = glia.check_clf_dict(clf_params_dict)

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
    """
    if uncertainties:
        logger.info('Uncertainties is set to "True"')

        # extract clf parameters needed for uncertainties
        if clf_type == 'gaussian_IA':
            mu_vec_all_classes  = classifier_dict['gaussian_IA_params']['mu']
            cov_mat_all_classes = classifier_dict['gaussian_IA_params']['Sigma']
            n_classes           = int(classifier_dict['gaussian_IA_params']['n_class'])
            n_features          = int(classifier_dict['gaussian_IA_params']['n_feat'])
            IA_0                = classifier_dict['gaussian_IA_params']['IA_0']
            IA_slope            = classifier_dict['gaussian_IA_params']['b']
        elif clf_type =='gaussian':
            mu_vec_all_classes   = classifier_dict['gaussian_params']['mu']
            cov_vmat_all_classes = classifier_dict['gaussian_params']['Sigma']
            n_classes            = int(classifier_dict['gaussian_params']['n_class'])
            n_features           = int(classifier_dict['gaussian_params']['n_feat'])
        else:
            logger.error('This clf type is not implemented yet')
            raise NotImplementedError('This clf type is not implemented yet')


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


        # user uncertainty_dict if given
        if uncertainty_dict == []:
            logger.info('Using default parameters for uncertainty estimation')

        elif type(uncertainty_dict) == dict:
            logger.info('Using parameters from uncertainty_dict for uncertainty estimation')

            uncertainty_keys = uncertainty_dict.keys()
            logger.debug(f'uncertainty_dict.keys(): {uncertainty_keys}')

            # overwrite uncertainty_params with correct values from uncertainty_dict
            for key in uncertainty_keys:
                if key in valid_uncertainty_keys:
                    logger.debug(f'overwriting default value for "{key}" with: {uncertainty_dict[key]}')
                    uncertainty_params[key] = uncertainty_dict[key]
                else:
                    logger.warning(f'uncertainty_dict key "{key}" is unknown and will not be used')

        else:
            logger.warning(f'Expected "uncertainty_dict" of type "dict", but found type "{type(uncertainty_dict)}"')
            logger.warning('Using default parameters for uncertainty estimation')
    """
# -------------------------------------------------------------------------- #

    # CHECK EXISTING AND REQUIRED FEATURES
    # AND LOAD DATA

    # get list of existing features
    existing_features = sorted([f for f in os.listdir(feature_folder) if f.endswith('img') or f.endswith('tif') or f.endswith('tiff')])

    # features will be read into a dictionary
    feature_dict = dict()

    # loop through required features, check, and load
    for f in required_features:

        feature_found = False

        if f'{f}.img' in existing_features:
            feature_found = True
            feature_dict[f] = gdal.Open((feature_folder/f'{f}.img').as_posix()).ReadAsArray()
         # check for tif/tiff and avoid doubles 


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

    # CHECK VALID MASK

    if valid_mask:
        logger.info('Using valid mask')


        if 'valid.img' in existing_features:
            valid_mask = gdal.Open((feature_folder/'valid.img').as_posix()).ReadAsArray()
        # same clever way to check for tif/tiff

        # check that valid_mask dimensions match feature dimensions
        if not valid_mask.shape[0]==Ny and valid_mask.shape[1]==Hx:
            logger.error(f'valid_mask dimensions do not match featured imensions')
            return

# ---------------------------------- #

    # CHECK IA MASK

    if clf_type == 'GLIA':
        logger.info('Classifier requires IA information')

        # check that IA.img exists
        if 'IA.img' in existing_features:
            IA = gdal.Open((feature_folder/'IA.img').as_posix()).ReadAsArray()
        # same checks as above

        # check that IA dimensions match feature dimensions
        if not IA.shape[0]==Ny and IA.shape[1]==Nx:
            logger.error(f'IA dimensions do not match featured imensions')
            return

# -------------------------------------------------------------------------- #

    """
    may not be needed without memory mapping
    # initialize labels and probabilities
    labels_img = np.zeros(N)

    if uncertainties:
        # for uncertainties
        mahal_img  = np.zeros((N,n_classes))
        mahal_img.fill(np.nan)
        probs_img  = np.zeros((N,n_classes))
        probs_img.fill(np.nan)

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

    labels = np.zeros(N)

    if uncertainties:
        mahal  = np.zeros((N,n_classes))
        mahal.fill(np.nan)
        probs  = np.zeros((N,n_classes))
        probs.fill(np.nan)



    # STACK FEATURES TO FEATURE, VALID, AND IA VECTORS

    X
    valid_vec
    IA_vec

# -------------------------------------------------------------------------- #

    # CLASSIFY

    # predict labels where valid==1
    if clf_type == 'gaussian_IA':

        if uncertainties:
            labels[valid_vec==1], probs[valid_vec==1] = clf.predict(X_vec, IA_vec)

            # for uncertainties
            logger.debug('Estimating mahal_img for current block')
            mahal[valid_vec] = uncertainty_utils.get_mahalanobis_distance(X, mu_vec_all_classes, cov_mat_all_classes, IA_test=IA_vec, IA_0=IA_0, IA_slope=IA_slope)
	
        else:
            labels[valid_vec==1], _ = clf.predict(X, IA_vec)


    elif clf_type == 'gaussian':

        if uncertainties:
            labels[valid_vec==1],probs[valid_block==1] = clf.predict(X)

            # for uncertainties
            logger.debug('Estimating mahal for current block')
            mahal_img[valid_vec==1] = uncertainty_utils.get_mahalanobis_distance(X, mu_vec_all_classes, cov_mat_all_classes)

        else:
            labels[valid_block==1], _ = clf.predict(X)

    else:
        logger.error('This clf type is not implemented yet')



    # set labels to 0 where valid==0
    labels_img[idx_start:idx_end][valid_block==0] = 0

    logger.info('Finished classification')

# -------------------------------------------------------------------------- #

    if uncertainties:

        logger.info('Estimating apost and mahal uncertainties')

        uncertainty_apost, uncertainty_mahal = uncertainty_utils.uncertainty(
            probs_img,
            mahal_img,
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
    labels_img = np.reshape(labels_img, shape)

    if uncertainties:
        if uncertainty_mahal is not False:
            uncertainty_mahal  = np.reshape(uncertainty_mahal, shape)
        if uncertainty_apost is not False:
            uncertainty_apost  = np.reshape(uncertainty_apost, shape)

        logger.info('Finished uncertainty estimation')

# -------------------------------------------------------------------------- #

    # create result_folder if needed
    result_folder.mkdir(parents=True, exist_ok=True)

    # write labels
    output_labels = gdal.GetDriverByName('Envi').Create(result_path.as_posix(), Nx, Ny, 1, gdal.GDT_Byte)
    output_labels.GetRasterBand(1).WriteArray(labels_img)
    output_labels.FlushCache()


    # write uncertainties
    if uncertainties:
        if uncertainty_mahal is not False:
            output_mahal = gdal.GetDriverByName('Envi').Create(result_path_mahal.as_posix(), Nx, Ny, 1, gdal.GDT_Float32)
            output_mahal.GetRasterBand(1).WriteArray(uncertainty_mahal)
            output_mahal.FlushCache()
        if uncertainty_apost is not False:
            output_apost = gdal.GetDriverByName('Envi').Create(result_path_apost.as_posix(), Nx, Ny, 1, gdal.GDT_Float32)
            output_apost.GetRasterBand(1).WriteArray(uncertainty_apost)
            output_apost.FlushCache()

    logger.info(f'Result writtten to {result_path}')

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

def inspect_classifier_pickle(
    classifier_model_path,
    loglevel='INFO',
):

    """Retrieve information about a classifier stored in a pickle file

    Parameters
    ----------
    classifier_mode_path : path to pickle file with classifier dict
    loglevel : loglevel setting (default='INFO')
    """

    # remove default logger handler and add personal one
    logger.remove()
    logger.add(sys.stderr, level=loglevel)

    logger.info('Inspecting classifier pickle file')

# -------------------------------------------------------------------------- #

    # convert folder strings to paths
    classifier_model_path = pathlib.Path(classifier_model_path).resolve()

    logger.debug(f'classifier_model_path: {classifier_model_path}')

    if not classifier_model_path.is_file():
        logger.error(f'Cannot find classifier_model_path: {classifier_model_path}')
        raise FileNotFoundError(f'Cannot find classifier_model_path: {classifier_model_path}')

# -------------------------------------------------------------------------- #

    # load classifier dictionary
    classifier_dict = gia.read_classifier_dict_from_pickle(classifier_model_path.as_posix())

    # check that pickle file contains a dictionary
    if type(classifier_dict) is not dict:
        logger.error(f'Expected a classifier dictionary, but type is {type(classifier_dict)}')
        raise TypeError(f'Expected a classifier dictionary, but type is {type(classifier_dict)}')

    logger.debug('pickle file contains a classifier dictionary')
    logger.debug(f'dict keys are: {list(classifier_dict.keys())}')

    if not 'type' in classifier_dict.keys():
        logger.error(f'classifier_dict does not contain `type` key')
        raise KeyError(f'classifier_dict does not contain `type` key')

    if not 'required_features' in classifier_dict.keys():
        logger.error(f'classifier_dict does not contain `required_features` key')
        raise KeyError(f'classifier_dict does not contain `required_features` key')

    if not 'label_value_mapping' in classifier_dict.keys():
        logger.error(f'classifier_dict does not contain `label_value_mapping` key')
        raise KeyError(f'classifier_dict does not contain `label_value_mapping` key')

    if not 'trained_classes' in classifier_dict.keys():
        logger.error(f'classifier_dict does not contain `trained_classes` key')
        raise KeyError(f'classifier_dict does not contain `trained_classes` key')

    if not 'invalid_swaths' in classifier_dict.keys():
        logger.error(f'classifier_dict does not contain `invalid_swaths` key')
        raise KeyError(f'classifier_dict does not contain `invalid_swaths` key')

    if not 'info' in classifier_dict.keys():
        logger.error(f'classifier_dict does not contain `info` key')
        raise KeyError(f'classifier_dict does not contain `info` key')

# -------------------------------------------------------------------------- #

    # a "gaussian_IA" type classifier must have a "gaussian_IA_params" key
    # values in this key are used to build the classifier object when needed
    # this circumvents issues with changes in the gaussin_IA_classifier module

    if classifier_dict['type'] == 'gaussian_IA':
        if not 'gaussian_IA_params' in classifier_dict.keys():
            logger.error(f'classifier_dict does not contain `gaussian_IA_params` key')
            raise KeyError(f'classifier_dict does not contain `gaussian_IA_params` key')
      

    # extract information for inspection output
    classifier_type     = classifier_dict['type']
    features            = classifier_dict['required_features']
    label_value_mapping = classifier_dict['label_value_mapping']
    trained_classes     = classifier_dict['trained_classes']
    invalid_swaths      = classifier_dict['invalid_swaths']
    info                = classifier_dict['info']


    print(f'\n=== CLASSIFIER ===')
    print(classifier_model_path)

    print('\n=== CLASSIFIER TYPE: ===')
    print(classifier_type)

    print('\n=== REQUIRED FEATURES: ===')
    for idx, feature_name in enumerate(features):
        print(f'{idx:2d} -- {feature_name}')

    print('\n=== LABEL VALUE MAPPING: ===')
    for idx, key in enumerate(label_value_mapping):
        print(f'{key} -- {label_value_mapping[key]}')

    print('\n=== TRAINED CLASSES: ===')
    print(f'{trained_classes}')

    print('\n=== INVALID SWATHS: ===')
    print(f'{invalid_swaths}')

    if 'texture_settings' in classifier_dict.keys():
        print('\n=== TEXTURE PARAMETER SETTINGS: ===')
        for idx, key in enumerate(classifier_dict['texture_settings']):
            print(f'{key}: {classifier_dict["texture_settings"][key]}')

    print('\n=== INFO: ===')
    print(f'{info}')

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

# ---- End of <classifcation.py> ----
