import GLIA_classifier.gaussian_linear_IA_classifier as glia

import GLIA_classifier.classification as glass

import numpy as np

feat_folder = 'tests/test_data/features/S1A_EW_GRDM_1SDH_20220503T082621_20220503T082725_043044_0523D1_AF89_subsampled'
result_folder = 'tests/test_data/results'
clf_model = 'src/GLIA_classifier/clf_models/belgica_bank_ice_types_2022.pickle'

loglevel = 'DEBUG'
loglevel = 'INFO'

valid_mask = False
estimate_uncertainties = True
uncertainty_params_dict = []
overwrite = True








# set default values for uncertainty estimation
uncertainty_params_dict = dict()
uncertainty_params_dict['apost_uncertainty_measure'] = 'Entropy'
uncertainty_params_dict['DO_apost_uncertainty'] = True
uncertainty_params_dict['DO_mahal_uncertainty'] = True
uncertainty_params_dict['discrete_uncertainty'] = True
uncertainty_params_dict['mahal_thresh_min'] = 6
uncertainty_params_dict['mahal_thresh_max'] = 12
uncertainty_params_dict['mahal_discrete_thresholds'] = np.array([6, 8, 10, 12])
uncertainty_params_dict['apost_discrete_thresholds'] = ['default']




"""
feature_dict, IA, valid_mask, X = glass.classify_image_from_feature_folder(
    feat_folder,
    result_folder,
    clf_model,
    loglevel = loglevel,
    valid_mask = valid_mask,
    estimate_uncertainties = estimate_uncertainties,
    uncertainty_params_dict = uncertainty_params_dict
)
"""

glass.classify_image_from_feature_folder(
    feat_folder,
    result_folder,
    clf_model,
    loglevel = loglevel,
    valid_mask = valid_mask,
    estimate_uncertainties = estimate_uncertainties,
    uncertainty_params_dict = uncertainty_params_dict,
    overwrite = overwrite
)


from osgeo import gdal
import matplotlib.pyplot as plt

labels = gdal.Open('tests/test_data/results/S1A_EW_GRDM_1SDH_20220503T082621_20220503T082725_043044_0523D1_AF89_subsampled_labels.img').ReadAsArray()
mahal = gdal.Open('tests/test_data/results/S1A_EW_GRDM_1SDH_20220503T082621_20220503T082725_043044_0523D1_AF89_subsampled_mahal_uncertainty.img').ReadAsArray()
apost = gdal.Open('tests/test_data/results/S1A_EW_GRDM_1SDH_20220503T082621_20220503T082725_043044_0523D1_AF89_subsampled_apost_uncertainty.img').ReadAsArray()


fig, axes = plt.subplots(1,3,sharex=True,sharey=True)
axes = axes.ravel()
axes[0].imshow(labels)
axes[1].imshow(mahal)
axes[2].imshow(apost)
