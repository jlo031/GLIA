# ---- This is <test_gaussian_clf.py> ----

import numpy as np
from sklearn.model_selection import train_test_split

import os
from loguru import logger

import GLIA_classifier.gaussian_linear_IA_classifier as GLIA

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

# 1) Create two classes A and B with 2-D Gaussian distributions around a mean and diagonal covariance matrices
#    and split into training and test data set

# define class mean and covariance for class A and B
A_mean  = np.array([-10,-15])
A_sigma = np.array([[2,0],[0,2]])
B_mean  = np.array([-14,-19])
B_sigma = np.array([[2,0],[0,2]])

# define number of samples per class
samples = 4000

# draw random samples from each distribution
A_X = np.random.multivariate_normal(A_mean,A_sigma,samples)
B_X = np.random.multivariate_normal(B_mean,B_sigma,samples)

# create labels
A_index = 1
B_index = 3
A_y = A_index * np.ones(samples,)
B_y = B_index * np.ones(samples,)

# stack samples and labels
X = np.vstack((A_X,B_X))
y = np.hstack((A_y,B_y))

# split training and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.6)

# total number of samples and dimensions
n_train = X_train.shape[0]
n_test  = X_test.shape[0]
n_feat  = X_train.shape[1]

logger.info(f'Created a Gaussian data set with 2 classes.')
logger.info(f'training samples: {n_train}')
logger.info(f'test samples:     {n_test}')
logger.info(f'dimensions:       {n_feat}')

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

# 2) Train a Bayesian classifier with Gaussian PDF and save it to a pickle file

# create Gaussian clf object
clf = GLIA.gaussian_clf()

# fit clf to training data
clf.fit(X_train, y_train)

# make parameter dictionary from clf object
clf_params_dict = GLIA.make_clf_params_dict_from_gaussian_clf_object(clf)

# save parameter dictionary to disk
pickle_file = 'gaussian_test_clf.pickle'
GLIA.write_classifier_dict_2_pickle(pickle_file, clf_params_dict)

logger.info(f'Trained a gaussian_clf and wrote parameter dictionary to disk.')
logger.info(f'Class A mean: {A_mean}; clf.mu: {clf.mu[A_index-1,:]}')
logger.info(f'Class B mean: {B_mean}; clf.mu: {clf.mu[B_index-1,:]}')

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

# 3) Load classifier from pickle file and use both classifiers to predict test labels

# read gaussian_params_dict from pickle file and create new clf object
clf_params_dict_1 = GLIA.read_classifier_dict_from_pickle(pickle_file)
clf_1 = GLIA.make_gaussian_clf_object_from_params_dict(clf_params_dict_1)

# predict labels for test data
y_pred, p = clf.predict(X_test)
y_pred_1, p_1 = clf_1.predict(X_test)

# get per-class accuracy
CA = 100 * GLIA.get_per_class_score(y_pred, y_test, average=False)
CA_1 = 100 * GLIA.get_per_class_score(y_pred_1, y_test, average=False)

precision = 2
logger.info('Predicted test labels from both clf instances.')
logger.info(f'Original clf: Class A={np.round(CA[0],precision)}, Class B={np.round(CA[1],precision)}')
logger.info(f'Reloaded clf: Class A={np.round(CA_1[0],precision)}, Class B={np.round(CA_1[1],precision)}')

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

# 5) Clean up and remove test pickle file

os.remove(pickle_file)

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

# ---- End of <test_gaussian_clf.py> ----
