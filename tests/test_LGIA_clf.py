# ---- This is <test_LGIA_clf.py> ----

import numpy as np
from sklearn.model_selection import train_test_split

import os
from loguru import logger

import LGIA_classifier.linear_gaussian_IA_classifier as LGIA

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

# 1) Create two classes A and B with 2-D Gaussian distributions around a mean and diagonal covariance matrices
#    introduce linear variation with IA
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

# Assume that the distributions above are samples projected to a reference angle IA_ref=30
IA_ref = 30

# draw random IA samples for class A and B from uniform distribution
A_IA = np.random.uniform(18,45,samples)
B_IA = np.random.uniform(18,45,samples)

# define slopes for class A and B
A_slope = [-0.5, -0.2]
B_slope = [-0.2, -0.1]

# project normal distributed data points along the IA slopes
A_X_projected = np.zeros(A_X.shape)
B_X_projected = np.zeros(B_X.shape)

for dimemension in np.arange(A_X.shape[1]):
    logger.debug(f'Projecting dimension {dimemension}')
    A_X_projected[:,dimemension] = A_X[:,dimemension] + (A_slope[dimemension]*(A_IA-IA_ref))
    B_X_projected[:,dimemension] = B_X[:,dimemension] + (B_slope[dimemension]*(B_IA-IA_ref))

# add IA as a third dimension
A_X_IA = np.hstack((A_X_projected, np.expand_dims(A_IA,1)))
B_X_IA = np.hstack((B_X_projected, np.expand_dims(B_IA,1)))

# stack samples and labels
X = np.vstack((A_X_IA,B_X_IA))
y = np.hstack((A_y,B_y))

# split training and test data and separate IA again
X_IA_train, X_IA_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

X_train  = X_IA_train[:,0:-1]
IA_train = X_IA_train[:,-1]
X_test   = X_IA_test[:,0:-1]
IA_test  = X_IA_test[:,-1]

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

# 2) Train a Bayesian classifier with LGIA PDF and save it to a pickle file

# create LGIA clf object
clf = LGIA.LGIA_clf()

# fit clf to training data
clf.fit(X_train, y_train, IA_train)

# make parameter dictionary from clf object
LGIA_params_dict = LGIA.make_LGIA_params_dict_from_clf_object(clf)

# save parameter dictionary to disk
pickle_file = 'LGIA_test_clf.pickle'
LGIA.write_classifier_dict_2_pickle(pickle_file, LGIA_params_dict)

logger.info(f'Trained a LGIA_clf and wrote parameter dictionary to disk.')
logger.info(f'Class A mean: {A_mean}; clf.mu at IA={clf.IA_0}: {clf.mu[A_index-1,:]}')
logger.info(f'Class B mean: {B_mean}; clf.mu at IA={clf.IA_0}: {clf.mu[B_index-1,:]}')
logger.info(f'Class A slope: {A_slope}; clf.mu: {clf.b[A_index-1,:]}')
logger.info(f'Class B slope: {B_slope}; clf.mu: {clf.b[B_index-1,:]}')

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

# 3) Load classifier from pickle file and use both classifiers to predict test labels

# read LGIA_params_dict from pickle file and create new clf object
LGIA_params_dict_1 = LGIA.read_classifier_dict_from_pickle(pickle_file)
clf_1 = LGIA.make_LGIA_clf_object_from_params_dict(LGIA_params_dict_1)

# predict labels for test data
y_pred, p = clf.predict(X_test, IA_test)
y_pred_1, p_1 = clf_1.predict(X_test, IA_test)

# get per-class accuracy
CA = 100 * LGIA.get_per_class_score(y_pred, y_test, average=False)
CA_1 = 100 * LGIA.get_per_class_score(y_pred_1, y_test, average=False)

precision = 3
logger.info('Predicted test labels from both clf instances.')
logger.info(f'Original clf: Class A={np.round(CA[0],precision)}, Class B={np.round(CA[1],precision)}')
logger.info(f'Reloaded clf: Class A={np.round(CA_1[0],precision)}, Class B={np.round(CA_1[1],precision)}')

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

# 4) Train a gaussian clf to demonstrate the advantage of including IA dependence

clf_gaussian = LGIA.gaussian_clf()
clf_gaussian.fit(X_train, y_train)
y_pred_gaussian, p_gaussian = clf_gaussian.predict(X_test)
CA_gaussian = 100 * LGIA.get_per_class_score(y_pred_gaussian, y_test, average=False)

logger.info('Trained gaussian_clf without IA for comparison.')
logger.info(f'Gaussian clf: Class A={np.round(CA_gaussian[0],precision)}, Class B={np.round(CA_gaussian[1],precision)}')

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

# 5) Clean up and remove test pickle file

os.remove(pickle_file)

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

# ---- End of <test_LGIA_clf.py> ----
