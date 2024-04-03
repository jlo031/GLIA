import numpy as np

import src.LGIA_classifier.linear_gaussian_IA_classifier as LGIA

from sklearn.model_selection import train_test_split

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

# test gaussian clf

# ---------------- #

# define class mean and covariance for class A and B
A_mean  = np.array([-10,-15])
A_sigma = np.array([[2,0],[0,2]])
B_mean  = np.array([-14,-19])
B_sigma = np.array([[2,0],[0,2]])

# define number of samples per class
samples = 4000

# draw samples from each distribution
A_X = np.random.multivariate_normal(A_mean,A_sigma,samples)
B_X = np.random.multivariate_normal(B_mean,B_sigma,samples)

# create labels
A_y = np.ones(samples,)
B_y = 3*np.ones(samples,)

# stack samples and labels
X = np.vstack((A_X,B_X))
y = np.hstack((A_y,B_y))

# split training and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

# ---------------- #

# create Gaussian clf object
clf = LGIA.gaussian_clf()

# fit clf to training data
clf.fit(X_train, y_train)

# predict labels for test data
y_pred, p = clf.predict(X_test)

# get per-class accuracy
CA = LGIA.get_per_class_score(y_pred, y_test, average=False)

print(f'Per-class accuracy for test data set: Class A={np.round(CA[0],3)}, Class B={np.round(CA[1],3)}')

# ---------------- #

# create gausian_params_dict and write to pickle file
gaussian_params_dict = LGIA.make_gaussian_params_dict_from_clf_object(clf)
LGIA.write_classifier_dict_2_pickle('gaussian_test_clf.pickle', gaussian_params_dict)

# read gaussian_params_dict from pickle file and create new clf object
gaussian_params_dict_1 = LGIA.read_classifier_dict_from_pickle('gaussian_test_clf.pickle')
clf_1 = LGIA.make_gaussian_clf_object_from_params_dict(gaussian_params_dict_1)


# predict labels for test data
y_pred_1, p_1 = clf_1.predict(X_test)

# get per-class accuracy
CA_1 = LGIA.get_per_class_score(y_pred_1, y_test, average=False)
print(f'Per-class accuracy for test data set: Class A={np.round(CA_1[0],3)}, Class B={np.round(CA_1[1],3)}')

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

# test LGIA clf

# ---------------- #

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
    print(f'Projecting dimension {dimemension}')
    A_X_projected[:,dimemension] = A_X[:,dimemension] + (A_slope[dimemension]*(A_IA-IA_ref))
    B_X_projected[:,dimemension] = B_X[:,dimemension] + (B_slope[dimemension]*(B_IA-IA_ref))

# add IA as a third dimension
A_X_IA = np.hstack((A_X_projected, np.expand_dims(A_IA,1)))
B_X_IA = np.hstack((B_X_projected, np.expand_dims(B_IA,1)))

# stack samples and labels
X = np.vstack((A_X_IA,B_X_IA))
y = np.hstack((A_y,B_y))

# split training and test data
X_IA_train, X_IA_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

X_train  = X_IA_train[:,0:-1]
IA_train = X_IA_train[:,-1]
X_test   = X_IA_test[:,0:-1]
IA_test  = X_IA_test[:,-1]

# ---------------- #

# create LGIA clf object
clf = LGIA.LGIA_clf()

# fit clf to training data (last dimension of X_train should be IA)
clf.fit(X_train, y_train, IA_train)

# predict labels for test data
y_pred, p = clf.predict(X_test, IA_test)

# get per-class accuracy
CA = LGIA.get_per_class_score(y_pred, y_test, average=False)

print(f'Per-class accuracy for test data set: Class A={np.round(CA[0],3)}, Class B={np.round(CA[1],3)}')