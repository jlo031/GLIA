"""
---- This is <linear_gaussian_IA_classifier.py> ----

Implementation of different supervised multi-dimensional Bayesian classifiers:
    -> gaussian_clf: Gaussian PDF with mean vector and covariance matrix per-class
    -> LGIA_clf: Gausian PDF with linearly variable mean vector (slope and intercept) and constant covariance matrix.
"""

from loguru import logger
import pickle

import numpy as np

from scipy.stats import multivariate_normal as mvn
from sklearn.linear_model import LinearRegression

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

class LGIA_clf:

    def __init__(self, IA_0=30, override_slopes=False):
        self.info = dict()
        self.info['type'] = 'LGIA'
        self.IA_0         = IA_0

# ---------------- #

    def add_info(self, info_dict):
        """Add info_dict to classifier object info

        Parameters
        ----------
        info_dict: dictionary to update classifier information
        """

        self.info.update(info_dict)

        return

# ---------------- #

    def fit(self, X_train, y_train, IA_train):
        """Fit classifier to input training data X_train with labels y_train and incidence angle IA_train

        Parameters
        ----------
        X_train: training data [N,d]
        y_train: training labels [N,]
        IA_train: training incidence angle [N,1] or [N,]

        Returns
        -------
        self.n_feat : number of features
        self.n_class : number of classes
        self.trained_classes : list of trained classes (in case y_train does not contain all classes indices)
        self.a : slope intercept at IA=0
        self.b : slope covariance matrix for each class [n_class,d,d]
        self.mu : class-dependent mean vector at IA_0 [n_class,d]
        self.Sigma : class-dependent covariance matrix at IA_0 [n_class,d,d]
        self.mvn : class-dependent multivariate normal distribution at IA_0
        """

        logger.warning('self.fit() is currently implemented only for balanced training distribution over the full IA range')
        logger.warning('Unbalanced training data distribution may result in erroneous slope estiamtes and poor classification results')

        # assert correct dimensionality of X_train
        assert len(X_train.shape) == 2, 'X_train must be of shape (N,d)'
    
        # assert correct dimensionality of IA_train
        assert X_train.shape[0] == IA_train.shape[0], 'X_train and IA_train must have same number of samples'

        # assert correct dimensionality of y_train
        assert X_train.shape[0] == y_train.shape[0], 'X_train and y_train must have same number of samples'

        # get number of training points and number of features
        N, self.n_feat = X_train.shape
        logger.debug(f'Number of training points: {N}')
        logger.debug(f'Number of features: {self.n_feat}')

        # find all classes in training data and set n_class to highest class index
        unique_classes = np.unique(y_train).astype(int)
        self.n_class = unique_classes.max()
        logger.debug(f'Unique class labels in training data: {unique_classes}')
        logger.debug(f'Highest class index in training data: {self.n_class}')

        # initialize slopes and intercepts for all classes and fill with nan
        self.a = np.full([self.n_class, self.n_feat], np.nan)
        self.b = np.full([self.n_class, self.n_feat], np.nan)

        # initialize projected training data X_projected
        X_projected = np.zeros(X_train.shape)

        # loop over classes in training data
        for i, cl in enumerate(unique_classes):

            logger.debug(f'Processing class {cl}')

            # loop over all dimensions
            for feat in range(self.n_feat):

                logger.debug(f'Estimating a and b for class {cl} and dimension {feat}')

                # fit a model and do the regression
                model = LinearRegression()
                model.fit(np.reshape(IA_train[y_train==cl],(-1,1)), np.reshape(X_train[y_train==cl,feat],(-1,1)))

                # extract intercept and slope
                self.a[cl-1,feat] = model.intercept_[0]
                self.b[cl-1,feat] = model.coef_[0][0]

                # project current dimension of X_train along slope b to IA_0
                X_projected[y_train==cl,feat] = X_train[y_train==cl,feat] - self.b[cl-1,feat] * (IA_train[y_train==cl]-self.IA_0)

        logger.debug('Estimated slope and intercept for all classes in training data')
        logger.debug('Projected X_train values to IA_0')


        # Slopes and intercepts for each class and dimension are found.
        # Training samples X_train are projected along class-dependent slopes to IA_0
        # Now estimate mu and Sigma at IA_0 exactly as it is done for Gausian

        # initialize means and covariance matrices for all classes and fill with nan
        self.mu    = np.full([self.n_class, self.n_feat], np.nan)
        self.Sigma = np.full([self.n_class, self.n_feat, self.n_feat], np.nan)

        # initialize multivariate_normal as dict
        self.class_mvn = dict()

        # initialize list of trained classes
        self.trained_classes = []

        # loop over classes in training data
        for cl in unique_classes:

            logger.debug(f'Estimating mu and Sigma for class {cl}')

            # add current class to list of trained classes
            self.trained_classes.append(cl)

            # select training for current class
            X_cl = X_projected[y_train==cl,:]

            logger.debug(f'Number of training points for current class: {X_cl.shape[0]}')
            logger.debug(f'Number of dimensions for current class: {X_cl.shape[1]}')

            # estimate mu and Sigma
            self.mu[cl-1,:]      = X_cl.mean(0)
            self.Sigma[cl-1,:,:] = np.cov(np.transpose(X_cl))

            # initialise multivariate_normal
            self.class_mvn[str(cl)] = mvn(self.mu[cl-1,:],self.Sigma[cl-1,:,:])

        return

# ---------------- #

    def predict(self, X_test, IA_test):
        """Predict class labels y_pred for input data X_test with IA_test

        Parameters
        ----------
        X_test : test data [N,d]
        IA_test : training incidence angle [N,1] or [N,]
    
        Returns
        -------
        y_pred : predicted class label [N,]
        p : probabilities [N,n_class]
        """

        # assert correct dimensionality of X_test
        assert len(X_test.shape) == 2, 'Test data must be of shape (N,d)'

        # assert correct dimensionality of IA_test
        assert X_test.shape[0] == IA_test.shape[0], 'X_test and IA_test must have same number of samples'

        # get number of test points and dimensionality
        N_test, d_test = X_test.shape
        logger.debug(f'Number of test points: {N_test}')
        logger.debug(f'Number of dimensions: {d_test}')

        # assert correct number of features
        assert d_test == self.n_feat, f'Classifier is trained with {self.n_feat} features but X_test has {d_test} features'

        # initialize labels and probabilities and fill with nan
        p      = np.full([N_test, self.n_class], np.nan)
        y_pred = np.full([N_test], np.nan)

        # loop over trained classes and estimate p from multivariate_normal
        for cl in (self.trained_classes):
            logger.debug(f'Working on class {cl}')
            logger.debug('Projecting data according to current class slopes.')

            # initialize projected X_test_projected
            X_test_projected = np.zeros(X_test.shape)

            # correct X according to class-dependent slope
            for feat in range(self.n_feat):
                logger.debug(f'Projecting current class along feature dimension {feat}')
                X_test_projected[:,feat] = X_test[:,feat] - self.b[cl-1,feat] * (IA_test-self.IA_0)

            # estimate p from multivariate_normal on projected data
            logger.debug(f'Calculating p for class {cl}')
            p[:,cl-1] = self.class_mvn[str(cl)].pdf(X_test_projected)

        # find maximum p and set labels
        y_pred  = np.nanargmax(p,1) + 1

        return y_pred, p

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

class gaussian_clf:

    def __init__(self):
        self.info = dict()
        self.info['type'] = 'gaussian'

# ---------------- #

    def add_info(self, info_dict):
        """Add info_dict to classifier object info

        Parameters
        ----------
        info_dict: dictionary to update classifier information
        """

        self.info.update(info_dict)

        return

# ---------------- #

    def fit(self, X_train, y_train):
        """Fit classifier to input training data X_train with labels y_train

        Parameters
        ----------
        X_train : training data [N,d]
        y_train : training labels [N,1] or [N,]

        Returns
        -------
        self.n_feat : number of features
        self.n_class : number of classes
        self.trained_classes : list of trained classes (in case y_train does not contain all classes indices)
        self.mu : class-dependent covariance matrix [n_class,d]
        self.Sigma : covariance matrix for each class [n_class,d,d]
        self.mvn : class-dependent multivariate normal distribution
        """

        # assert correct dimensionality of X_train
        assert len(X_train.shape) == 2, 'X_train must be of shape (N,d)'

        # assert correct dimensionality of y_train
        assert X_train.shape[0] == y_train.shape[0], 'X_train and y_train must have same number of samples'

        # get number of training points and number of features
        N, self.n_feat = X_train.shape
        logger.debug(f'Number of training points: {N}')
        logger.debug(f'Number of features: {self.n_feat}')

        # find all classes in training data and set n_class to highest class index
        unique_classes = np.unique(y_train).astype(int)
        self.n_class = unique_classes.max()
        logger.debug(f'Unique class labels in training data: {unique_classes}')
        logger.debug(f'Highest class index in training data: {self.n_class}')

        # initialize means and covariance matrices for all classes and fill with nan
        self.mu    = np.full([self.n_class,self.n_feat], np.nan)
        self.Sigma = np.full([self.n_class,self.n_feat,self.n_feat], np.nan)

        # initialize multivariate_normal as dict
        self.class_mvn = dict()

        # initialize list of trained classes
        self.trained_classes = []

        # loop over classes in training data
        for cl in unique_classes:

            logger.debug(f'Estimating mu and Sigma for class {cl}')

            # add current class to list of trained classes
            self.trained_classes.append(cl)

            # select training for current class
            X_cl = X_train[y_train==cl,:]

            logger.debug(f'Number of training points for current class: {X_cl.shape[0]}')
            logger.debug(f'Number of dimensions for current class: {X_cl.shape[1]}')

            # estimate mu and Sigma
            self.mu[cl-1,:]      = X_cl.mean(0)
            self.Sigma[cl-1,:,:] = np.cov(np.transpose(X_cl))

            # initialise multivariate_normal
            self.class_mvn[str(cl)] = mvn(self.mu[cl-1,:],self.Sigma[cl-1,:,:])

        return

# ---------------- #

    def predict(self, X_test):
        """Predict class labels y_pred for input data X_test.

        Parameters
        ----------
        X_test : test data [N,d]

        Returns
        -------
        y_pred : predicted class label [N,]
        p : probabilities [N,n_class]
        """

        # assert correct dimensionality of X_test
        assert len(X_test.shape) == 2, 'Test data must be of shape (N,d).'

        # get number of test points and dimensionality
        N_test, d_test = X_test.shape
        logger.debug(f'Number of test points: {N_test}')
        logger.debug(f'Number of dimensions: {d_test}')

        # assert correct number of features
        assert d_test == self.n_feat, f'Classifier is trained with {self.n_feat} features but X_test has {d_test} features'

        # initialize labels and probabilities and fill with nan
        p      = np.full([N_test, self.n_class], np.nan)
        y_pred = np.full([N_test], np.nan)

        # loop over trained classes and estimate p from multivariate_normal
        for cl in (self.trained_classes):
            logger.debug(f'Calculating p for class {cl}')
            p[:,cl-1] = self.class_mvn[str(cl)].pdf(X_test)

        # find maximum p and set labels
        y_pred  = np.nanargmax(p,1) + 1

        return y_pred, p

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

def get_per_class_score(y_true, y_pred, average=True, subsample=1):
    """Calculate (average) per class classification accuracy

    Parameters
    ----------
    y_true : True class labels [N,]
    y_pred : Predicted class labels [N]
    average : True/False for average or individual per-class CA (default True)
    subsample : Subsample labels (default=1)

    Returns
    -------
    CA: Classification accuracy (per class or average per class)
    """

    # convert labels to array
    y_true = np.array(y_true)[::subsample]
    y_pred = np.array(y_pred)[::subsample]

    assert y_true.shape == y_pred.shape , 'true labels and predicted labels must have the same shape.'

    classes = set(y_true)
    CA = np.array(list(np.equal(y_true[y_true==label], y_pred[y_true==label]).sum() / float(sum(y_true==label)) for label in classes))

    if average:
        CA = np.mean(CA)

    return CA

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

def write_classifier_dict_2_pickle(output_file, classifier_dict):
    """Write classifier dictionary to pickle file

    Parameters
    ----------
    output_file : pickle output file
    classifier_dict : classifier dictionary
    """

    with open(output_file, 'wb') as f: 
        pickle.dump(classifier_dict, f)

    return

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

def read_classifier_dict_from_pickle(input_file):
    """Read classifier dictionary from pickle file

    Parameters
    ----------
    input_file : pickle input file

    Returns
    -------
    classifier_dict: classifier dictionary
    """

    with open(input_file, 'rb') as f: 
        classifier_dictionary = pickle.load(f)

    return classifier_dictionary

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

def make_LGIA_clf_object_from_params_dict(LGIA_params_dict):
    """Make an LGIA_classifier object from parameters in input dict

    Parameters
    ----------
    LGIA_params_dict : dictionary with with classifier parameters

    Returns
    -------
    clf : gaussian_clf classifier object

    :param  gaussian_IA_params_dict: dictionary with with classifier parameters
    :return clf: classifier object (gaussian_IA_clf)
     """

    # initialize classifier object
    clf = LGIA_clf()

    # set classifier parameters from input dict
    clf.a               = LGIA_params_dict['a']
    clf.b               = LGIA_params_dict['b']
    clf.mu              = LGIA_params_dict['mu']
    clf.Sigma           = LGIA_params_dict['Sigma']
    clf.IA_0            = LGIA_params_dict['IA_0']
    clf.n_class         = LGIA_params_dict['n_class']
    clf.n_feat          = LGIA_params_dict['n_feat']
    clf.trained_classes = LGIA_params_dict['trained_classes']

    # define the multivariate_normal for each class
    clf.class_mvn = dict()
    for cl in clf.trained_classes:
        clf.class_mvn[str(cl)] = mvn(clf.mu[cl-1,:],clf.Sigma[cl-1,:,:])

    return clf

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

def make_LGIA_params_dict_from_clf_object(clf):
    """Make an LGIA_params dict from a LGIA_classifier object

    Parameters
    ----------
    clf : LGIA_clf classifier object

    Returns
    -------
    LGIA_params_dict : dictionary with with classifier parameters
     """

    # initialize dict for clf parameters
    LGIA_params_dict = dict()

    # fill in dict with clf parameters
    LGIA_params_dict['a']               = clf.a
    LGIA_params_dict['b']               = clf.b
    LGIA_params_dict['mu']              = clf.mu
    LGIA_params_dict['Sigma']           = clf.Sigma
    LGIA_params_dict['IA_0']            = clf.IA_0
    LGIA_params_dict['n_class']         = clf.n_class
    LGIA_params_dict['n_feat']          = clf.n_feat
    LGIA_params_dict['trained_classes'] = clf.trained_classes

    return LGIA_params_dict

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

def make_gaussian_params_dict_from_clf_object(clf):
    """Make a gaussian_params dict from a gaussian_classifier object

    Parameters
    ----------
    clf : gaussian_clf classifier object

    Returns
    -------
    gaussian_params_dict : dictionary with with classifier parameters
     """

    # initialize dict for clf parameters
    gaussian_params_dict = dict()

    # fill in dict with clf parameters
    gaussian_params_dict['mu']              = clf.mu
    gaussian_params_dict['Sigma']           = clf.Sigma
    gaussian_params_dict['n_class']         = clf.n_class
    gaussian_params_dict['n_feat']          = clf.n_feat
    gaussian_params_dict['trained_classes'] = clf.trained_classes

    return gaussian_params_dict

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

def make_gaussian_clf_object_from_params_dict(gaussian_params_dict):
    """Make a gaussian_classifier object from parameters in input dict

    Parameters
    ----------
    gaussian_params_dict : dictionary with with classifier parameters

    Returns
    -------
    clf : gaussian_clf classifier object
    """

    # initialize classifier object
    clf = gaussian_clf()

    # set classifier parameters from input dict
    clf.mu              = gaussian_params_dict['mu']
    clf.Sigma           = gaussian_params_dict['Sigma']
    clf.n_class         = gaussian_params_dict['n_class']
    clf.n_feat          = gaussian_params_dict['n_feat']
    clf.trained_classes = gaussian_params_dict['trained_classes']

    # define the multivariate_normal for each class
    clf.class_mvn = dict()
    for cl in clf.trained_classes:
        clf.class_mvn[str(cl)] = mvn(clf.mu[cl-1,:],clf.Sigma[cl-1,:,:])

    return clf

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

# ---- End of <linear_gaussian_IA_classifier.py> ----
