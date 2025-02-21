# Supervised classification with Gaussian Linear Incidence Angle (GLIA) classifier

This library provides code for supervised classification of SAR imagery using the GLIA classifier, which accounts for linear per-class variation of backscatter intensity with incident angle. The concept was initially developed for sea ice classification in Sentinel-1 EW data but can be transferred to other sensors, imaging modes, and applications. Further explanation and theoretical background of classifer is given in:
- [Lohse et al (2020)]
- [Lohse et al (2021)]



## 1) Preparation
The Geospatial Data Abstraction Layer ([GDAL]) library is required to run the code.  
The simplest way to use GDAL with Python is to get the Anaconda Python distribution.  
It is recommended to run the code in a virtual environment.

    # create new conda environment
    conda create -y -n GLIA gdal
    
    # activate environment
    conda activate GLIA
    
    # install required packages
    conda install -y scipy loguru scikit-learn
    pip install ipython


## 2) Installation

You can install this library directly from github (1) or locally after cloning (2).  
For both installation options, first set up the environment as described above.

1. **Installation from github**

       # install this package
       pip install git+https://github.com/jlo031/GLIA

2. **Local installation**

       # clone the repository
       git clone git@github.com:jlo031/glia.git

   Change into the main directory of the cloned repository (it should contain the *setup.py* file) and install the library:

       # installation
       pip install .

You can check the succesful installation by running the scripts provided in the *tests* folder.


## 3) Usage

#### 3.1) Run a pre-trained classifier with the the GLIA_classifier.classification module

If you already have a trained classifier model or want to run one of the models provided in this library, you can perform a classification from a feature folder. It is up to the user to extract the features and name them correctly.

    import GLIA_classifier.classification as glass
    feature_folder = '/path/to/folder/with/extracted/features'
    results_folder = '/path/to/folder/with/results'
    clf_model_path = '/path/to/classifier/pickle/file'
    glass.classify_image_from_feature_folder(
        feature_folder,
        result_folder,
        clf_model_path,
        use_valid_mask=True,
        estimate_uncertainties=False,
        uncertainty_params_dict=[],
        overwrite=False,
        loglevel='INFO',
    )

The features must be stored in *ENVI* format (e.g. *Sigma0_HH_db.img* and *Sigma0_HH_db.hdr*). The code will check that the pickle file contains a valid classifier dictionary with all necessary parameters and that the required features exist in the feature_folder. If using the valid mask (*use_valid_mask=True*), there must be a file called *valid.img* in the featur folder. Only pixels with the valid mask = 1 will be classified.  
An example of how to perform batch classification of multiple images is provided in *examples/batch_classification.py*.

#### 3.2) Use the GLIA.gaussian_linear_IA_classifier module

To train a new classifier or perform other tasks that require additional manipulation, you need to work directly with the GLIA.gaussian_linear_IA_classifier module.

    # import classifier module
    import GLIA_classifier.gaussian_linear_IA_classifier as glia

You can create classifier objects similar to the syntax of sklearn.  
The library offers a traditional Bayesian classifier with a multi-variate Gaussian PDF, or the GLIA classifer with the linearly variable mean value.
Both work similar, but note that the GLIA classifier requires explicit input of the incidence angle.

Traditional Gaussian PDF classifer:

    # create Bayesian classifier with Gaussian PDFs
    clf = glia.gaussian_clf()

    # train classifier
    clf.fit(X_train, y_train)

    # you can now check the class parameters
    print(f'clf mean values: \n{clf.mu}\n')
    print(f'clf covariance matrix: \n{clf.Sigma}\n')

    # predict new class labels and probabilities
    y_pred, prob = clf.predict(X_test)

GLIA classifer:

    # create Bayesian classifier with Gaussian PDFs
    clf = glia.GLIA_clf()

    # train classifier
    clf.fit(X_train, y_train, IA_train)

    # you can now check the class parameters
    print(f'clf intercept: \n{clf.a}\n')
    print(f'clf slope: \n{clf.b}\n')
    print(f'clf covariance matrix: \n{clf.Sigma}\n')

    # predict new class labels and probabilities
    y_pred, prob = clf.predict(X_test, IA_test)


## 4) Reading and saving classifiers

Classifier can be read saved in any way that is convenient for you. However, the current implementation strongly encourages to save classifier parameters in the form of dictionaries and write these to pickle files.  
The *gaussian_linear_IA_classifier.py* module provides routines to read and write these pickle files, convert the dictionaries to clf objects, or create a dictionary from an clf object. It also offers an easy routint for inspection of the pickle files.

    # import module and define path to clf pickle file
    import GLIA_classifier.gaussian_linear_IA_classifier as glia
    clf_pickle_path = 'src/GLIA_classifier/clf_models/belgica_bank_ice_types_2022.pickle'

    # inspect the pickle file
    glia.inspect_classifier_pickle_file(clf_pickle_path)

To use this classifier, load the dictionary and then create the the clf object like this:

    # read dictionary into clf_params_dict
    clf_params_dict = glia.read_classifier_dict_from_pickle(clf_pickle_path)

    # create clf object from that dictionary
    clf = glia.make_GLIA_clf_object_from_clf_params_dict(clf_params_dict)

To save a classifier, you first need to convert the classifier object into a dictionary. Then write the dictionary to a pickle file.

    # create clf_params_dict from clf object
    clf_params_dict = glia.make_clf_params_dict_from_GLIA_clf_object(clf)
    
    # save clf_params_dict to  pickle file
    glia.write_classifier_dict_2_pickle(output_file, clf_params_dict)

Example scripts to test the classifier on simulated data and for reading/writing from/to pickle files are given in *tests/test_GLIA_clf.py* and *tests/test_gaussian_clf.py*


## 5) Training new classifier models

To train a new classifier model, you can simply load all your trainind data into python, create a new clf object, and run the clf.fit() method.

    # read your training data in X_train, y_train, IA_train

    # create new clf object
    import GLIA_classifier.gaussian_linear_IA_classifier as glia
    clf = glia.GLIA_clf()

    # train the new classifier
    clf.fit(X_train, y_train, IA_train)

    # save the trained classifier to pickle file 
    clf_params_dict = glia.make_clf_params_dict_from_GLIA_clf_object(clf)
    glia.write_classifier_dict_2_pickle(output_file, clf_params_dict)

Note that the clf.fit() method performs a simple linear regression through all provided training samples per class.
In some cases, e.g. when the training data is strongly affected by thermal noise or not equally distributed over incidence angle ranges, it will be better to estimnate the classifier parameters more carefully.
You can then manually create a classifier dictionary and input the required parameters that you have determined externally.


[GDAL]: https://gdal.org/
[Lohse et al (2020)]: https://www.researchgate.net/publication/342396165_Mapping_sea-ice_types_from_Sentinel-1_considering_the_surface-type_dependent_effect_of_incidence_angle
[Lohse et al (2021)]: https://www.researchgate.net/publication/349055291_Incident_Angle_Dependence_of_Sentinel-1_Texture_Features_for_Sea_Ice_Classification
