# Supervised classification with Gaussian Linear Incidence Angle (GLIA) classifier

This library provides code for supervised classification of SAR imagery using the GLIA classifier, which accounts for linear per-class variation of backscatter intensity with incident angle. The concept was initially developed for sea ice classification in Sentinel-1 EW data but can be transferred to other sensors, imaging modes, and applications. Further explanation and theoretical background of classifer is given in:
- [Lohse et al (2020)]
- [Lohse et al (2021)]



### Preparation
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


### Installation

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


### Usage

Import the main classifier module like this:

    # import classifier module
    import GLIA_classifier.gaussian_linear_IA_classifier as glia

You can create classifier objects similar to the syntax of sklearn.  
For a simple Bayesian classifier with Gaussian PDF

    # create Bayesian classifier with Gaussian PDFs
    clf = glia.gaussian_clf()

    # train classifier
    clf.fit(X_train, y_train)

    # you can now check the class parameters
    print(f'clf mean values: \n{clf.mu}\n')
    print(f'clf covariance matrix: \n{clf.Sigma}\n')

    # predict new class labels and probabilities
    y_pred, prob = clf.predict(X_test)

The Bayesian classifier with linear IA depencency of the Gaussian mean valuecan be trained very similar, but requires explicit input of the IA.

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


### Saving classifiers

Classifier can be saved in any way that is convenient for you. However, the current implementation strongly encourages to save classifier parameters in the form of dictionaries and write these to pickle files.  
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



[GDAL]: https://gdal.org/
[Lohse et al (2020)]: https://www.researchgate.net/publication/342396165_Mapping_sea-ice_types_from_Sentinel-1_considering_the_surface-type_dependent_effect_of_incidence_angle
[Lohse et al (2021)]: https://www.researchgate.net/publication/349055291_Incident_Angle_Dependence_of_Sentinel-1_Texture_Features_for_Sea_Ice_Classification
