# Supervised classification with Gaussian Linear Incidence Angle (GLIA) classifier

This library provides code for supervised classification of SAR imagery using the LGIA classifier, which accounts for linear per-class variation of backscatter intensity with incident angle. The concept was initially developed for sea ice classification in Sentinel-1 EW data but can be transferred to other sensors and applications. Further explanation and theoretical background of classifer is given in:
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
    conda install -y ipython scipy loguru scikit-learn


### Installation

Clone the repository:

    # clone the repository
    git clone git@github.com:jlo031/glia.git

Change into the main directory of the cloned repository (it should contain the '_setup.py_' file) and install the library:

    # installation
    pip install .










[GDAL]: https://gdal.org/
[Lohse et al (2020)]: https://www.researchgate.net/publication/342396165_Mapping_sea-ice_types_from_Sentinel-1_considering_the_surface-type_dependent_effect_of_incidence_angle
[Lohse et al (2021)]: https://www.researchgate.net/publication/349055291_Incident_Angle_Dependence_of_Sentinel-1_Texture_Features_for_Sea_Ice_Classification
