# DNN Models for Chemical Kinetics
This repository contains Deep Neural Network (DNN) models for predicting chemical kinetics in hydrogen/air combustion systems, as detailed in our peer-reviewed publication.

## Overview
- Implementation available in both Python and Fortran
- Includes test cases for auto-ignition predictions
- Features a multi-scale sampling approach for robust and accurate predictions
## Contents
- Chem/: Chemical mechanism files
- Model/: DNN model implementations
- FortranDNN_Module.F: Fortran implementation of DNN
- ZeroDAutognition.py: Python code for zero-dimensional auto-ignition tests
- modelclass.py: Python model definitions
- useFortranDNN.F: Example usage of Fortran implementation
- DME Dataset: Training and validation datasets, consisting of 3.5 million samples generated from the penta-brachial flame (CVODE solver). The penta-brachial flame configuration provides access to warm flame states inaccessible through auto-ignition, ensuring coverage of all flame branches in phase space.
  - dataTPY_DME_X.npy: A 3,500,000 × 41 matrix, [T, P, Y_1, ..., Y_39]
  - dataTPY_DME_Y.npy: A 3,500,000 × 41 matrix, evolved Δt=1e-6s using Cantera

## Reference
[1] Zhang, T., Yi, Y., Xu, Z.J., et al. "A multi-scale sampling for accurate and robust deep neural network to predict combustion chemical kinetics," Combustion and Flame, 245, 2022.
https://doi.org/10.1016/j.combustflame.2022.112319

[2] Wang, T., Yi, Y., Yao, J., Xu, Z.J., Zhang, T., Chen, Z., "Enforcing Physical Conservation in Neural Network Surrogate Models for Complex Chemical Kinetics," Combustion and Flame, under review.

## License
Apache-2.0
