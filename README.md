# D3T4SOC
Summary: 

The software generates synthetic Soil Organic Carbon (SOC) profiles using a generative AI-based conditional Wasserstein generative adversarial network, preserving statistical characteristics of real observations to address data scarcity and reduce uncertainty in SOC stock estimates. 

Software Detail:

1. Practical Application

Given longitude/latitude (and optional covariates such as land-cover type), the software generates synthetic geospatial samples and supports prediction/augmentation of soil organic carbon (SOC) stock for grasslands, croplands, and boreal forests. It is used to augment sparse observational datasets, support downstream modeling/analysis, and study spatial heterogeneity.

2. Method of Solution
   
The software uses conditional Wasserstein GANs (cWGANs) to generate synthetic samples conditioned on location (latitude/longitude), land-cover class, and other covariates. Training uses mini-batch stochastic optimization with backpropagation under a Wasserstein objective with gradient-penalty regularization. Data preprocessing includes normalization and handling missing values. Unsupervised clustering (e.g., kkk-means / GMM) is used to represent spatial heterogeneity via regime/group identification. Post-generation statistical checks evaluate fidelity between real and synthetic data (e.g., marginals/joint distributions, correlations, and summary statistics) to verify that key characteristics are preserved. The workflow includes model calibration/hyperparameter tuning and automated quality-control metrics for synthetic-data validity. 
