Analysis of Car Crashes in the United States using Seaborn and Tensorflow  
========================================================================
 
This is the project for the DS110 Final project
By James Fourie, Shaurya Gupta, Jay Patel
-----------------------------------------------

Overview
--------
This project seeks to answer this question: What factors contribute to the frequency and severity
of car accidents in the United States, and can these be predicted or
mitigated through an analysis of historical accident data?

Code functionalities
--------------------

1) Deep Neural Network (DNN) using TensorFlow and Keras

The architecture of the network is flexible, allowing it to be fine-tuned and optimized according to the specific characteristics of the dataset at hand. To enhance performance and prevent overfitting, the network employs a combination of L1 and L2 regularization (LASSO and RIDGE), as well as dropout layers that randomly deactivate nodes during training. Batch normalization is incorporated to both stabilize and speed up the learning process. The learning rate is dynamically adjusted using a ReduceLROnPlateau strategy for improved convergence. Early stopping is another critical feature, terminating training when validation accuracy no longer shows any improvement, thus saving computational resources.

2) Random Forest Regressor

For the random forest classifier, car crash severity is predicted by the following variables: temperature, the presence of a stop sign, weather condition, and visibility in miles

The resulting mean squared error from the model was 0.231.

3) Density Heatmap

Included in the density_map.py file is a density heatmap of the continental US. It represents the quantity
of car crashes in every US county. For the counties with no data, there is no color filling them. The data was logarithmically scaled in order to highlight the differences between counties more.

This visualization was made using Plotly express, and is a dynamic visualization. The user has the ability to zoom in and out, and focus on specific regions. This visualization does not show any obvious regional differences in the quantity of car crashes occurring.
The only potential trend in this density map could be the slightly elevated number of car crashes in the midwest compared to other regions. However, this could be due to sampling bias.

**Licenses**
