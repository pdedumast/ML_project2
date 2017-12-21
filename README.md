# Project Road Segmentation

Collaborators:

- Mathilde Guillaumot 
- Priscille Guerrier de Dumast
- Thibaut Chamard

# Table of Contents

* [Introduction](#introduction)
* [Environment Setup](#setup)
* [Results](#results)

# <a name="introduction"></a>Introduction
This repo contains our work on the road segmentation project from the Machine Learning Course at EPFL. 
For this project task, we have been provided a set of satellite images acquired from GoogleMaps and their ground-truth images where each pixel is labeled as road or background. 
We were tasked to train a classifier to segment roads in these images, i.e. assigns a label `road=1, background=0` to each pixel.

# <a name="setup"></a>Environment Setup
In this repo, you can find the following architecture of files:

As mentioned in the report, we have worked both on a logistic regression and the training of a Convolutional Neural Network. 
You can find both py files to run and their relatives helper functions in the relative files "helpers.py" and "helpersCNN.py".
They need to be in the root folder from where the python file is ran. You also need to have the data folder located in the root folder to properly run the programs. 
Running the runCNN.py file will automatically create a csv file called "predictionsCNN.csv"

Running in python in the temrinal with command line: `python runCNN.py`

* Python packages used when running the CNN:

```python: version 3.6.1
 matplotlib : version 2.0.2
 numpy : version 1.12.1
 scipy : version 0.19.0
 Pillow : version 4.1.1
 tensorflow : version 1.4.0
 ```
 
 The time of computation is above 3 hours in this configuration on our machine with 8GB of RAM, avoid running other programs while training the CNN.

# <a name="results"></a>Results

![Picture](https://github.com/pdedumast/ML_project2/blob/master/results.png)

![Picture](https://github.com/pdedumast/ML_project2/blob/master/f1scores.png)

Above is the kind of final result we have when having trained the CNN for 10000 iterations in our basic setup. We also plot the curves representing evolution of F1-Score for both the training and validation sets in regard of the number of iterations.


Evaluation Metric:
 [https://www.kaggle.com/wiki/MeanFScore]
