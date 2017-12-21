# Project Road Segmentation

- Mathilde Guillaumot 
- Priscille Guerrier de Dumast
- Thibaut Chamard

# Table of Contents

* [Introduction](#introduction)
* [Setup](#setup)
* [Results](#results)

# <a name="introduction"></a>Introduction
This repo contains our work on the road segmentation project from the Machine Learning Course at EPFL. 
For this project task, we have been provided a set of satellite images acquired from GoogleMaps and their ground-truth images where each pixel is labeled as road or background. 
We were tasked to train a classifier to segment roads in these images, i.e. assigns a label `road=1, background=0` to each pixel.

# <a name="setup"></a>Setup
In this repo, you can find the following architecture of files:
- 
-
-
-

As mentioned in the report, we have worked both on a logistic regression and the training of a Convolutional Neural Network. 
You can find both py files to run and their relatives helper functions in the relative files "helpers.py" and "helpersCNN.py".
They need to be in the root folder from where the python file is ran. You also need to have the data folder located in the root folder to properly run the programs. 
Running the runCNN.py file will automatically create a csv file called "predictionsCNN.csv"

Submission system environment setup:

1. The dataset is available from the Kaggle page, as linked in the PDF project description

2. Obtain the python notebook `segment_aerial_images.ipynb` from this github folder,
to see example code on how to extract the images as well as corresponding labels of each pixel.

The notebook shows how to use `scikit learn` to generate features from each pixel, and finally train a linear classifier to predict whether each pixel is road or background. Or you can use your own code as well. Our example code here also provides helper functions to visualize the images, labels and predictions. In particular, the two functions `mask_to_submission.py` and `submission_to_mask.py` help you to convert from the submission format to a visualization, and vice versa.

3. As a more advanced approach, try `tf_aerial_images.py`, which demonstrates the use of a basic convolutional neural network in TensorFlow for the same prediction task.

Evaluation Metric:
 [https://www.kaggle.com/wiki/MeanFScore]
