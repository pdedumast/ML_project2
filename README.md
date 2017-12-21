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
```
helpersCNN.py: helpers needed to run the runCNN.py file
helpers_LogReg.py: helpers needed to run the run_LogReg_CV.py and run_LogReg_eval.py files
run.py: run used to perform best score on Kaggle by training the CNN
run_LogReg_CV.py: /// A COMPLETER ///
run_LogReg_eval.py: /// A COMPLETER ///
README.md: read me file
```

As mentioned in the report, we have worked both on a logistic regression and the training of a Convolutional Neural Network. 
You can find both py files to run and their relatives helper functions in the relative files in this archive.
They need to be in the root folder from where the python file is ran. You also need to create a `data/` folder in the root folder to properly run the programs in which you will paste the training and test sets (download at https://www.kaggle.com/c/epfml17-segmentation/data). Your architecture should be `root_folder/data/training/` and `root_folder/data/test_set_images`.
Recap, folder should look like:
```
helpersCNN.py
helpers_LogReg.py
run.py
run_LogReg_CV.py
run_LogReg_eval.py
data/ : - training/
        - test_set_images/
```
Running the `run.py` file will automatically create a csv file called `submissionCNN.csv` ready for submission on Kaggle, it is this one which will produce the final score on kaggle.
Running the `run_LogReg_eval.py` file will automatically create a csv file called `submissionLogReg.csv` /// A VERIFIER ///

Simply execute in the terminal the following command line: `python run.py`

Python installation guidelines at https://www.python.org and install packages using pip https://docs.python.org/3.5/installing/
* Python packages used when running the CNN:

```
python: version 3.6.1
matplotlib: version 2.0.2
numpy: version 1.12.1
scipy: version 0.19.0
pillow: version 4.1.1
tensorflow: version 1.4.0
sklearn: version 0.18.1
```
 
The time of computation is above 3 hours in the configuration of our CNN on our machine with 8GB of RAM, avoid running other programs while training the CNN.

# <a name="results"></a>Results

![Picture](https://github.com/pdedumast/ML_project2/blob/master/display/results_CNN.png)

![Picture](https://github.com/pdedumast/ML_project2/blob/master/display/f1scores_CNN.png)

Above is the kind of final result we have when having trained the CNN for 10000 iterations in our basic setup. We also plot the curves representing evolution of F1-Score for both the training and validation sets in regard of the number of iterations.


Evaluation Metric:
 [https://www.kaggle.com/wiki/MeanFScore]
