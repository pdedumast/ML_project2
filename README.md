# Project Road Segmentation

Contributors:

- Mathilde Guillaumot
- Priscille Guerrier de Dumast
- Thibaut Chamard

# Table of Contents

* [Introduction](#introduction)
* [Content](#content)
* [Environment Setup](#setup)
* [Results](#results)

# <a name="introduction"></a>Introduction
This repo contains our work on the road segmentation project from the Machine Learning Course at EPFL.
For this project task, we have been provided a set of satellite images acquired from GoogleMaps and their ground-truth images where each pixel is labeled as road or background.
We were tasked to train a classifier to segment roads in these images, i.e. assigns a label `road=1`, `background=0` to each pixel.

# <a name="content"></a>Content

In this repo, you can find the following architecture of files:

`helpersCNN.py`: Contains the helper functions to run our CNN model, in `run.py`
`helpers_LogReg.py`: Contains the helper functions to run our Logistic Regression model
`run.py`: CNN giving the best score on Kaggle (model used for the final submission)
`run_LogReg_CV.py`: Cross validation used to identify the best parameters for the Logistic Regression model
`run_LogReg_eval.py`: Logistic Regression model giving the best score

# <a name="setup"></a>Environment Setup


### Architecture

As mentioned in the report, we have worked both on a logistic regression and the training of a Convolutional Neural Network.
You can find both python files to run and their relatives helper functions in the relative files in this archive. They need to be in the root folder from where the python file is ran.

You also need to create a `/data` folder in the root directory to properly run the programs in which you will paste the training and test sets (downloadable [here]( https://www.kaggle.com/c/epfml17-segmentation/data) ). Your architecture should be `/root_folder/data/training` and `/root_folder/data/test_set_images`.

Recap, root directory should look like:

```
helpersCNN.py
helpers_LogReg.py
run.py
run_LogReg_CV.py
run_LogReg_eval.py
data/training/
    /test_set_images/
```

### Settings

This project is developped in `Python 3.5` (See installation guidelines [here](https://www.python.org)).
* Python packages used when running our models:
```
matplotlib: version 2.0.2
numpy: version 1.12.1
scipy: version 0.19.0
pillow: version 4.1.1
tensorflow: version 1.4.0
scikit-learn: version 0.18.1
open-cv: version 3.3.0
```

### Program execution

* Main model : Convolutional Neural Network
From the root directory, you can train our CNN, as well as classifying using it by running the command:
`python run.py`
This will produce a csv file called `submissionCNN.csv` ready to be submitted on Kaggle. It is this one which will produce the final score on kaggle.

The time of computation is above 3 hours in the configuration of our CNN on our machine with 8GB of RAM, avoid running other programs while training the CNN.

* Logistic Regression
From the root directory, you can train our Logistic Regression model by running the command:
`python run_LogReg_eval.py`
This will produce two csv file for submission
  - `submission_LogReg.csv` which is the first classification issue by the logistic regression.
  - `submission_LogReg_pp.csv` which is the rectified prediction issue by the logistic regression, after running a postprocessing.


# <a name="results"></a>Results

Final score on Kaggle by TEAM CPE: 0.84628

![Picture](https://github.com/pdedumast/ML_project2/blob/master/display/results_CNN.png)

![Picture](https://github.com/pdedumast/ML_project2/blob/master/display/f1scores_CNN.png)

Above is the kind of final result we have when having trained the CNN for 10000 iterations in our basic setup. We also plot the curves representing evolution of F1-Score for both the training and validation sets in regard of the number of iterations.


Evaluation Metric:
 [https://www.kaggle.com/wiki/MeanFScore]
