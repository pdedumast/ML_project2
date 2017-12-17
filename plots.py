# -*- coding: utf-8 -*-
"""a function of ploting figures."""
import numpy as np
import matplotlib.pyplot as plt


def cross_validation_visualization_f1score(lambds, f1score_train, f1score_test, f1score_test_pp):
    """visualization the curves of f1score_train, f1score_test and f1score_test_pp."""
    plt.semilogx(lambds, f1score_train, marker=".", color='b', label='F1score train')
    plt.semilogx(lambds, f1score_test, marker=".", color='r', label='F1score test')
    plt.semilogx(lambds, f1score_test_pp, marker=".", color='g', label='F1score test PP')
    plt.xlabel("lambda")
    plt.ylabel("F1-score")
    plt.title("Cross Validation")
    plt.legend(loc=2)
    plt.grid(True)
    plt.savefig("cross_validation_f1score")

    
def cross_validation_visualization_rmse(lambds, rmse_train, rmse_test):
    """visualization the curves of rmse_train, rmse_test and rmse_test_pp."""
    plt.semilogx(lambds, rmse_train, marker=".", color='b', label='RMSE train')
    plt.semilogx(lambds, rmse_test, marker=".", color='r', label='RMSE test')
    plt.xlabel("lambda")
    plt.ylabel("RMSE")
    plt.title("Cross Validation")
    plt.legend(loc=2)
    plt.grid(True)
    plt.savefig("cross_validation_rmse")

