"""Performs regression on the drug data and cell line factors"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
from xgboost import XGBRegressor


def xgbPred(xTrain, yTrain, xTest):
    '''
    Makes a prediction after fitting the model to the training data

    Inputs: 2D Numpy Array, 1D Numpy Array, 2D Numpy Array

    Outputs: 1D Numpy Array
    '''

    xgb_model = XGBRegressor(objective='reg:squarederror', n_estimators=1000, seed=24,
                             subsample=0.1, colsample_bytree=0.1)
    xgb_model.fit(X_train, y_train)
    yPred = xgb_model.predict(xTest)
    return yPred

def xgbPlot(xTrain, yTrain, xTest, yTest, title, tree=False):#could potentially generalize to all regressions
    '''
    Plots and saves the Predicted vs. Actual plot and the tree plot if desired

    Inputs: Numpy arrays for all data, string for title, boolean for tree

    Outputs: Saves plots for the xgBoost
    '''
    xgb_model = XGBRegressor(objective='reg:squarederror', n_estimators=1000, seed=24,
                             subsample=0.1, colsample_bytree=0.1)
    xgb_model.fit(X_train, y_train)
    yPred = xgb_model.predict(xTest)

    plt.figure()
    plt.title(title)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.scatter(y_test, y_pred)
    plt.savefig('Predicted vs. Actual.png', dpi=500)

    if tree:
        plt.figure
        xgb.plot_tree(xgb_model)
        plt.savefig('tree.png', dpi=1000)

def errMetric(y_test, y_pred):
    '''
    Determines error values based off of predicted and actual values
    Inputs: 1D Numpy Arrays
    Outputs: rmse and r2 as Float64
    '''
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    return rmse, r2
