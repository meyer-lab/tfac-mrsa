"""Performs regression on the drug data and cell line factors"""
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor


sns.set()


def errorMetrics(y_test, y_pred):
    '''
    Determines error values based off of predicted and actual values
    Inputs: 1D Numpy Arrays
    Outputs: Prints rmse and r2  and returns them as Float64
    '''
    weightedRes = (y_pred - y_test) / y_test
    absError = abs(weightedRes) * 100
    sqError = (weightedRes**2) * 100

    mape = np.round(np.mean(absError), 2)
    mspe = np.round(np.mean(sqError), 2)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print('Model Performance')
    print('Root Mean Squared Error: {:0.4f}'.format(rmse))
    print('Mean Absolute Error: {:0.4f}'.format(mae))
    print('Accuracy (MSPE): {:0.2f}'.format(100 - mspe))
    print('Accuracy (MAPE): {:0.2f}'.format(100 - mape))
    print('R2 Score: {:0.4f}'.format(r2))
    metrics = np.array([rmse, mae, mspe, mape, r2])
    return metrics

# Decision Tree Model


def dTreePred(xTrain, yTrain, xTest, yTest):
    '''
    Makes a prediction after fitting the model to the training data

    Inputs: 2D Numpy Array, 1D Numpy Array, 2D Numpy Array, 1D Numpy Array

    Outputs: 1D Numpy Array, 1D Numpy Array
    '''

    dTree = DecisionTreeRegressor(random_state=42)
    dTree.fit(xTrain, yTrain)
    yPred = dTree.predict(xTest)
    metrics = errorMetrics(yTest, yPred)
    return yPred, metrics

# XGBoost Model


def xgbPred(xTrain, yTrain, xTest, yTest):
    '''
    Makes a prediction after fitting the model to the training data

    Inputs: 2D Numpy Array, 1D Numpy Array, 2D Numpy Array, 1D Numpy Array

    Outputs: 1D Numpy Array, 1D Numpy Array
    '''

    xgb_model = XGBRegressor(objective='reg:squarederror', n_estimators=1000, seed=24,
                             subsample=0.1, colsample_bytree=0.1)
    xgb_model.fit(xTrain, yTrain)
    yPred = xgb_model.predict(xTest)
    metrics = errorMetrics(yTest, yPred)
    return yPred, metrics


def xgbPlot(xTrain, yTrain, xTest, yTest, title, tree=False):  # could potentially generalize to all regressions
    '''
    Plots and saves the Predicted vs. Actual plot and the tree plot if desired

    Inputs: Numpy arrays for all data, string for title, boolean for tree

    Outputs: Saves plots for the xgBoost
    '''
    xgb_model = XGBRegressor(objective='reg:squarederror', n_estimators=1000, seed=24,
                             subsample=0.1, colsample_bytree=0.1)
    xgb_model.fit(xTrain, yTrain)
    yPred = xgb_model.predict(xTest)

    plt.figure()
    plt.title(title)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.scatter(yTest, yPred)
    plt.savefig('Predicted vs. Actual.png', dpi=500)

    if tree:
        plt.figure
        xgb.plot_tree(xgb_model)
        plt.savefig('tree.png', dpi=1000)

# Random Forest Model


def rfPred(xTrain, yTrain, xTest, yTest):
    '''
    Makes a prediction after fitting the model to the training data

    Inputs: 2D Numpy Array, 1D Numpy Array, 2D Numpy Array, 1D Numpy Array

    Outputs: 1D Numpy Array, 1D Numpy Array
    '''

    rf = RandomForestRegressor(random_state=42, n_estimators=2000)
    rf.fit(xTrain, yTrain)
    yPred = rf.predict(xTest)
    metrics = errorMetrics(yTest, yPred)
    return yPred, metrics
