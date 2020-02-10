"""Performs regression on the drug data and cell line factors"""
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor

def errMetric(y_test, y_pred):
    '''
    Determines error values based off of predicted and actual values
    Inputs: 1D Numpy Arrays
    Outputs: Prints rmse and r2  and returns them as Float64
    '''
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    print('Model Performance')
    print('Root Mean Squared Error: {:0.4f}'.format(rmse))
    print('R2 Score = {:0.4f}'.format(r2))
    return rmse, r2

# XGBoost Model
def xgbPred(xTrain, yTrain, xTest, yTest):
    '''
    Makes a prediction after fitting the model to the training data

    Inputs: 2D Numpy Array, 1D Numpy Array, 2D Numpy Array, 1D Numpy Array

    Outputs: 1D Numpy Array, float64, float64
    '''

    xgb_model = XGBRegressor(objective='reg:squarederror', n_estimators=1000, seed=24,
                             subsample=0.1, colsample_bytree=0.1)
    xgb_model.fit(X_train, y_train)
    yPred = xgb_model.predict(xTest)
    rmse, r2 = errMetric(yTest, yPred)
    return yPred, rmse, r2

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

# Random Forest Model
def rfPred(xTrain, yTrain, xTest, yTest):
    '''
    Makes a prediction after fitting the model to the training data

    Inputs: 2D Numpy Array, 1D Numpy Array, 2D Numpy Array, 1D Numpy Array

    Outputs: 1D Numpy Array, float64, float64
    '''

    rf = RandomForestRegressor(random_state = 42, n_estimators = 2000)
    rf.fit(X_train, y_train)
    yPred = rf.predict(xTest)
    rmse, r2 = errMetric(yTest, yPred)
    return yPred, rmse, r2