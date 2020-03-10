"""Performs regression on the drug data and cell line factors"""
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
#from xgboost import XGBRegressor
from sklearn import svm
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet


path = os.path.dirname(os.path.abspath(__file__))
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


def dTreePred(xTrain, yTrain, xTest):
    '''
    Makes a prediction after fitting the model to the training data

    Inputs: 2D Numpy Array, 1D Numpy Array, 2D Numpy Array, 1D Numpy Array

    Outputs: 1D Numpy Array, 1D Numpy Array
    '''

    dTree = DecisionTreeRegressor(random_state=42)
    dTree.fit(xTrain, yTrain)
    yPred = dTree.predict(xTest)
    return yPred

# XGBoost Model


def xgbPred(xTrain, yTrain, xTest):
    '''
    Makes a prediction after fitting the model to the training data

    Inputs: 2D Numpy Array, 1D Numpy Array, 2D Numpy Array, 1D Numpy Array

    Outputs: 1D Numpy Array, 1D Numpy Array
    '''

    xgb_model = XGBRegressor(objective='reg:squarederror', n_estimators=1000, seed=24,
                             subsample=0.1, colsample_bytree=0.1)
    xgb_model.fit(xTrain, yTrain)
    yPred = xgb_model.predict(xTest)
    return yPred


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


def rfPred(xTrain, yTrain, xTest):
    '''
    Makes a prediction after fitting the model to the training data

    Inputs: 2D Numpy Array, 1D Numpy Array, 2D Numpy Array, 1D Numpy Array

    Outputs: 1D Numpy Array, 1D Numpy Array
    '''

    rf = RandomForestRegressor(random_state=42, n_estimators=2000)
    rf.fit(xTrain, yTrain)
    yPred = rf.predict(xTest)
    return yPred


def OLSPred(xTrain, yTrain, xTest):
    '''
    Makes a prediction after fitting the model to the training data

    Inputs: 2D Numpy Array, 1D Numpy Array, 2D Numpy Array, 1D Numpy Array

    Outputs: 1D Numpy Array, 1D Numpy Array
    '''

    OLS = LinearRegression()
    OLS.fit(xTrain, yTrain)
    yPred = OLS.predict(xTest)
    return yPred


def LASSOPred(xTrain, yTrain, xTest):
    '''
    Makes a prediction after fitting the model to the training data

    Inputs: 2D Numpy Array, 1D Numpy Array, 2D Numpy Array, 1D Numpy Array

    Outputs: 1D Numpy Array, 1D Numpy Array
    '''

    LASSO = Lasso(alpha=0.075, random_state=42)
    LASSO.fit(xTrain, yTrain)
    yPred = LASSO.predict(xTest)
    return yPred


def RidgePred(xTrain, yTrain, xTest):
    '''
    Makes a prediction after fitting the model to the training data
    Inputs: 2D Numpy Array, 1D Numpy Array, 2D Numpy Array, 1D Numpy Array
    Outputs: 1D Numpy Array, 1D Numpy Array
    '''

    ridge = Ridge(alpha=122.358, random_state=42)
    ridge.fit(xTrain, yTrain)
    yPred = ridge.predict(xTest)
    return yPred


def ElasticNetPred(xTrain, yTrain, xTest):
    '''
    Makes a prediction after fitting the model to the training data
    Inputs: 2D Numpy Array, 1D Numpy Array, 2D Numpy Array, 1D Numpy Array
    Outputs: 1D Numpy Array, 1D Numpy Array
    '''

    elasticNet = ElasticNet(alpha=0.59, l1_ratio=0.031)
    elasticNet.fit(xTrain, yTrain)
    yPred = elasticNet.predict(xTest)
    return yPred


def svrPred(xTrain, yTrain, xTest):
    '''
    Makes a prediction after fitting the model to the training data
    Inputs: 2D Numpy Array, 1D Numpy Array, 2D Numpy Array, 1D Numpy Array
    Outputs: 1D Numpy Array, 1D Numpy Array
    '''
    clf = svm.SVR()
    clf.fit(xTrain, yTrain)
    yPred = clf.predict(xTest)
    return yPred


def KFoldCV(X, y, reg, n_splits=5):
    '''Performs KFold Cross Validation on data'''
    kfold = KFold(n_splits, True, 19)
    y_pred = 0
    r2_scores = np.zeros(n_splits)
    yPredicted = 0
    yActual = 0
    for rep, indices in enumerate(kfold.split(X)):
        X_train, X_test = X[indices[0]], X[indices[1]]
        y_train, y_test = y[indices[0]], y[indices[1]]
        if reg == 'XG':
            y_pred = xgbPred(X_train, y_train, X_test)
        elif reg == 'RF':
            y_pred = rfPred(X_train, y_train, X_test)
        elif reg == 'DT':
            y_pred = dTreePred(X_train, y_train, X_test)
        elif reg == 'OLS':
            y_pred = OLSPred(X_train, y_train, X_test)
        elif reg == 'LASSO':
            y_pred = LASSOPred(X_train, y_train, X_test)
        elif reg == 'SVR':
            y_pred = svrPred(X_train, y_train, X_test)
        elif reg == 'Ridge':
            y_pred = RidgePred(X_train, y_train, X_test)
        elif reg == 'ENet':
            y_pred = ElasticNetPred(X_train, y_train, X_test)

        if rep == 0:
            yPredicted = y_pred
            yActual = y_test
        else:
            yPredicted = np.concatenate((yPredicted, y_pred))
            yActual = np.concatenate((yActual, y_test))
        r2 = r2_score(yActual, yPredicted)
    return r2, yPredicted, yActual
