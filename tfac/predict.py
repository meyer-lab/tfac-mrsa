from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_predict, KFold
from sklearn.metrics import balanced_accuracy_score


def SVC_predict(X, y):
    """ Perform the prediction with a SVC model. Performs nested cross-validation. """
    p_grid = {"C": [1, 10, 100], "gamma": [0.01, 0.1, 1.0]}
    CV = KFold(n_splits=10, shuffle=True)

    svm = SVC(kernel="rbf")
    clf = GridSearchCV(estimator=svm, param_grid=p_grid, cv=CV, refit=True)

    nested_pred = cross_val_predict(clf, X, y, cv=CV, n_jobs=-1)
    nested_score = balanced_accuracy_score(y, nested_pred)
    clf.fit(X, y)

    return nested_pred, nested_score, clf.best_estimator_
