import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import median_absolute_error, mean_squared_error, mean_absolute_error

def estimate_tree_variance(X, y, runs=100, **kwargs):
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    
    imputer = SimpleImputer(strategy='mean')
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)
    
    
    """
    The BaggingRegressor here estimates the average of multiple decision trees
    each trained on a random sub-sample of the training set (~80% of it) by
    averaging their predictions on each testing sample as the final prediction
    """
    h_bar = BaggingRegressor(
        base_estimator=DecisionTreeRegressor(random_state=42, **kwargs),
        n_estimators=runs,
        n_jobs=6,
        max_samples=0.8,
        random_state=42
    )
    h_bar.fit(X_train, y_train)
    
    h_bar_preds = h_bar.predict(X_test)
    
    """
    Here we access each individual tree and take its prediction on the test samples
    """
    estimators_preds = []
    for tree in h_bar.estimators_:
        estimators_preds.append(tree.predict(X_test))
    
    
    """
    Here we simply implement the variance defintion
    Var(h) = E[E_x[(h(x) - h_bar(x)) ^ 2]]
    """
    estimators_preds = np.array(estimators_preds)
    var = np.mean((estimators_preds - h_bar_preds) ** 2)
    
    return var


def estimate_forest_variance(X, y, runs=100, **kwargs):
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    
    imputer = SimpleImputer(strategy='mean')
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)
    
    h_bar = BaggingRegressor(
        base_estimator=RandomForestRegressor(n_estimators=100, **kwargs),
        n_estimators=runs,
        n_jobs=1,
        max_samples=0.8
    )
    h_bar.fit(X_train, y_train)
    
    h_bar_preds = h_bar.predict(X_test)
    
    estimators_preds = []
    for tree in h_bar.estimators_:
        estimators_preds.append(tree.predict(X_test))
    
    estimators_preds = np.array(estimators_preds)
    var = np.mean((estimators_preds - h_bar_preds) ** 2)
    
    return var

def draw_tree_learning_curves(X_train, y_train, X_test, y_test):
    
    variances = []
    train_error, test_error = [], []
    leaves_counts = []
    max_l = 21

    np.random.seed(42)

    for n in range(1, max_l):
        variances.append(estimate_tree_variance(
            np.concatenate((X_train, X_test)), 
            np.concatenate((y_train, y_test)), 
            runs=10, 
            min_samples_leaf=n
        ))
        
        imputer = SimpleImputer()
        X_train_imputed = imputer.fit_transform(X_train)
        X_test_imputed = imputer.transform(X_test)
        
        model = DecisionTreeRegressor(min_samples_leaf=n)
        model.fit(X_train_imputed, y_train)
    
        leaves_counts.append(model.get_n_leaves())
    
        train_error.append(mean_squared_error(y_train, model.predict(X_train_imputed)))
        test_error.append(mean_squared_error(y_test, model.predict(X_test_imputed)))
    
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    

    ax1.plot(leaves_counts, (train_error), color='green', label='Training Error')
    ax1.plot(leaves_counts, (test_error), color='red', linestyle='--', label='Testing Error')

    ax1.legend()
    ax1.set_yticks([])
    ax1.set_xlabel("Number of Leaves")

    ax2.plot(leaves_counts, np.array(variances), color='black', label='Variance')
    ax2.plot(leaves_counts, (np.array(test_error) - np.array(train_error)), color='blue', linestyle='--', label='Test Err - Train Err')

    ax2.legend()
    ax2.set_yticks([])
    ax2.set_xlabel("Number of Leaves")