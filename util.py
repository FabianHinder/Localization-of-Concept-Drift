import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
from sklearn.metrics import matthews_corrcoef as MCC
from sklearn.metrics import f1_score as F1
from sklearn.metrics import zero_one_loss as ACC

from drift_localization import simple_drift_localization, supervised_drift_localization

def evaluate_model(X,y,y_true, model, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits)
    skf.get_n_splits(X, y)
    nll,acc,tv = 0,0,0
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model.fit(X_train, y_train)
        acc += model.score(X_test,y_test)
        nll += log_loss(y_test,model.predict_proba(X_test))
        tv += np.abs(2*(model.predict_proba(X_test).max(axis=1)-0.5) - y_true[test_index]).mean()
    return nll/n_splits, acc/n_splits, tv/n_splits

def score_drift_localization(X,y,y_true, model,k,p):
    nll,acc,tv = evaluate_model(X,y, y_true, model)
    
    res = simple_drift_localization(X,y, model,k)
    #res = supervised_drift_localization(X,y, model,k)
    return dict([(p_i,(1-ACC(y_true,res < p_i),F1(y_true,res < p_i),MCC(y_true,res < p_i))) for p_i in p]+[("MODEL_ACC",acc),("MODEL_NLL",nll),("MODEL_TV",tv)])
