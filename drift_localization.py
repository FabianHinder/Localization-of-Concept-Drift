import numpy as np
from scipy.stats import binom

"""
Simple drift localization. H0 for kNN and approximative/virtual leaf for RF.
"""
def simple_drift_localization(X,y,model,k):
    res = model.fit(X,y).predict_proba(X)[:,1]
    y_mean = y.mean()
    p0 = binom.cdf( res*k, k, y_mean )
    return np.vstack( (p0,1-p0) ).min(axis=0)

"""
Simple drfit localization. H0 for kNN. Makes use of the time-point label 
(e.g. before/after drift) thus cannot be extended outside of sample set. 
Usually performes better than simple_drift_localization.
"""
def supervised_drift_localization(X,y,model,k):
    sel = np.vstack( (y==0,y==1) ).T
    res = (k*model.fit(X,y).predict_proba(X)[sel]-1)/(k-1)
    return 1-binom.cdf(res*(k-1), k-1, y.mean())