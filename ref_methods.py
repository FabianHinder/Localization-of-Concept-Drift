import numpy as np
from sklearn.metrics import matthews_corrcoef as MCC
from sklearn.metrics import f1_score as F1
from sklearn.metrics import zero_one_loss as ACC
from sklearn.neighbors import KNeighborsClassifier
from scipy.stats import norm
import logging

"""
Computes drift localization based on kdq-Trees (Dasu et al., 2006)
"""
def kdqtree(X,y,y_true, min_size=0.1,min_samps=10, dim=0, theta=[0.5], max_disc=[0]):
    if X.shape[1] > 50:
        X = X[:,np.random.choice(range(X.shape[1]),50,replace=False)]
    y_tp = kdqtree0(X,y,y_true, y.sum(),(1-y).sum(), min_size, min_samps, dim)
    y_true, y_pred = np.array([x[0] for x in y_tp]), np.array([x[1] for x in y_tp])
    return dict([(t , (1-ACC(y_true,y_pred > t),F1(y_true,y_pred > t),MCC(y_true,y_pred > t)) ) for t in theta])
def kdqtree0(X,y,y_true, y0,y1, min_size=0.1,min_samps=10, dim=0,stop_dim=-1):
    y,y_true = y.astype(bool),y_true.astype(bool)
    if (~y).sum() < min_samps or dim == stop_dim:
        p = y.sum() / y0
        q = (1-y).sum() / y1
        d_kl_v = (p-q)*np.log( (p*(1-q))/(q*(1-p)) ) if (1-p)*(1-q)*p*q != 0 else np.inf # Computes *symmetrized* Kullback–Leibler divergence for the spacial search statistic
        return list(zip(list(y_true),len(y_true)*[2*np.abs(y.mean()-0.5)]))
    else:
        min_, max_ = X[:,dim][y==0].min(),X[:,dim][y==0].max()
        if abs(min_-max_) < min_size:
            return kdqtree0(X,y,y_true, y0,y1, min_size=min_size,min_samps=min_samps, dim=(dim+1)%X.shape[1], stop_dim=stop_dim if stop_dim != -1 else dim)
        else:
            s = (min_+max_)/2
            I = X[:,dim] < s; J = np.logical_not(I)
            return   kdqtree0(X[I],y[I],y_true[I], y0,y1, min_size=min_size,min_samps=min_samps, dim=(dim+1)%X.shape[1]) \
                   + kdqtree0(X[J],y[J],y_true[J], y0,y1, min_size=min_size,min_samps=min_samps, dim=(dim+1)%X.shape[1])

"""
Computes drift localization based on LDD-DSI (Liu et al., 2017)
"""
def LDD(X,y,y_true, k=8, alpha=[0.05]):
    sigma = delta(X,np.random.permutation(y),k).std()
    delta_ = delta(X,y,k)
    
    res = []
    for alph in alpha:
        theta_dec = norm.ppf(alph, loc=0, scale=sigma)
        theta_inc = norm.ppf(1-alph, loc=0, scale=sigma)
        y_pred = np.logical_or(delta_ < theta_dec, theta_inc < delta_)
        res.append( (alph, (1-ACC(y_true,y_pred),F1(y_true,y_pred),MCC(y_true,y_pred)) ) )
    return dict(res)
def delta(X,y,k):
    y = y.astype(int)
    x = KNeighborsClassifier(n_neighbors=k).fit(X,y).predict_proba(X)[np.vstack( (y==0,y==1) ).T]
    x *= 1-2*1e-16;x += 1e-16
    return x/(1-x)-1

"""
Computes drift localization based on knn. Optimal threashold is defined in a post hoc fashion (BASELINE ONLY!.
"""
def knn_post_hoc(X,y,y_true, k=8, theta=None):
    if abs(y.mean()-0.5) > 0.05:
        logging.warning("knn assumes P[t = 0] ~ 0.50, current is %.3f"%y.mean())
    if theta is None:
        theta = np.linspace(0,1,20)
    sigma = 2*np.abs(KNeighborsClassifier(n_neighbors=k).fit(X,y).predict_proba(X)[:,0]-0.5)
    return {-1: ( max([1-ACC(y_true, sigma > t) for t in theta]),
             max([F1(y_true, sigma > t) for t in theta]),
             max([MCC(y_true, sigma > t) for t in theta]) )}