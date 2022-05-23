from asyncio.log import logger
from glob import glob
from multiprocessing.sharedctypes import Value
import numpy as np
import random
from sklearn.datasets import make_blobs, fetch_openml
import logging

"""
Generates a normalized (z-scored) dataset with additional gaussian noise dimensions and time points as well as labels.
"""
def generate_dataset(X,y_type, add_dim):
    X = (X-X.mean(axis=0)[None,:])/(X.var(axis=0)+1e-32)[None,:]
    X = np.nan_to_num(X,nan=0,posinf=0,neginf=0)
    X = np.concatenate( (X,np.random.normal(size=(X.shape[0],add_dim))), axis=1 )
    X = X[:,np.random.permutation(range(X.shape[1]))]

    y_type = (y_type % 3) -1
    y_true = y_type != 0

    y = np.empty(y_type.shape)
    y[y_type ==  0] = np.random.choice([0,1],size=(y_type==0).sum(), replace=True)
    y[y_type == -1] = 0
    y[y_type ==  1] = 1

    return X,y.astype(int),y_true.astype(int)

"""
Generates a random RHP dataset sample. Drifting samples are those for which the labeling funcitons do not coincide.
"""
def generate_rhp(n_samps,n_dim,n_add_dim,n_1=2):
    while True:
        X = np.random.random(size=(n_samps*2,n_dim+n_add_dim) )*2-1
        w1,w2 = np.random.normal(size=(n_dim)),np.random.normal(size=(n_dim))
        w1,w2 = np.hstack( (w1,np.zeros(n_add_dim)) ), np.hstack( (w2,np.zeros(n_add_dim)) )
        w1,w2 = (n_1*w1+(n_1-1)*w2)/(2*n_1-1),((n_1-1)*w1+n_1*w2)/(2*n_1-1) # Make hyperplanes sufficently similar so that the dirft is not too easy to find
        
        y1,y2 = np.inner(w1,X) > 0,np.inner(w2,X) > 0
        
        y_true = np.logical_xor(y1,y2)
        X = np.vstack( ((5*X+5).T, np.hstack((y1[:n_samps],y2[n_samps:])) ) ).T
        y = np.array(n_samps*[0]+n_samps*[1])
        
        if 0.2 < y_true.mean() and y_true.mean() < 0.8:
            return X,y,y_true
"""
Generates a random SEA dataset sample. Drifting samples are those for which the labeling funcitons do not coincide.
"""
def generate_sea(n_samps,n_dim,n_add_dim,n_1=2):
    while True:
        X = np.random.random(size=(n_samps*2,n_dim+n_add_dim) )*2-1
        w,theta1,theta2 = np.random.normal(size=(n_dim)),np.random.normal(),np.random.normal()
        w = np.hstack( (w,np.zeros(n_add_dim)) )

        y1,y2 = np.inner(w,X) > theta1,np.inner(w,X) > theta2
        
        y_true = np.logical_xor(y1,y2)
        X = np.vstack( ((5*X+5).T, np.hstack((y1[:n_samps],y2[n_samps:])) ) ).T
        y = np.array(n_samps*[0]+n_samps*[1])
        
        if 0.2 < y_true.mean() and y_true.mean() < 0.8:
            return X,y,y_true
"""
Generates dataset of random gaussian blobs. State is assigned per blob
"""
def generate_blobs(clusts, samps_per_clust, add_dim):
    clusts -= clusts % 3 #equal number per state
    X,y_type = make_blobs(n_samples=clusts*[samps_per_clust])
    return generate_dataset(X,y_type,add_dim)
"""
Generates dataset of uniform randoms on a chessbord. State is assigned per square
"""
def generate_chess(n, m, select, samps_per_clust, add_dim):
    X = np.random.random(size=(samps_per_clust*select*3,2))
    y_type = []
    for t,p in enumerate(random.sample([(i,j) for i in range(n) for j in range(m)], 3*select)): #Select used squares
        y_type.extend(samps_per_clust*[t])
        X[t*samps_per_clust:(t+1)*samps_per_clust,:] += np.array(p)[None,:]
    y_type = np.array(y_type)
    return generate_dataset(X,y_type,add_dim)

def load_mnist():
    try:
        X,y = np.load("MNIST_X.npy"),np.load("MNIST_y.npy", allow_pickle=True).astype(int)
    except:
        logger.warning("Failed to load local copy of MNIST. Try to fatch...")
        X,y=fetch_openml('mnist_784', version=1, return_X_y=True)
        np.save("MNIST_X", X)
        np.save("MNIST_y", y)
    return X,y

"""
Generates dataset as specified by dataset_name.
"""
def load_data(dataset_name, n_additional_noise_dimensions):
    y_true = [1]
    while np.abs(np.mean(y_true)-0.5) > 0.25: #Assure suffices balance
        if type(dataset_name) == tuple:
            if dataset_name[0] == "blob":
                return generate_blobs(dataset_name[1],dataset_name[2],n_additional_noise_dimensions)
            elif dataset_name[0] == "chess":
                return generate_chess(dataset_name[1],dataset_name[2],dataset_name[3],dataset_name[4],n_additional_noise_dimensions)
            elif dataset_name[0] == "SEA":
                return generate_sea(dataset_name[1],dataset_name[2],n_additional_noise_dimensions)
            elif dataset_name[0] == "RHP":
                return generate_rhp(dataset_name[1],dataset_name[2],n_additional_noise_dimensions)
            else:
                raise ValueError("Not found: "+str(dataset_name))
        elif dataset_name == "MNIST":
            X,y = load_mnist()
        else:
            raise ValueError("Not found: "+str(dataset_name))
        
        if X.shape[0] > 1000: #subsampling in case of too many samples
            sel = np.random.choice(range(X.shape[0]),size=1000,replace=False)
            X,y = X[sel],y[sel]
        y = (y-y.min()).astype(int)
        X,y,y_true = generate_dataset(X,np.random.choice([0,1,2],size=y.max()+1,replace=True)[y],n_additional_noise_dimensions)
    return X,y,y_true
