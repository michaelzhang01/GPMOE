# -*- coding: utf-8 -*-
"""
Created on Sun Feb 27 12:49:08 2022

Some utility functions

@author: Michael Zhang
"""

import numpy as np
import GPy

def inv_logexp(f): # positive reals -> reals
    _lim_val = 36.0
    return np.where(f>_lim_val, f, np.log(np.expm1(f)))
    
def _unscaled_dist(X, X2=None):
    """
    Returns pairwise L2 norm between elements of X and X2. If X2 = None, then
    between the elements in X.
    """
    if X2 is None:
        Xsq = np.sum(np.square(X),1)
        r2 = -2.*GPy.util.linalg.tdot(X) + (Xsq[:,None] + Xsq[None,:])
        GPy.util.diag.view(r2)[:,]= 0. # force diagnoal to be zero: sometime numerically a little negative
        r2 = np.clip(r2, 0, np.inf)
        return np.sqrt(r2)
    else:
        X1sq = np.sum(np.square(X),1)
        X2sq = np.sum(np.square(X2),1)
        r2 = -2.*np.dot(X, X2.T) + X1sq[:,None] + X2sq[None,:]
        r2 = np.clip(r2, 0, np.inf)
        return np.sqrt(r2)

def pad_kernel_matrix(K, X, X_star):
    """
    Adds the elements of X_star to the covariance matrix K, generated from the
    array X

    Parameters:
        K: N x N numpy array, kernel matrix of the elements in X
        X: N x 1 numpy array, X array that generated kernel matrix K
        X_star: N_star x 1, data to be appended to the kernel matrix K
    """
    K = np.copy(K)
    X = np.copy(X)
    X_star = np.copy(X_star)
    assert(K.shape[0] == K.shape[1])
    N, D = X.shape
    assert(K.shape[0] == N)
    assert(X.shape[1] == X_star.shape[1])
    N_star,_ = X_star.shape
    K_new = np.hstack((K, np.zeros((N,N_star))))
    K_new = np.vstack((K_new, np.zeros((N_star,N+N_star)) ))
    KXX_star = _unscaled_dist(X,X_star)
    K_new[-N_star:,:N]=KXX_star.T
    K_new[:N,-N_star:] =KXX_star
    K_new[-N_star:,-N_star:] =_unscaled_dist(X_star)
    return(K_new)