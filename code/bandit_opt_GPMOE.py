# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 16:36:14 2022

@author: Michael
"""
import os
os.chdir('/home/michaelzhang/Dropbox/hk/hku/code/online_gp/code')

import pdb
import numpy as np
import pymp
from   numpy.random import RandomState
from pymp_GPMOE import ParticleGPMOE
from gp_base import GPBase
from mvt import multivariate_t 
from scipy.special import logsumexp
from utils import _unscaled_dist, pad_kernel_matrix

# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from matplotlib import cm
# from matplotlib.ticker import LinearLocator, FormatStrFormatter

class BanditGPMOE(ParticleGPMOE):
    
    def __init__(self, rng, num_threads, X, Y, J, alpha, X_mean, prior_obs, 
                 nu, psi, alpha_a, alpha_b, mb_size, beta):
        super().__init__(rng=rng,
                         num_threads=num_threads, 
                         X=X, 
                         Y=Y, 
                         J=J, 
                         alpha=alpha, 
                         X_mean=X_mean, 
                         prior_obs=prior_obs, 
                         nu=nu, 
                         psi=psi, 
                         alpha_a=alpha_a, 
                         alpha_b=alpha_b, 
                         mb_size=mb_size)
        # self.T = T
        # self.sample_size = sample_size 
        self.beta = beta        
        
    def sample_points(self):
        """
        Function for sampling points for optimization, must be manually defined 
        """
        raise NotImplementedError()
    
    def opt_samples(self, samples):
        assert(samples.shape[1] == self.D)
        mean_reward, m, v = self.predict(samples)
        max_idx = np.argmax(mean_reward.flatten())
        return(samples[max_idx], mean_reward[max_idx], m[max_idx], v[max_idx], max_idx)

    def predict(self, X_star):        
        pred_j = pymp.shared.dict()
        with pymp.Parallel(self.num_threads) as p:
            for j in p.range(self.J):        
                pred_j[j] = self.predict_j(j,X_star)
        UCB = np.vstack([pred_j[j][0] for j in range(self.J)])
        mean = np.vstack([pred_j[j][1] for j in range(self.J)])
        var = np.vstack([pred_j[j][2] for j in range(self.J)])
        UCB = self.W.dot(UCB)
        mean = self.W.dot(mean)            
        var = self.W.dot(var)
        return(UCB, mean, var)

    def predict_j(self, j, X_star):
        N_star, _ = X_star.shape
        nnz       = self.Z_count[j].nonzero()
        log_prob_N = np.zeros((N_star,nnz[0].size))
        for i, X_i in enumerate(X_star):
            log_prob  = np.log(self.Z_count[j][nnz] + self.alpha[j]) 
            log_prob += [self.posterior_mvn_t(self.X[np.where(self.Z[j]==k)],X_i) for k in nnz[0]]
            log_prob -= logsumexp(log_prob)
            log_prob  = np.exp(log_prob)
            log_prob[log_prob < .01] = 0
            log_prob /= log_prob.sum()
            log_prob_N[i] = log_prob
        out_mean  = np.zeros((N_star,nnz[0].size))
        out_var   = np.zeros((N_star,nnz[0].size))

        for k in np.where(log_prob > 0)[0]:        
            K_mask = self.Z[j] == k
            X_k = np.copy(self.X[K_mask])
            Y_k = np.copy(self.Y[K_mask]).reshape(-1,1)
            N_k = Y_k.size
            if N_k <= self.mb_size:
                gpb = GPBase(rng=self.rng, 
                             hyp=self.models[j][k],
                             mb_weight=1.,
                             full_cov=self.full_cov)
                pred_GP_mean, pred_GP_cov = gpb.predict(X_k, 
                                                        X_star, 
                                                        Y_k, 
                                                        self.kernels[j][k])
            else:
                U = self.rng.choice(a=N_k, size=self.mb_size, replace=False)
                X_k_mb = X_k[U]
                Y_k_mb = Y_k[U]
                kernel_mb = _unscaled_dist(X_k_mb)
                mb_weight = float(N_k) / float(self.mb_size)
                gpb = GPBase(rng=self.rng, 
                             hyp=self.models[j][k],
                             mb_weight=mb_weight,
                             full_cov=self.full_cov)
                pred_GP_mean, pred_GP_cov = gpb.predict(X_k_mb, 
                                                        X_star, 
                                                        Y_k_mb, 
                                                        kernel_mb)
            out_mean[:,k] = pred_GP_mean.flatten()
            out_var[:,k]  = pred_GP_cov.flatten()        
        UCB = np.sum(log_prob_N*(out_mean + np.sqrt(self.beta*out_var)),axis=1)
        mean = np.sum(log_prob_N*out_mean,axis=1)
        var = np.sum(log_prob_N*out_var,axis=1)
        return(UCB, mean, var)

if __name__ == '__main__':
    from scipy.optimize import minimize    
    def f(X, a=1, b=5.1/(4*np.pi**2), c= 5/np.pi,
          r=6.,s=10., t=1./(8*np.pi)):
        """
        Branin-Hoo function
        """
        assert(X.size==2)
        Y = a*(  X[:,1] -b*X[:,0] -r)**2 + s*(1.-t)*np.cos(X[:,0]) +s
        return -Y

    
    seed =0
    rng  = RandomState(seed)
    X = rng.uniform([-5,0],[10,15],size=(1,2))
    Y = np.array(f(X))*np.ones((1,1))
    sample_size = 10
    T = 500
    regret = np.zeros(T)
    bo = BanditGPMOE(rng=rng,
                    num_threads=4,
                    X=X[0,None],
                    Y=Y[0,None],
                    J=100,
                    alpha=1, 
                    X_mean=np.zeros(2), 
                    prior_obs=1, 
                    nu=4, 
                    psi=.5*np.eye(2),
                    alpha_a=10,
                    alpha_b=1,
                    mb_size=50,
                    beta=1.2)


    for t in range(T):
        samples = rng.uniform([-5,0],[10,15],size=(sample_size,2))
        X_opt, ucb, m, v, idx = bo.opt_samples(samples)
        Y_t = f(X_opt[None])*np.ones((1,1))
        bo.particle_update(X_opt[None], Y_t)
        X = np.vstack((X,X_opt))
        Y = np.vstack((Y,Y_t))
        regret[t] = m - f(X_opt[None])        
        print("Time: %i\tAbs. Regret: %.2f" % (t, np.abs(regret[t])))
    plt.plot(np.cumsum(np.abs(regret)) / np.arange(1,T+1))
