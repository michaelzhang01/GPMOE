# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 22:13:42 2022

SMC GP-MOE Code

@author: Michael Zhang
"""
from gp_base import GPBase
from mvt import multivariate_t 
from scipy.special import logsumexp
from scipy.stats import norm
from utils import _unscaled_dist, pad_kernel_matrix
import numpy as np
import pdb
import pymp

class ParticleGPMOE(object):

    def __init__(self, rng, num_threads, X, Y, J, alpha, X_mean, prior_obs, 
                 nu, psi, alpha_a, alpha_b, mb_size):
        """
        Initialize the GP-MOE object.
        Parameters:
            rng: Numpy RandomState object, used to set random seed
            num_threads: int, number of cores to use for OpenMP process
            X: 1 x D numpy array, initial input data
            Y: 1 x 1 numpy array, initial output data
            J: int, number of particles to use
            alpha: positive float, initial concentration parameter value
            X_mean: 1 x D numpy array, prior mean of X
            prior_obs: positive int, prior number of observations for Normal-I.W. mixture
            nu: positive int > D - 1, prior degrees of freedom for I.W. distribution
            psi: D x D positive definite numpy array, prior covariance matrix for I.W. distribution
            alpha_a: positive float, prior shape of alpha
            alpha_b: positive float, prior scale of alpha
            mb_size: positive int, minibatch size; set to None to not use minibatching
        """


        # global settings
        self.rng = rng
        self.num_threads = num_threads
        self.X = X
        self.Y = Y
        self.N, self.D = self.X.shape
        self.X_star = np.zeros((1,self.D))
        self.J = J
        self.W = np.ones(self.J)/self.J

        # dpmm hyper parameters
        self.K = 1
        self.alpha = alpha*np.ones(self.J)
        self.X_mean = X_mean
        self.prior_obs = prior_obs
        self.nu = nu
        self.psi = psi
        self.alpha_a = alpha_a
        self.alpha_b = alpha_b

        # dpmm parameters
        self.Z = np.zeros((self.J,1)).astype(int)
        self.max_Z = self.Z.max()
        self.K = np.ones(self.J,dtype=int)
        self.Z_count = np.array([np.bincount(self.Z[j],
                                    minlength=self.Z.max()) for j in range(self.J)])
        self.alpha = self.parallel_alpha()
        self.dpmm_marg_LL = self.parallel_init_marg_LL()

        # gp parameters
        if mb_size is None:
            self.mb_size = np.inf
        else:
            self.mb_size = mb_size

        self.full_cov = False
        self.kernels = {j:None for j in range(self.J)}
        out = self.parallel_model_init()
        self.kernels = {j: dict(out[j][2]) for j in range(self.J)}
        self.gp_marg_LL_k = {j:dict(out[j][1]) for j in range(self.J)}
        self.gp_marg_LL = np.array([sum(self.gp_marg_LL_k[j].values()) for j in range(self.J)])
        self.models = {j:dict(out[j][0]) for j in range(self.J)}
        
    def parallel_model_init(self):
        out = pymp.shared.dict()
        with pymp.Parallel(self.num_threads) as p:
            for j in p.range(self.J):
                out[j] = self.model_init(j)
        return(out)
        
    def model_init(self,j):
        """
        Initializes GPbase objects and optimizes hyperparameters
        Parameters:
            j: int, index for particle j
        """
        gp_model = {}
        gp_marg_LL_k = {}
        kernel_j_k = {0:_unscaled_dist(self.X)}
        Y_k = np.copy(self.Y).reshape(-1,1)
        X_k = np.copy(self.X)
        init_hyp = self.rng.gamma(1,1,size=3)
        gpb = GPBase(rng=self.rng, hyp=init_hyp, mb_weight=1.,
                     full_cov=self.full_cov)

        gp_model[0], gp_marg_LL_k[0] = gpb.ess(kernel_j_k, Y_k)

        return(gp_model, gp_marg_LL_k, kernel_j_k)

    def parallel_init_marg_LL(self):
        out = pymp.shared.array((self.J,), dtype='float')
        with pymp.Parallel(self.num_threads) as p:
            for j in p.range(self.J):
                out[j] = self.init_marg_LL(j)
        return out
    
    def init_marg_LL(self, j):
        Z_count = (self.Z_count[j])
        Z = (self.Z[j])
        alpha = self.alpha[j]
        log_prob  = np.log(np.hstack((Z_count,alpha)) / (alpha + self.N))
        log_prob += np.hstack([self.posterior_mvn_t(self.X[Z==k,:],self.X[0]) for k in range(log_prob.size)])
        marg_LL   = logsumexp(log_prob)
        return(marg_LL)

    def particle_update(self, X_star, Y_star):
        """
        Update the GP-MOE object with a new observation.
        Parameters:
            X: 1 x D numpy array, new input data
            Y: 1 x 1 numpy array, new output data
        """
        self.X_star = X_star
        self.X_star.shape[1] == self.D
        self.Y_star = Y_star
        self.alpha = self.parallel_alpha()
        out = self.parallel_update()        
        self.X  = np.vstack((self.X,self.X_star))
        self.Y  = np.vstack((self.Y,self.Y_star))
        self.N += 1
        self.X_star = np.zeros((1,self.D))
        self.Z = np.array([out[j][0] for j in range(self.J)])
        self.max_Z = self.Z.max()
        self.K = np.array([np.unique(self.Z[j]).size for j in range(self.J)],dtype=int)
        self.Z_count = np.array([np.bincount(self.Z[j], minlength=self.max_Z+1) for j in range(self.J)])
        self.dpmm_marg_LL = np.array([out[j][1] for j in range(self.J)])
        self.W = np.log(self.W) + self.dpmm_marg_LL - self.gp_marg_LL
        self.models = {j:dict(out[j][2]) for j in range(self.J)}
        self.gp_marg_LL_k = {j:dict(out[j][3]) for j in range(self.J)}
        self.gp_marg_LL = np.array([sum(self.gp_marg_LL_k[j].values()) for j in range(self.J)])
        self.kernels = {j:dict(out[j][4]) for j in range(self.J)}        
        self.W = self.W + self.gp_marg_LL
        self.W = np.exp(self.W  - logsumexp(self.W))
        self.W = self.W / self.W.sum()
        ESS = 1./((self.W**2).sum())
        print("ESS: %.2f" % ESS)
        if ESS < .5*self.J:
            print("Resampling.")
            resample_idx = self.rng.choice(self.J,
                                            p=self.W,
                                            size=self.J)
            self.Z = self.Z[resample_idx]
            self.max_Z = self.Z.max()
            self.Z_count = np.array([np.bincount(self.Z[j], minlength=self.max_Z+1) for j in range(self.J)])
            self.dpmm_marg_LL = self.dpmm_marg_LL[resample_idx]
            self.gp_marg_LL_k = {idx:dict(self.gp_marg_LL_k[j]) for idx,j in enumerate(resample_idx)}
            self.models = {idx:dict(self.models[j]) for idx, j in enumerate(resample_idx)}
            self.kernels = {idx:dict(self.kernels[j]) for idx, j in enumerate(resample_idx)}
            self.gp_marg_LL = np.array([sum(self.gp_marg_LL_k[j].values()) for j in range(self.J)])
            self.alpha = self.alpha[resample_idx]
            self.W = (1./self.J)*np.ones(self.J)

    def parallel_update(self):
        out = pymp.shared.dict()
        with pymp.Parallel(self.num_threads) as p:
            for j in p.range(self.J):        
                Z, log_norm = self.crp_predict(j)
                out[j] = self.gp_update(j, Z, log_norm)
        return(out)

    def crp_predict(self, j):
        """
        Assigns new data sequentially to according to CRP and multivariate-t
        marginal likelihood
        """
        Z_count = self.Z_count[j][self.Z_count[j].nonzero()]
        Z = np.copy(self.Z[j])
        alpha = float(self.alpha[j])
        K = int(self.K[j])        
        log_prob  = np.log(np.hstack((Z_count,alpha)) / (alpha + self.N))        
        log_prob += [self.posterior_mvn_t(self.X[np.where(Z==k)],self.X_star[0]) for k in range(log_prob.size)]
        log_norm  = logsumexp(log_prob)
        log_prob  = np.exp(log_prob-log_norm)
        Z_i = self.rng.multinomial(1,log_prob).argmax()
        Z = np.append(Z,Z_i)
        return(Z,log_norm)

    def gp_update(self, j, Z, log_norm):
        k = Z[-1]        
        K_mask = Z[:-1]==k    
        if k > Z[:-1].max():
            kernel_j_k = _unscaled_dist(self.X_star)
            init_hyp = self.rng.gamma(1,1,size=3)
            gpb = GPBase(rng=self.rng, hyp=init_hyp, 
                         mb_weight=1.,
                         full_cov=self.full_cov)            
            models_j_k, gp_marg_LL_k = gpb.ess(kernel_j_k, self.Y_star)
        else:
            X_k = np.copy(self.X[K_mask])
            Y_k = np.copy(self.Y[K_mask]).reshape(-1,1)        
            N_k = int(Y_k.size)            
            if N_k <= self.mb_size:
                new_Y = np.vstack((Y_k,self.Y_star))
                new_X = np.vstack((X_k,self.X_star))
                kernel_j_k = pad_kernel_matrix(self.kernels[j][k], X_k, self.X_star)    
                gpb = GPBase(rng=self.rng, hyp=self.models[j][k], 
                             mb_weight=1.,
                             full_cov=self.full_cov)            
            else:
                U = self.rng.choice(a=N_k, size=self.mb_size, replace=False)
                new_X = np.vstack((X_k[U], self.X_star))
                new_Y = np.vstack((Y_k[U], self.Y_star))
                kernel_j_k = _unscaled_dist(new_X)
                mb_weight = float(N_k) / float(self.mb_size)
                gpb = GPBase(rng=self.rng, hyp=self.models[j][k], 
                             mb_weight=mb_weight,
                             full_cov=self.full_cov)            
            models_j_k, gp_marg_LL_k = gpb.ess(kernel_j_k, new_Y)
        model_j = dict(self.models[j])
        model_j[k] = models_j_k
        new_gp_marg_j = dict(self.gp_marg_LL_k[j])
        new_gp_marg_j[k] = gp_marg_LL_k
        kernel_j = dict(self.kernels[j])
        kernel_j[k] = kernel_j_k
        return(Z, log_norm, model_j, new_gp_marg_j, kernel_j)

    def posterior_mvn_t(self,X_k,X_star_i):
        """
        Calculates the multivariate-t distributed marginal likelihood of a
        for a NIW mixture model.
        Parameters:
            X_k: N_k x D numpy array, data assigned to cluster k
            X_star_i : 1 x D numpy array, likelihood calculated for this
                       observation
        """
        if X_k.shape[0] > 0:
            X_bar = X_k.mean(axis=0)
            diff = X_k - X_bar
            SSE = np.dot(diff.T,diff)
            N_k = X_k.shape[0]
            prior_diff = X_bar - self.X_mean
            SSE_prior = np.outer(prior_diff.T, prior_diff)
        else:
            X_bar = 0.
            SSE = 0.
            N_k = 0.
            SSE_prior = 0.

        mu_posterior = (self.prior_obs * self.X_mean) + (N_k * X_bar)
        mu_posterior /= (N_k + self.prior_obs)
        nu_posterior = self.nu + N_k
        lambda_posterior = self.prior_obs + N_k
        psi_posterior = self.psi + SSE
        psi_posterior += ((self.prior_obs * N_k) / (
                    self.prior_obs + N_k)) * SSE_prior
        psi_posterior *= (lambda_posterior + 1.) / (
                    lambda_posterior * (nu_posterior - self.D + 1.))
        df_posterior = (nu_posterior - self.D + 1.)
        return multivariate_t.logpdf(X_star_i, mu_posterior, psi_posterior,
                                     df_posterior)     

    def parallel_alpha(self):
        alpha_array = pymp.shared.array((self.J,), dtype='float')
        with pymp.Parallel(self.num_threads) as p:
            for j in p.range(self.J):
                alpha_array[j] = self.alpha_resample(j)
        return(alpha_array)
    
    def alpha_resample(self,j):
        """code to gibbs sample alpha"""
        K_j = self.K[j]
        eta = self.rng.beta(self.alpha[j] + 1, 
                              self.N)        
        ak1 = self.alpha_a + K_j - 1
        pi = ak1 / (ak1 + self.N * (self.alpha_b - np.log(eta)))
        a = self.alpha_a + K_j
        b = self.alpha_b - np.log(eta)
        gamma1 = self.rng.gamma(a, 1./ b) 
        gamma2 = self.rng.gamma(a - 1, 1./ b) 
        return(pi * gamma1 + (1 - pi) * gamma2)

    def predict(self, X_star):     
        """
        Predict at test point X_star.
        Parameters:
            X_star: N_star x D numpy array, test data
        """
        pred_j = pymp.shared.dict()
        with pymp.Parallel(self.num_threads) as p:
            for j in p.range(self.J):        
                pred_j[j] = self.predict_j(j,X_star)
        mean = np.vstack([pred_j[j][0] for j in range(self.J)])
        var = np.vstack([pred_j[j][1] for j in range(self.J)])
        mean = self.W.dot(mean)
        var = self.W.dot(var)
        return(mean,var)
                      
    def predict_j(self, j, X_star):
        N_star, _ = X_star.shape
        nnz       = self.Z_count[j].nonzero()
        log_prob_N = np.zeros((N_star,nnz[0].size))
        for i,X_i in enumerate(X_star):
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
        pred_mean = np.sum(log_prob*out_mean, axis=1)
        pred_var  = np.sum(log_prob*out_var, axis=1)  
        return(pred_mean, pred_var)

if __name__ == '__main__':
    from   numpy.random import RandomState
    import time
    rng  = RandomState(0)
    motorcycle = np.loadtxt("../data/motorcycle.txt")
    motorcycle[:,0] -= motorcycle[:,0].mean()
    motorcycle[:,0] /= np.sqrt(motorcycle[:,0].var())
    motorcycle[:,1] -= motorcycle[:,1].mean()
    motorcycle[:,1] /= np.sqrt(motorcycle[:,1].var())
    X = motorcycle[:,0][:,None]
    Y = motorcycle[:,1][:,None]
    Y -= Y.mean()
    Y /= Y.std()
    N = Y.size
    X = np.linspace(-1,1,N)[:,None]
    X -= X.mean()
    X /= X.std()
    gpmoe  = ParticleGPMOE(rng=rng,
                           num_threads=16,
                           X=X[0,None],
                           Y=Y[0,None],
                           J=100,
                           alpha=1, 
                           X_mean=np.zeros(1), 
                           prior_obs=1, 
                           nu=3, 
                           psi=.5*np.eye(1),
                           alpha_a=10,
                           alpha_b=1,
                           mb_size=1)

    pred_m = np.empty((0,1))
    pred_v = np.empty((0,1))

    for i in range(1,N,1):
        m, v = gpmoe.predict(X[i,None])
        pred_m = np.vstack((pred_m,m))
        pred_v = np.vstack((pred_v,v))
        start=time.time()
        gpmoe.particle_update(X[i,None], Y[i,None])
        end_time=time.time()-start
        MSE = np.mean((m - Y[i])**2)
        print("Obs: %i\tPredict Time: %.2f\tMSE: %.2f" % (i, end_time, MSE))
