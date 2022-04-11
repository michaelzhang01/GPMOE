# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 09:43:00 2021

GP core class, taken from GPy

@author: Michael Zhang
"""

import autograd.numpy as np 
from GPy.util.linalg import dtrtrs, tdot, jitchol, dpotrs
# from online_ISMOE import _unscaled_dist
from utils import _unscaled_dist
from autograd import jacobian
from scipy.optimize import minimize
from autograd.scipy.stats import norm
from autograd.scipy.linalg import solve_triangular
import pdb

def logexp(x): # reals -> positive reals
    _log_lim_val = np.log(np.finfo(np.float64).max)
    _lim_val = 36.0
    return np.where(x>_lim_val, x, np.log1p(np.exp(np.clip(x, -_log_lim_val, _lim_val)))) #+ epsilon

def inv_logexp(f): # positive reals -> reals
    _lim_val = 36.0
    return np.where(f>_lim_val, f, np.log(np.expm1(f)))

def jitchol_ag(A, maxtries=5): # autograd friendly version of jitchol
    diagA = np.diag(A)
    if np.any(diagA <= 0.):
        raise np.linalg.LinAlgError("not pd: non-positive diagonal elements")
    jitter = diagA.mean() * 1e-6
    num_tries = 1
    while num_tries <= maxtries and np.isfinite(jitter):
        try:
            L = np.linalg.cholesky(A + np.eye(A.shape[0]) * jitter)
            return L
        except:
            jitter *= 10
        finally:
            num_tries += 1
    raise np.linalg.LinAlgError("not positive definite, even with jitter.")    


class GPBase(object):
    # designed to be a light version of the GPy base object
    def __init__(self, rng, hyp, mb_weight=1, full_cov=False, LL=None, 
                 hyp_var = 10.):
        self.rng = rng
        self.mb_weight = mb_weight
        self.hyp = np.array(hyp)
        self.N_hyp = len(self.hyp)
        self.LL = LL
        self.full_cov = full_cov
        self.grad_obj = jacobian(self.obj_fun)
        self.hyp_var = hyp_var
        # self.mean_func = 0.
        # self.mean_param = 0.

    def compute_mean(self, X, Y):        
        mean_param = np.linalg.solve(X.T @ X, X.T @ Y)        
        mean_func = X @ mean_param
        return mean_func, mean_param
        
    def obj_fun(self, hyp, norm_k, Y_k):
        return -1.*(self.marginal_likelihood(hyp, norm_k, Y_k) + self.hyp_prior(hyp))
    
    def optimize(self,norm_k, Y_k, *args):
        opt = minimize(fun=self.obj_fun, x0=inv_logexp(self.hyp),
                       args=(norm_k, Y_k),
                       jac=self.grad_obj, *args)
        self.LL = -opt.fun
        self.hyp = logexp(opt.x)
        return(self.hyp, self.LL)

    def _computeH(self, hyp, norm_k, Y_k, p, M, Minv):
        H = self.obj_fun(hyp,norm_k, Y_k)
        H += self.N_hyp*np.log(2*np.pi)/2.
        H += np.linalg.slogdet(M)[1]/2.
        H += np.dot(p, np.dot(Minv,p[:,None]))/2.
        return H

    def _update(self, norm_k, Y_k, p, hmc_iters, stepsize, hyp, Minv):
        for i in range(hmc_iters):
            p[:] += -stepsize/2.*self.grad_obj(hyp, norm_k, Y_k)
            hyp[:] += stepsize*np.dot(Minv, p)
            p[:] += -stepsize/2.*self.grad_obj(hyp, norm_k, Y_k)
            if np.any(np.isnan(hyp)) or np.any(np.isnan(p)):
                # if inf grad, quit
                # pdb.set_trace()
                break
        return hyp, p

    def ess(self, norm_k, Y_k):
        nu = self.rng.normal(scale=np.sqrt(self.hyp_var),size = self.N_hyp)
        log_u = np.log(self.rng.uniform())
        LL = self.marginal_likelihood(inv_logexp(self.hyp), norm_k, Y_k) + log_u
        theta = self.rng.uniform(0.,2.*np.pi)
        theta_min = theta - 2.*np.pi
        theta_max = float(theta)
        hyp_proposal = inv_logexp(self.hyp) * np.cos(theta) + nu * np.sin(theta)
        proposal_LL = self.marginal_likelihood(hyp_proposal, norm_k, Y_k)
        while proposal_LL < LL:
            if theta < 0:
                theta_min = float(theta)
            else:
                theta_max = float(theta)
            theta = self.rng.uniform(theta_min,theta_max)
            hyp_proposal = inv_logexp(self.hyp) * np.cos(theta) + nu * np.sin(theta)
            proposal_LL = self.marginal_likelihood(hyp_proposal, norm_k, Y_k)
        
        self.hyp = logexp(hyp_proposal)
        self.LL = proposal_LL
        return(self.hyp, self.LL)
        
    def hmc(self, norm_k, Y_k, M=None, hmc_iters=20, 
            stepsize=1e-1):
        p = np.empty(self.N_hyp)
        if M is None:
            M = np.eye(p.size)
            Minv = np.eye(p.size)
        else:
            M = M
            Minv = np.linalg.inv(M)        

        p[:] = self.rng.multivariate_normal(np.zeros(p.size),M)
        theta_old = inv_logexp(self.hyp)
        H_old = self._computeH(theta_old, norm_k, Y_k, p, M, Minv)            
        hyp, p = self._update(norm_k, Y_k, p, hmc_iters, stepsize, theta_old, Minv)
        if np.any(np.isnan(hyp)) or np.any(np.isnan(p)): # if any nans in p or hyp, go back to old
            p[:] = self.rng.multivariate_normal(np.zeros(p.size),M)
            self.LL = self.marginal_likelihood(theta_old, norm_k, Y_k)
            self.hyp = logexp(theta_old)
        else:
            H_new = self._computeH(hyp, norm_k, Y_k, p, M, Minv)
            if H_old>H_new:
                k = 0
            else:
                k = H_old-H_new
                
            if np.log(self.rng.rand()) < k:
                self.LL = self.marginal_likelihood(hyp, norm_k, Y_k)
                self.hyp = logexp(hyp)
            else:
                self.LL = self.marginal_likelihood(theta_old, norm_k, Y_k)
                self.hyp = logexp(theta_old)
                    
        return self.hyp, self.LL
    
    def predict(self, X, Xnew, Y_k, norm_k):
        N_star = Xnew.shape[0]
        N_k = Y_k.size
        kernel_k = self.hyp[1]*np.exp(  -.5 * (norm_k/ self.hyp[0] )**2)
        kernel_k += (self.hyp[2]/(self.mb_weight)  + 1e-6)*np.eye(N_k)
        woodbury_chol = jitchol(kernel_k)
        woodbury_vector, _ = dpotrs(woodbury_chol, 
                                    # Y_k-self.mean_func, 
                                    Y_k, 
                                    lower=1)

        Kx = self.hyp[1]*np.exp(  -.5 * (_unscaled_dist(X,Xnew)/ self.hyp[0] )**2)
        mu = np.dot(Kx.T, woodbury_vector) #+ np.dot(Xnew, self.mean_param)
            
        if len(mu.shape) == 1:
            mu = mu.reshape(-1, 1)
        if self.full_cov:
            norm_X_star = _unscaled_dist(Xnew)
            Kxx = self.hyp[1]*np.exp(  -.5 * (norm_X_star/ self.hyp[0] )**2)
            Kxx += ((self.hyp[2]/self.mb_weight)  + 1e-6)*np.eye(N_star)
            tmp = dtrtrs(woodbury_chol, Kx)[0]
            var = Kxx - tdot(tmp.T)
            var = var
        else:
            Kxx = (self.hyp[1]+(self.hyp[2]/self.mb_weight))
            tmp = dtrtrs(woodbury_chol, Kx)[0]
            var = (Kxx - np.square(tmp).sum(0))[:, None]
            var = var
        return mu, var

    def hyp_prior(self,hyp):
        prior = norm.logpdf(hyp, np.zeros(self.N_hyp), 
                          np.sqrt(self.hyp_var)*np.ones(self.N_hyp)).sum()
        return(prior)
        
    def marginal_likelihood(self, hyp, norm_k, Y_k): # hyp is log [lengthscale, amplitude, gaussian noise]
        """
        Calculates the marginal likelihood of the GP model
        Parameters:
            hyp: M x 1 array, hyperparameters of the kernel
            norm_k: N x N numpy array, pairwise distances of data
            Y_k: N_k x 1 numpy array, output data
        """
        hyp  = logexp(hyp)
        N_k  = Y_k.size
        #Y_k -= self.mean_func
        if N_k > 1:
            kernel_k  = hyp[1]*np.exp(  -.5 * (norm_k/ hyp[0] )**2)
            kernel_k += ((hyp[2]/self.mb_weight)  + 1e-6)*np.eye(N_k)
            try:
                LW = jitchol_ag(kernel_k)
            except np.linalg.LinAlgError: # if cholesky fails, give up
                return(-np.inf)
            W_logdet = 2.*np.sum(np.log(np.diag(LW)))
            alpha = solve_triangular(LW.T, solve_triangular(LW, Y_k, lower=1)) #dpotrs
            LL =  0.5*(-N_k*self.mb_weight * np.log(2.*np.pi) -  W_logdet - np.sum(alpha * Y_k))
        else:
            kernel_k = (hyp[2]/(self.mb_weight)  + hyp[1])
            LL = 0.5*(-N_k*self.mb_weight*  np.log(2.*np.pi) - np.log(kernel_k) - (Y_k[0][0]**2 / kernel_k))

        return(LL)
        
if __name__ == '__main__':
    from sklearn.metrics.pairwise import rbf_kernel
    from GPy.models import GPRegression
    from sklearn.model_selection import train_test_split
    X = np.linspace(-1,1,1000)[:,None]
    Kx = rbf_kernel(X)
    f = self.rng.multivariate_normal(np.zeros(1000),Kx)
    Y = self.rng.normal(f, np.sqrt(.5))[:,None]
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y,train_size=.8)

    Kx2 = _unscaled_dist(X_train)
    gpb = GPBase([1,1,1])
    gpb.optimize(Kx2, Y_train)
    g_m,g_v = gpb.predict(X_train,X_test,Y_train,Kx2)
    m = GPRegression(X_train,Y_train)
    m.optimizer_array = inv_logexp(np.ones(3))
    m.optimize()
    m_m, m_v = m.predict(X_test)
    hmc_hyp, hmc_LL = gpb.hmc(Kx2, Y_train)
    ess_hyp, ess_LL = gpb.ess(Kx2, Y_train)
