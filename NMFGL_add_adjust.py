from abc import ABCMeta,abstractmethod
import numpy as np
from sklearn.covariance import graph_lasso,shrunk_covariance
from sklearn.cluster import spectral_clustering,SpectralClustering,k_means
from scipy.linalg.misc import norm
import copy
import time

"""
NMF plus GL algorithm
:emp_env:S
:H prior:cluster indicator prior
:theta prior:prec estimator prior
:H threshold:prec threshold
:theta threshold:prec threshold
:param iterations:max interations
:return:local optimal of parameters
"""
class expModel(object):
    __meta__ = ABCMeta
    
    def __init__(self,name,desc=""):
        self.name = name
        self.desc = desc
    
    @abstractmethod
    def predict(self,data):
        return
    
    def clean(self):
        self._clf = None
        return

##################################################################################################
class NMFplusGL(expModel):
    name = "NMFGL"
    desc = "NMF plus GL"
    category = "coherent"
    
    def __init__(self,n_cluster,_lambda,clus_threshold,iterations,assign_labels,random_state=1991):
        super(NMFplusGL,self).__init__('NMFGL','NMF plus GL')
        self.n_cluster = n_cluster
        self._lambda = _lambda
        self.clus_threshold = clus_threshold
        self.iterations = iterations
        self.assign_labels = assign_labels
        self.model_seed = random_state
        
    def predict(self,data):
        start_time = time.time()
        S = abs(np.dot(data.T,data)/data.shape[0])               
        H_threshold = self.clus_threshold
        iterations = self.iterations
        _,est_groups,_ = k_means(data.transpose(),self.n_cluster,init="k-means++",random_state=self.model_seed)
        #est_groups = sc.fit_predict(data.transpose())#get H prior
        H_prior = np.mat(np.zeros((data.shape[1],self.n_cluster)))
        for ind,val in enumerate(est_groups):#transfer H prior into cluster indicator
            H_prior[ind,val] = 1
        H_prior = H_prior + .2
        #A = 1./np.sqrt(sum(H_prior.T*H_prior))
        #B = A.tolist()
        #H_prior = H_prior*np.diag(B[0])
        
        emp_cov = H_prior.T * S * H_prior#get HSH
        #emp_cov = np.array(emp_cov)
        shrunk_cov = shrunk_covariance(emp_cov,shrinkage=0.8)
        _,theta_prior = graph_lasso(shrunk_cov,self._lambda)#get theta prior!!!!
    
        #start point of iteration
        iteration = 0
        #iterations =50
        iteration_H = 0
        while iteration < iterations:
            while iteration_H < 100:
                H_new = iter_H(H_prior,S,theta_prior)
                H_change = np.sum(np.abs(H_prior-H_new))/float(len(H_new))
                
                if H_change < H_threshold:
                    break
                else:
                    H_prior = H_new
                    iteration_H += 1
            
            emp_cov = H_new.T * S * H_new
            shrunk_cov = shrunk_covariance(emp_cov,shrinkage=0.8)
            _,theta_new = graph_lasso(shrunk_cov,self._lambda)
            theta_change = norm(theta_new - theta_prior)
                
            if theta_change < 0.01:
                break
            else:
                theta_prior = theta_new
                H_prior = H_new
                iteration += 1
                iteration_H = 0
                
        hh = np.argmax(H_new,1)
        labels = np.array(hh).T[0]
        self.timeTotal = time.time() - start_time
        theta_global = H_new * theta_new * H_new.T
    
        return [labels,theta_new,H_new,theta_global,iteration]


########################################################################################
def iter_H(H_prior,S,theta_prior):#function of iteration H
    H = copy.deepcopy(H_prior)
    theta = copy.deepcopy(theta_prior)
    gamma = - H.T*S*H*theta
    gamma_plus = (abs(gamma) + gamma)/2.
    gamma_minus = (abs(gamma) - gamma)/2.
    theta_plus = (abs(theta) + theta)/2.
    theta_minus = (abs(theta) - theta)/2.
    grad_H = S*H*theta_plus+H*gamma_plus-S*H*theta_minus-H*gamma_minus
    
    nx,ny = H.shape
    #H_bar = np.zeros((nx,ny))
    alpha_deno = np.zeros((nx,ny))
    
    #modify H_bar
    #for i in xrange(ny):
    #    for j in xrange(ny):
    #        if grad_H[i,j]>=0:
    #            H_bar[i,j]=H[i,j]                
    #        else:
    #            H_bar[i,j]=max(H[i,j],0.1)
    
    #H_bar = np.mat(H_bar)
    #gamma_bar = - H_bar.T*S*H_bar*theta
    #gamma_bar_minus = (abs(gamma_bar) - gamma_bar)/2.
    alpha_deno = S*H*theta_minus+H*gamma_minus+np.exp(-5)
    
    #update H matrx
    for i in xrange(nx):
        for j in xrange(ny):
            H[i,j] = H[i,j] + H[i,j]/float(alpha_deno[i,j])*grad_H[i,j]
    
    #for i in xrange(nx):
    #    for j in xrange(ny):
    #        H[i,j] = H[i,j]*a[-1]/b[-1]
    #H = (abs(H)>0).astype(int)
    return H

#def transfer_emp(X):
#    nx,ny = X.shape
#    hat = np.max(X)
#    for i in xrange(nx):
#        for j in xrange(ny):
#            X[i,j] = X[i,j]/hat
#    X = np.array(X)
#    return X
