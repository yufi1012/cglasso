from abc import ABCMeta,abstractmethod
import numpy as np
#from sklearn.covariance import graph_lasso,shrunk_covariance
from sklearn.cluster import spectral_clustering,SpectralClustering,k_means
from scipy.linalg.misc import norm
import copy
import time

"""
orthogonal nonnegative matrix tri-factorizations
:emp_env:S
:H prior:k-means + .2
:M prior:H.T*S*H
:H threshold:
:M threshold:
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
class onmtf(expModel):
    name = "onmtf"
    desc = "orthogonal nonnegative matrix tri-factorizations"
    category = "cluster"
    
    def __init__(self,n_cluster,clus_threshold,assign_labels,random_state=1991):
        super(onmtf,self).__init__('onmtf','orthogonal nonnegative matrix tri-factorizations')
        self.n_cluster = n_cluster
        self.clus_threshold = clus_threshold
        self.assign_labels = assign_labels
        self.model_seed = random_state
        
    def predict(self,data):
        start_time = time.time()
        S = abs(np.dot(data.T,data)/data.shape[0])               
        H_threshold = self.clus_threshold
        
        #get H prior
        _,est_groups,_ = k_means(data.transpose(),self.n_cluster,init="k-means++",random_state=self.model_seed)
        H_prior = np.mat(np.zeros((data.shape[1],self.n_cluster)))
        for ind,val in enumerate(est_groups):#transfer H prior into cluster indicator
            H_prior[ind,val] = 1
        H_prior = H_prior + .2
        
        #get M prior
        M_prior = H_prior.T*S*H_prior
        
        #start point of iteration
        iteration = 0
        iterations =50
        iteration_H = 0
        while iteration < iterations:
            while iteration_H < 50:
                H_new = iter_H(H_prior,S,M_prior)
                H_change = np.sum(np.abs(H_prior-H_new))/float(len(H_new))
                
                if H_change < H_threshold:
                    break
                else:
                    H_prior = H_new
                    iteration_H += 1
            
            M_new = iter_M(H_new,S,M_prior)
            M_change = norm(M_new - M_prior)
                
            if M_change < 0.01:
                break
            else:
                M_prior = M_new
                H_prior = H_new
                iteration += 1
                iteration_H = 0
                
        hh = np.argmax(H_new,1)
        labels = np.array(hh).T[0]
        self.timeTotal = time.time() - start_time
        #theta_new = np.linalg.inv(M_new)
        #theta_global = np.linalg.inv(H_new * M_new * H_new.T)
    
        return labels


########################################################################################
def iter_H(H_prior,S,M_prior):#function of iteration H
    H = copy.deepcopy(H_prior)
    M = copy.deepcopy(M_prior)
    numerator = S.T*H*M
    denominator = H*H.T*S*H*M
    
    nx,ny = H.shape
    for i in xrange(nx):
        for j in xrange(ny):
            H[i,j] = H[i,j]*numerator[i,j]/(float(denominator[i,j])+np.exp(-5))
    
    return H

def iter_M(H_new,S,M_prior):
    H = copy.deepcopy(H_new)
    M = copy.deepcopy(M_prior)
    numerator = H.T*S*H
    denominator = H.T*H*M*H.T*H
    
    nx,ny = M.shape
    for i in xrange(nx):
        for j in xrange(ny):
            M[i,j] = M[i,j]*numerator[i,j]/(float(denominator[i,j])+np.exp(-5))
    
    return M
    
