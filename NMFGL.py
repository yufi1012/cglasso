from abc import ABCMeta,abstractmethod
import numpy as np
from sklearn.covariance import graph_lasso
from sklearn.cluster import spectral_clustering,SpectralClustering,k_means
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
    
    def __init__(self,n_cluster,_lambda,clus_threshold,assign_labels,random_state=1991):
        super(NMFplusGL,self).__init__('NMFGL','NMF plus GL')
        self.n_cluster = n_cluster
        self._lambda = _lambda
        self.clus_threshold = clus_threshold
        self.assign_labels = assign_labels
        self.model_seed = random_state
        
    def predict(self,data):
        start_time = time.time()
        S = np.dot(data.T,data)/data.shape[0]
        #S = np.cov(data.transpose())
        #theta_threshold = self.prec_threshold
        H_threshold = self.clus_threshold
        _,est_groups,_ = k_means(data.transpose(),self.n_cluster,init="k-means++",random_state=self.model_seed)
        #sc = SpectralClustering(n_clusters=self.n_cluster,
        #                        assign_labels = "kmeans",
        #                        random_state=self.model_seed)
        #est_groups = sc.fit_predict(data.transpose())#get H prior
        H_prior = np.mat(np.zeros((data.shape[1],self.n_cluster)))
        for ind,val in enumerate(est_groups):#transfer H prior into cluster indicator
            H_prior[ind,val] = 1
        emp_cov = H_prior.T * S * H_prior#get HSH
        emp_cov = transfer_emp(emp_cov)
        _,theta_prior = graph_lasso(emp_cov,self._lambda)#get theta prior!!!!
            #H_new = iter_H(H_prior,S,theta_prior)#get H new based on H_prior and theta_prior
    
        #start point of iteration
        iteration = 0
        iterations =500
        while iteration < iterations:
            H_new = iter_H(H_prior,S,theta_prior)
            emp_cov = H_new.T * S * H_new
            emp_cov = transfer_emp(emp_cov)
            _,theta_new = graph_lasso(emp_cov,self._lambda)
            H_change = np.sum(np.abs(H_prior[0]-H_new[0]))
    
            if H_change < H_threshold:
                break
            else:
                H_prior = H_new
                theta_prior = theta_new
                iteration += 1
        
        hh = np.argmax(H_new,1)
        labels = np.array(hh).T[0]
        self.timeTotal = time.time() - start_time
    
        return [labels,theta_new,iteration]


########################################################################################
def iter_H(H,S,theta):#function of iteration H
    #theta_plus = (abs(theta)+theta)/2
    #theta_minus = (abs(theta)-theta)/2
    #S_plus = (abs(S) + S)/2
    #S_minus = (abs(S) - S)/2
    gamma = - H.T*S*H*theta + H.T*S*H
    gamma_plus = (abs(gamma)+gamma)/2
    gamma_minus = (abs(gamma)-gamma)/2
    nx,ny = H.shape
    a = [];b = []
    for i in xrange(nx):
        for j in xrange(ny):
            a.append((H*gamma_minus+S*H-H*gamma_plus-S*H*theta)[i,j])
            b.append((H*gamma_plus+S*H*theta)[i,j]+np.exp(-10))
            H[i,j] = H[i,j]+H[i,j]*a[-1]/b[-1]
    H = (abs(H)>0).astype(int)
    return H

def transfer_emp(X):
    nx,ny = X.shape
    hat = np.max(X)
    for i in xrange(nx):
        for j in xrange(ny):
            X[i,j] = X[i,j]/hat
    X = np.array(X)
    return X
