from abc import ABCMeta,abstractmethod
from sklearn.covariance import graph_lasso,shrunk_covariance
from sklearn.cluster import k_means,spectral_clustering,SpectralClustering
import numpy as np
import time

"""
Conventional Graphical Lasso using sklearn
Input:X, Gaussian sample data
Output: graph
"""
class expModel(object):
    __meta__=ABCMeta
    
    def __init__(self,name,desc=""):
        self.name = name
        self.desc = desc
    
    @abstractmethod
    def predict(self,X):
        return
    
    def clean(self):
        self._clf = None#what is the meaning of "_clf"?
        return


class expModelGL(expModel):
    name = "GL"
    desc = "Graphical Lasso"
    category = "graph"
    #the meaning of these things???
    
    def __init__(self,_lambda):
        super(expModelGL,self).__init__("GL","Graphical Lasso")
        self._lambda = _lambda
        self._clf = None
        
    def predict(self,X):
        emp_cov = np.cov(X.transpose())#important arg for future study
        start_time = time.time()
        
        shrunk_cov = shrunk_covariance(emp_cov,shrinkage=0.8)
        _,est_prec = graph_lasso(shrunk_cov,self._lambda)
        self.timeTotal = time.time() - start_time
        
        return est_prec

"""
Spectral Clustering
input:X,Gaussian sample data
output:labels
"""
class expModelSpectral(expModel):
    name = "spectral clustering"
    desc = "spectral clustering on time series"
    category = "cluster"
    
    def __init__(self,n_groups,assign_labels="kmeans",random_state=1991):
        super(expModelSpectral,self).__init__("spectral clustering","spectral clustering on time series")
        self.n_clusters = n_groups
        self.model_seed = random_state
        self.assign_labels = assign_labels
    
    def predict(self,X):
        start_time = time.time()
        sc = SpectralClustering(n_clusters = self.n_clusters,
                               random_state = self.model_seed,
                               eigen_solver = "arpack",
                               assign_labels = self.assign_labels)#assign_labels
        labels = sc.fit_predict(X.transpose())
        self.timeTotal = time.time() - start_time
        
        return labels

"""
Kmeans Clustering
Input:X,Gaussian sample data
Output:labels
"""
class expModelKmeans(expModel):
    name = "k-means"
    desc = "k-means"
    category = "cluster"
    
    def __init__(self,n_clusters,init_method="k-means++",random_state=1991):
        super(expModelKmeans,self).__init__("k-means","k-means")
        self.n_clusters = n_clusters
        self.init_method = init_method
        self.model_seed = random_state
    def predict(self,X):
        start_time = time.time()
        _,labels,_ = k_means(X.transpose(),self.n_clusters,
                            init=self.init_method,random_state=self.model_seed)
        self.timeTotal = time.time() - start_time
        
        return labels

"""
Graphical Lasso + Spectral Clustering (non-iterative)
Input:X.Gaussian sample data
Output:labels
"""
class expModelGLassoSpectralClus(expModel):
    name = "glasso-spectral-clus"
    desc = "use spectral clustering with output from glasso"
    category = "cluster"
    
    def __init__(self,n_cluster,_lambda,threshold=0,assign_labels="kmeans",random_state=1991):
        super(expModelGLassoSpectralClus,self).__init__()
        self.n_cluster = n_cluster
        self._lambda = _lambda
        self.prec_threshold = threshold
        self.assign_labels = assign_labels
        self.model_seed = random_state
    
    def predict(self,X):
        emp_cov = np.cov(X.transpose())
        threshold = self.prec_threshold
        start_time = time.time()
        _,affinity_matrix = graph_lasso(emp_cov,self._lamda)
        if threshold == 0:
            affinity_matrix = (abs(affinity_matrix) != 0).astype(int)
        else:
            affinity_matrix[abs(affinity_matrix)<=threshold]=0
        labels = spectral_clustering(affinity_matrix,
                                    n_cluster=self.n_cluster,
                                    random_seed = self.model_seed,
                                    assign_labels = self.assign_labels)
        self.timeToral = time.time() - start_time
        
        return labels
  