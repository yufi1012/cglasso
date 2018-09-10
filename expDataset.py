from abc import ABCMeta,abstractmethod
import numpy as np
from bunch import Bunch
from sklearn.datasets import make_sparse_spd_matrix
from scipy.linalg import block_diag
import itertools



def sample(sigma,n_features,n_samples,random_state=None,mu=None):
    if random_state == None:
        random_state = np.random.RandomState()#set random state
    if mu == None:
        mu = np.zeros(n_features)#set mu based on num of features
    X = random_state.multivariate_normal(mu,sigma,n_samples)
    X -= X.mean(axis=0)#compute mean by column
    X /= X.std(axis=0)#compute std by column
    S = np.cov(X.transpose())#estimator of sigma
    return S,X

def is_pos_def(x):#check out whether x is a positive definite matrix
    return np.all(np.linalg.eigvals(x)>0)

def build_diag_prec_block(P,L,density,smallest_coef=.1,largest_coef=.9,tol=0.05,seed=1):
    prec_block = []
    n_features = P/L#L:num of cluster or block,P:total num of features
    for i in xrange(L):
        alpha = 1 - (density + np.random.uniform(-tol,tol))#???
        #generate a sparse symmetric positive matrix, alpha=the probability that a coefficient is 0
        prec = make_sparse_spd_matrix(n_features,alpha=alpha,
                                      smallest_coef=smallest_coef,
                                      largest_coef=largest_coef,
                                      random_state=seed)
        prec_block.append(prec)
    prec = block_diag(*prec_block)#generate this matrix
    labels = []
    for i in xrange(L):
        labels.extend(np.repeat(i,P/L))#generate label of each sample
    labels = np.array(labels)
    H_true = np.mat(np.zeros((P,L)))
    for ind,val in enumerate(labels):#transfer H prior into cluster indicator
        H_true[ind,val] = 1
    
    return prec,H_true

def add_inter_block_connections(X,L,beta=.25,density=.25,
                               smallest_coef=.1,largest_coef=.4,
                               tol=.01,seed=1):
    np.random.seed(seed)
    half_n = X.shape[0]/L
    n_features = half_n*2#???
    pairs = np.array(list(itertools.combinations(range(L),2)))#generate the comparison list
    choosed = np.random.choice(len(pairs),int(beta*len(pairs)),replace=False)
    choosed_pairs = pairs[choosed]
    #inter_matrix = np.mat(np.zeros((L,L)))
    #for i in choosed_pairs:
    #    inter_matrix[i[0],i[1]] = 1
    #inter_matrix = inter_matrix + inter_matrix.T
    #diag_matrix = np.eye(L, dtype=int)
    #inter_matrix = diag_matrix + inter_matrix
    for a,b in choosed_pairs:
        alpha = 1 - (density + np.random.uniform(-tol,tol))
        conn = make_sparse_spd_matrix(n_features,alpha=alpha,
                                     smallest_coef = smallest_coef,
                                     largest_coef = largest_coef,
                                     random_state = seed)
        X[a*half_n:(a+1)*half_n,a*half_n:(a+1)*half_n]+=conn[:half_n,:half_n]
        X[b*half_n:(b+1)*half_n,b*half_n:(b+1)*half_n]+=conn[half_n:,half_n:]
        X[a*half_n:(a+1)*half_n,b*half_n:(b+1)*half_n]+=conn[:half_n,half_n:]
        X[b*half_n:(b+1)*half_n,a*half_n:(a+1)*half_n]+=conn[half_n:,:half_n]
    if not is_pos_def(X):
        print "error: non-spd matrix generated"
        return -1
    return X

'''
prec, X and S
'''
#########################################################################################################
class expDataset(object):
    __meta__ = ABCMeta
    
    def __init__(self,name,desc=""):
        self.name = name
        self.desc = desc
    
    @abstractmethod
    def load(self):
        return

class expDatasetSynthetic(expDataset):
    #name = 'DatasetSynthetic'
    #desc = 'Synthetic dataset generated using multivariate gaussian'
    
    def __init__(self,P,L,n_sample,inner_density,inter_density,beta=.25,
                inner_var=.05,inter_var=.05,
                smallest_coef=.5,largest_coef=.9,
                smallest_coef2=.1,largest_coef2=.5,seed=1):
        super(expDatasetSynthetic,self).__init__('DatasetSynthetic',
                                                 'Synthetic dataset generated using multivariate gaussian')
        self.P = P
        self.L = L
        self.n_sample = n_sample
        self.inner_density = inner_density
        self.inter_density = inter_density
        self.beta = beta
        self.inner_var = inner_var
        self.inter_var = inter_var
        self.smallest_coef = smallest_coef
        self.largest_coef = largest_coef
        self.smallest_coef2 = smallest_coef2
        self.largest_coef2 = largest_coef2
        self.seed = seed
    
    def load(self):#generate precison matrix
        np.random.seed(self.seed)
        prec,H_true = build_diag_prec_block(self.P,self.L,
                                    density=self.inner_density,
                                    smallest_coef=self.smallest_coef,
                                    largest_coef=self.largest_coef,
                                    tol=self.inner_var,
                                    seed=self.seed)
        prec = add_inter_block_connections(prec,self.L,
                                          beta=self.beta,
                                          density=self.inter_density,
                                          smallest_coef=self.smallest_coef2,
                                          largest_coef=self.largest_coef2,
                                          tol=self.inter_var,
                                          seed=self.seed)
        n_features = self.P
        cov = np.linalg.inv(prec)#inverse matrix of prec
        d = np.sqrt(np.diag(cov))
        cov /= d
        cov /= d[:,np.newaxis]
        X = np.random.multivariate_normal(np.zeros(n_features),cov,size=self.n_sample)
        X -= X.mean(axis=0)
        X /= X.std(axis=0)
        
        #prec *= d
        #prec *= d[:,np.newaxis]
        inter_matrix = H_true.T*prec*H_true
        
        emp_cov = np.dot(X.T,X)/self.n_sample#S,estimator of cov
        
        groups = []
        for i in xrange(self.L):
            groups.extend(np.repeat(i,self.P/self.L))#generate label of each sample
        groups = np.array(groups)
        
        self.dataset = Bunch()
        self.dataset.cov = cov
        self.dataset.prec = prec
        self.dataset.inter_matrix = inter_matrix
        self.dataset.X = X
        self.dataset.emp_cov = emp_cov
        self.dataset.groups = groups
        self.dataset.n_samples = self.n_sample
        self.dataset.n_features = X.shape[1]
        
        return self.dataset
    
    def sample(self,n_sample,random_state=None):#generate random sample based on self.dataset.prec
        if random_state is None:
            random_state = np.random.RandomState()
        n_features = self.P
        cov = np.linalg.inv(self.dataset.prec)
        X = random_state.multivariate_normal(np.zeros(n_features),
                                            cov,size=n_sample)
        X -= X.mean(axis=0)
        X /= X.std(axis=0)
        
        return np.cov(X.transpose()),X