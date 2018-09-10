from abc import ABCMeta,abstractmethod
import numpy as np
from sklearn.metrics import homogeneity_score
from sklearn.metrics import completeness_score
from sklearn.metrics import normalized_mutual_info_score as nmi_score

def purity_score(classes,clusters):
    """
    calculate the purity score for the given cluster assignments and truth classes
    
    :param clusters: the cluster assignments array
    :type clusters: numpy.array
    
    :param classes: the ground truth classes
    :type classes: numpy.array
    
    :returns: the purity score
    :rtype: float
    """
    A = np.c_[(clusters,classes)]
    
    n_accurate = 0.
    
    for j in np.unique(A[:,0]):
        z = A[A[:,0] == j, 1]
        x = np.argmax(np.bincount(z))#index of the largest num along the axis
        n_accurate += len(z[z == x])
    
    return n_accurate / A.shape[0]

def edge_recall(prec,est_prec,threshold=0):
    true_edges = (abs(prec)>threshold).astype(int)#transfer elements to sign of edge
    est_edges = (abs(est_prec)>threshold).astype(int)
    np.fill_diagonal(true_edges,0)#transfer diagonal elements to 0
    np.fill_diagonal(est_edges,0)
    n_d = np.sum(((est_edges+true_edges) == 2).astype(int))
    n_g = np.sum(true_edges)#true!!!!
    
    if n_g == 0:
        return 0
    return float(n_d)/float(n_g)

def edge_accuracy(prec,est_prec,threshold=0):
    true_edges = (abs(prec)>threshold).astype(int)
    est_edges = (abs(est_prec)>threshold).astype(int)
    np.fill_diagonal(true_edges,0)
    np.fill_diagonal(est_edges,-1)
    n_d = np.sum((est_edges == true_edges).astype(int))
    n_g = prec.shape[0]*prec.shape[1]-prec.shape[0]#all edge!!!
    
    if n_g == 0:
        return 0
    return float(n_d)/float(n_g)

def edge_prec(prec,est_prec,threshold=0):
    true_edges = (abs(prec)>threshold).astype(int)
    est_edges = (abs(est_prec)>threshold).astype(int)
    np.fill_diagonal(true_edges,0)
    np.fill_diagonal(est_edges,0)
    n_nd = np.sum(((true_edges-est_edges)==-1).astype(int))
    n_g = np.sum(est_edges)#est!!!
    if n_g == 0:
        return 0
    return float(n_g - n_nd)/float(n_g)

def edge_F1(prec,est_prec,threshold=0):
    rec = edge_recall(prec,est_prec,threshold=threshold)
    pre = edge_prec(prec,est_prec,threshold=threshold)
    if (pre+rec)==0:
        return 0
    else:
        return 2*pre*rec/float(pre+rec)


########################################################################################
'''
Normalized Mutual Information(NMI) is an normalization of the Mutual Information(MI) score
to scale the results between 0 and 1. In this function, mutual information is normalized by 
sqrt(H(labels_true)*H*(labels_pred))

'''
class expEval(object):
    __metaclass__ = ABCMeta
    
    def __init__(self,name,desc=""):
        self.name = name
        self.desc = desc
        self.value = []
    
    @abstractmethod
    def evaluate(self,true_groups,est_groups):
        pass


class expEvalNmiScore(expEval):
    name = "NMIScore"
    desc = "normalized_mutual_info_score"
    _type = "cluster"
    
    def __init__(self):
        super(expEvalNmiScore,self).__init__('NMIScore','normalized_mutual_info_score')
        
    def evaluate(self,true_groups,est_groups):
        if true_groups.shape!=est_groups.shape:
            print 'evaluation[NMIScore]:dim mismatch of label and prelabel'
            return
        value = nmi_score(true_groups,est_groups)
        self.value.append(value)#append the lastest value to the value list

########################################################################################
class expEvalHomoScore(expEval):
    name = "HomoScore"
    desc = "homogeneity score"
    _type = "cluster"
    
    def __init__(self):
        super(expEvalHomoScore,self).__init__('HomoScore','homogeneity score')
    
    def evaluate(self,true_groups,est_groups):
        if true_groups.shape != est_groups.shape:
            print 'evaluation[HomoScore]: dim mismatch of label and prelabel'
            return
        value = homogeneity_score(true_groups,est_groups)
        self.value.append(value)

########################################################################################
class expEvalPurityScore(expEval):
    name = "PurityScore"
    desc = "purity score"
    _type = "cluster"
    
    def __init__(self):
        super(expEvalPurityScore,self).__init__("PurityScore","purity score")
    
    def evaluate(self,true_groups,est_groups):
        if true_groups.shape != est_groups.shape:
            print 'evaluation[HomoScore]: dim mismatch of label and prelabel'
            return
        value = purity_score(true_groups,est_groups)
        self.value.append(value)
        
######################################################################################

class expEvalEdgeAccuracy(expEval):
    name = "EdgeAccuracy"
    desc = "Edge Accuracy"
    _type = "graph"
    
    def __init__(self):
        super(expEvalEdgeAccuracy,self).__init__('EdgeAccuracy','Edge Accuracy')
    
    def evaluate(self,prec,est_prec):
        if prec.shape != est_prec.shape:
            print 'evaluation[EdgeAccuracy]: dim mismatch of label and prelabel'
            return
        threshold = np.arange(0,.05,.001)
        value = []
        for i in xrange(len(threshold)):
            value.append(edge_accuracy(prec,est_prec,threshold=threshold[i]))
        self.value.append(max(value))

######################################################################################

class expEvalEdgeRecall(expEval):
    name = "EdgeRecall"
    desc = "Edge Recall"
    _type = "graph"
    
    def __init__(self):
        super(expEvalEdgeRecall,self).__init__('EdgeRecall','Edge Recall')
    
    def evaluate(self,prec,est_prec):
        if prec.shape != est_prec.shape:
            print 'evaluation[EdgeRecall]: dim mismatch of label and prelabel'
            return
        threshold = np.arange(0,.05,.001)
        value = []
        for i in xrange(len(threshold)):
            value.append(edge_recall(prec,est_prec,threshold=threshold[i]))
        
        self.value.append(max(value))
        
######################################################################################

class expEvalEdgePrec(expEval):
    name = "EdgePrecision"
    desc = "edge Precision"
    _type = "graph"
    
    def __init__(self):
        super(expEvalEdgePrec,self).__init__('EdgePrecision','edge Precision')
    
    def evaluate(self,prec,est_prec):
        if prec.shape != est_prec.shape:
            print 'evaluate[EdgePrecision]: dim mismatch of label and prelabel'
            return
        threshold = np.arange(0,.05,.001)
        value = []
        for i in xrange(len(threshold)):
            value.append(edge_prec(prec,est_prec,threshold=threshold[i]))
        
        self.value.append(max(value))

######################################################################################

class expEvalEdgeF1(expEval):
    name = "EdgeF1"
    desc = "Edge F1"
    _type = "graph"
    
    def __init__(self):
        super(expEvalEdgeF1,self).__init__('EdgeF1','Edge F1')
    
    def evaluate(self,prec,est_prec):
        if prec.shape != est_prec.shape:
            print 'evaluate[EdgeF1]: dim mismatch of label and prelabel'
            return
        threshold = np.arange(0,.05,.001)
        value = []
        for i in xrange(len(threshold)):
            value.append(edge_F1(prec,est_prec,threshold=threshold[i]))
        
        self.value.append(max(value))

######################################################################################

#class expEvalEdgeAccuracy_inter(expEval)


