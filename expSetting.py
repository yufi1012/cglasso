from abc import ABCMeta, abstractmethod
from expDataset import sample
import numpy as np

'''
Test Setting Class
run exp for one/multiple rounds
clustering model - ouptput is clusters
graphical model - output is precision matrix
coherent model - output is clusters + precision matrix
'''
class expSetting(object):
    __metaclass__ = ABCMeta
    
    def __init__(self):
        self.timeTotal = []
        self.timeTrain = []
        self.timeTest = []
    
    #def setModel(self,classifier):
    #    self.model = model
    
    #def setEval(self,evalMethods):
    #    self.evals = evalMethods
    
    #def update_abstract(self,abstract):
    #    self.abstract.update(abstract)
    
    def setup(self,dataset,model,evals):
        self.model = model
        self.dataset = dataset
        self.evals = evals
    
    @abstractmethod
    def evaluate(self):
        return
    @property
    def time_train(self):
        return self.timeTrain
    @time_train.setter
    def time_train(self,value):
        self.timeTrain.append(value)
    @property
    def time_test(self):
        return self.timeTest
    @time_test.setter
    def time_test(self):
        self.timeTest.append(value)


##############################################################################################
class expSettingTest(expSetting):
    name = "SettingTest"
    desc = "SettingTest"
    
    def __init__(self,n_rounds=1,random_seed=1991):
        super(expSettingTest,self).__init__()
        self.n_rounds = n_rounds
        self.setting_seed = random_seed
    
    def evaluate(self):
        data = self.dataset.load()
        prec, group = data.prec, data.groups#data from Synthetic Dataset.sample?
        n_features, n_samples = prec.shape[0], data.n_samples
        evals, n_evals = self.evals, len(self.evals)
        sigma = np.linalg.inv(prec)
        random_state = np.random.RandomState(self.setting_seed)
        for i in xrange(self.n_rounds):
            S, X = sample(sigma,n_features,n_samples,random_state=random_state)
            category = self.model.category#what is the model.category
            est_prec, est_group = None, None
            if category == "graph":#interesting
                est_prec = self.model.predict(X)
            elif category == "cluster":
                est_group = self.model.predict(X)
            elif category == "coherent":
                est_prec,est_group = self.model.predict(X)
            
            self.timeTrain = 0#?
            self.timeTest = 0#?
            self.timeTotal = self.model.timeTotal
            
            for i in xrange(n_evals):
                if evals[i]._type == "cluster" and est_group is not None:
                    evals[i].evaluate(group,est_group)
                if evals[i]._type == "graph" and est_prec is not None:
                    evals[i].evaluate(prec,est_prec)

#output of this settings?????
##############################################################################################
class expSettingTest_NMFGL(expSetting):
    name = "SettingTest"
    desc = "SettingTest"
    
    def __init__(self,n_rounds=1,random_seed=1991):
        super(expSettingTest_NMFGL,self).__init__()
        self.n_rounds = n_rounds
        self.setting_seed = random_seed
    
    def evaluate(self):
        data = self.dataset.load()
        prec, inter_matrix, group = data.prec, data.inter_matrix, data.groups
        n_features, n_samples = prec.shape[0], data.n_samples
        evals, n_evals = self.evals, len(self.evals)
        sigma = np.linalg.inv(prec)
        random_state = np.random.RandomState(self.setting_seed)
        for i in xrange(self.n_rounds):
            S, X = sample(sigma,n_features,n_samples,random_state=random_state)
            category = self.model.category#what is the model.category
            est_prec, est_group = None, None
            if category == "graph":#interesting
                est_prec = self.model.predict(X)
            elif category == "cluster":
                est_group = self.model.predict(X)
            elif category == "coherent":
                est_group,est_prec,est_globprec = self.model.predict(X)[0],self.model.predict(X)[1],self.model.predict(X)[3]
                
            
            self.timeTrain = 0#?
            self.timeTest = 0#?
            self.timeTotal = self.model.timeTotal
            
            for i in xrange(n_evals):
                if evals[i]._type == "cluster" and est_group is not None:
                    evals[i].evaluate(group,est_group)
                if evals[i]._type == "graph" and est_prec is not None:
                    evals[i].evaluate(prec,est_globprec)

        