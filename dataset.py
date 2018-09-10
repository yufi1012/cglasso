from abc import ABCMeta, abstractmethod
from sklearn.datasets import make_classification

class dataset():
    __metaclass__ = ABCMeta

    def __init__(self, name, desc=""):
        self.name = name
        self.desc = desc 

    @abstractmethod
    def load(self):
        return

class Toy(dataset):
    
    def __init__(self,n_samples=200,n_feature=2,n_classes=3,random_state=2412):
        super(Toy,self).__init__('toy','toy dataset')
        self.n_samples=n_samples
        self.n_feature=n_feature
        self.n_classes=n_classes
        self.random_state=random_state

    def load(self):
        X,y = make_classification( n_samples=self.n_samples, 
                                    n_features=self.n_feature,
                                    n_classes=self.n_classes,
                                    random_state=self.random_state,
                                    n_redundant=0, 
                                    n_informative=2,
                                    n_clusters_per_class=1)
        
        return X[::2], y[::2], X[1::2], y[1::2]
