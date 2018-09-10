import os
import pickle


class result:
    def __init__(self, root_dir):
        self.path = root_dir

    def load(self):
        results=[]
        for (dirpath, dirnames, filenames) in os.walk(self.path):
            for f in filenames:
                filepath = dirpath +'/'+ f
                r = pickle.load(open(filepath,"rb"))
                results.append(r)
        return results
