import pickle
import os

class expConfig(object):
    def __init__(self,dataset,setting,model,evals,skip_if_file_exist=True):
        self.dataset = dataset
        self.setting = setting
        self.model = model
        self.evals = evals
        self.skip_if_file_exist = True
    
    def run(self):
        if self.skip_if_file_exist and os.path.isfile(self.result_path):#self.result_path
            print "file exist,experiment skipped"
            return
        self.setting.setup(dataset=self.dataset,
                          model=self.model,
                          evals=self.evals)
        self.setting.evaluate()
        self.save_result(self.setting)
    
    def save_result(self,result):
        #result.model.clean()
        folder = os.path.dirname(self.result_path)
        if not os.path.exists(folder):
            os.makedirs(folder)
        with open(self.result_path,'wb') as out_file:
            pickle.dump(result,out_file,pickle.HIGHEST_PROTOCOL)
        
    @property#fix attribute
    def result_path(self):
        return os.path.join('results',
                           self.setting_dir,
                           self.dataset_dir,
                           self.model_dir,
                           'result.pkl')
    
    @property
    def setting_dir(self):
        return self.attr_to_str(self.setting,'setting',['n_rounds'])
    
    @property
    def dataset_dir(self):
        return self.attr_to_str(self.dataset,'dataset',['n_sample','P','L'])
    
    @property
    def model_dir(self):
        return self.attr_to_str(self.model,'model',[])
    
    def attr_to_str(self,obj,name,involve_list):
        out_dir = name + "-" + obj.name
        for k,v in obj.__dict__.items():
            if not k.startswith('_') and k in involve_list:
                out_dir += "/%s-%s" % (k,",".join([str(i) for i in v]) if type(v) == list else v)
        return out_dir
