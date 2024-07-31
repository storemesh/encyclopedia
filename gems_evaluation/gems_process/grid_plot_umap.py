from umap import UMAP
import string
import pandas as pd
import random
from tqdm.auto import tqdm

class UmapOptimization : 

    def __init__(self, pretrain=None):
        self.pretrain = pretrain
        
    def get_params(self, model):
        desired_params = ['n_neighbors', 'min_dist', 'spread', 'random_state']
        params = model.get_params()
        dict_result = {param: params.get(param, None) for param in desired_params}    
        return dict_result

    def generate_uid(self, size, chars=string.ascii_uppercase + string.digits):
        return ''.join(random.choice(chars) for x in range(size))

    def generate_random_state(self,):
        return int(''.join(map(str, [random.randint(0,9) for i in range(6)])))
        
    def evaluate_model(self, vector, params_dict):
        
        if self.pretrain is not None : 
            return self.pretrain.fit_transform(vector)
        else : 
            data_tmp = {}
            model = UMAP(**params_dict)
            model.fit(vector)
            return model
        
    def generate_params(self):
        list_of_search = []
        n_neighbors_list = self.params.get('n_neighbors')
        min_dist_list = self.params.get('min_dist')
        spread_list = self.params.get('spread')
        for i, n_neighbors in enumerate(n_neighbors_list):
            for j, min_dist in enumerate(min_dist_list):
                for k, spread in enumerate(spread_list):
                    random_state = self.generate_random_state()
                    list_of_search.append({'n_neighbors':n_neighbors, 
                                           'min_dist':min_dist, 
                                           'spread':spread, 
                                           'random_state':random_state,
                                          }
                                         )
        return list_of_search

    def input(self, params, vector) : 
        self.params = params
        self.vector = vector
            
    def output(self,):
        result = []
        grid_size = 800
        list_of_search = self.generate_params()
        for index, sub_params_ in enumerate(tqdm(list_of_search)):
            rs  = self.evaluate_model(self.vector, sub_params_)
            result.append(rs)
        return result