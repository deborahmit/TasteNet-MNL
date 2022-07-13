import pickle
from torch.utils.data import Dataset
import torch
import numpy as np 

class ChoiceDataset(Dataset):
    def __init__(self, data_path, data_file):
        """
        Parameters:
            data_file (string): name of data pickle file
        """
        data = pickle.load(open(data_path + "/" + data_file, "rb"))
        self.x = torch.Tensor(data["x"])
        self.x_names = data["x_names"]
        self.N = len(self.x)
        
        dic_attr = \
        {"TRAIN": ["TRAIN_TT", "TRAIN_HE", "TRAIN_CO"], \
         "SM": ["SM_TT", "SM_HE", "SM_CO", "SM_SEATS"],\
         "CAR": ["CAR_TT", "CAR_CO"]}
        
        self.x_dict = {}
        for mode in dic_attr:
            self.x_dict[mode] = {}
            for attr in dic_attr[mode]:
                self.x_dict[mode].update({attr: getAttribute(self.x, self.x_names, attr)})
        
        self.y = torch.LongTensor(data["y"])-1 # N
        
        # Availability 
        self.av = torch.cat([torch.ones(self.N,2),torch.Tensor(data["car_av"]).view(self.N,1)], dim=1) # (N,3) av for all modes 
        
        # all z 
        self.z_all_names = data['z_names']
        self.z_levels = data['z_levels']
        self.z_all = torch.Tensor(data['z']) # N,D socio-demo variables

        # select z
        self.z_names = ["MALE_1", "AGE_1", "AGE_2", "AGE_3", "AGE_4", \
               "INCOME_1", "INCOME_2", "INCOME_3", "FIRST_1", "WHO_1", "WHO_2", \
               "PURPOSE_1", "PURPOSE_2", "PURPOSE_3", "LUGGAGE_1", "LUGGAGE_2", "GA_1"]
        self.z = selectZ(self.z_all, self.z_names, self.z_all_names)
        
        _, self.D = self.z.size() # z size = (N,D)

        
    def __len__(self):
        return self.N
    
    def __getitem__(self, idx):
        '''
        Get the sample given its idx in the list 
        '''
        x = {}
        for mode in self.x_dict:
            x[mode] = {}
            for name in self.x_dict[mode]:
                x[mode][name] = self.x_dict[mode][name][idx]
        return {"x": x, "y": self.y[idx], "z":self.z[idx], "av": self.av[idx]}
    
def getAttribute(x, x_names, name):
    return x[:,x_names.index(name)]
    
def selectZ(z,z_selected, z_names):
    ind = []
    for var in z_selected:
        ind.append(z_names.index(var))
    return z[:,np.array(ind).astype(int)]