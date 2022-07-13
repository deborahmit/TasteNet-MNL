import torch
import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F

def get_act(nl_func):
    if nl_func=="tanh":
        return nn.Tanh()
    elif nl_func == "relu":
        return nn.ReLU()
    elif nl_func == "sigmoid":
        return nn.Sigmoid()
    else:
        return None
    
class ChoiceFlex(nn.Module):
    """TasteNet-MNL model for Swissmetro"""
    def __init__(self, args):
        super(ChoiceFlex, self).__init__()
        self.params_module = TasteParams(args.layer_sizes, args)
        self.util_module = Utility(args)
        self.args = args
    
    def forward(self, z, x, av):
        b = self.params_module(z) # taste parameters, (N,8)
        b = self.constraints(b)  ## this is another way to include constraint:  using transformation to include constraints 
        v = self.util_module(x,b) #no softmax here 
        exp_v = torch.exp(v)
        exp_v_av = exp_v * av
        
        prob = exp_v_av/exp_v_av.sum(dim=1).view(-1,1) # prob (N,J)
        
        return prob, None  
    
    def constraints(self,b):
        '''
            Put transformation for the sake of constraints on the value of times 
        '''
        if self.args.transform=='relu':
            return torch.cat([-F.relu(-b[:,:-3]),b[:,-3:]],dim=1)
        elif self.args.transform == 'exp':
            return torch.cat([-torch.exp(-self.args.mu * b[:,:-3]),b[:,-3:]],dim=1) # the last 3 dim of b are under constraints
        else:
            return b
    
    def getParameters(self):
        '''
        get coef and bias of the TasteParams of the model 
        '''
        count = 0
        bias = []
        coef = []
        for params in self.parameters():
            if count % 2==0:
                coef.append(params)
            else:
                bias.append(params)
            count += 1
        return coef, bias
    
    def L2Norm(self):
        '''
        L2 norm, not including bias
        '''
        coef, bias = self.getParameters()
        norm = torch.zeros(1)
        for params in coef:
            norm += (params**2).sum()
        return norm            

    def L1Norm(self):
        '''
        L1 norm, not including bias
        '''
        coef, bias = self.getParameters()
        norm = torch.zeros(1)
        for params in coef:
            norm += (torch.abs(params).sum())
        return norm

class Utility(nn.Module):
    def __init__(self, args):
        super(Utility, self).__init__()
        self.args = args
        self.index = OrderedDict(zip(['TRAIN_TT', 'SM_TT', 'CAR_TT', 'TRAIN_HE', 'SM_HE', 'SM_SEATS', 'TRAIN_ASC', 'SM_ASC'], range(8)))

        
    def forward(self, x, b):
        '''
        x: attributes of each alternative, 
           including the intercept (N,K+1,J)  J alternatives, each have K+1 attributes including 1 for intercept. 
        b: taste parameters (K+1,J)  Some paramters are constant, some come from neural network hidden layer.  
        '''
        index = self.index
        N = len(b)        
        # last hidden nodes correspond to b_names
        v = torch.zeros(N,3)        
        v[:,0] = torch.ones(N) * b[:,index["TRAIN_ASC"]] + x["TRAIN"]["TRAIN_TT"]*b[:,index["TRAIN_TT"]] + x["TRAIN"]["TRAIN_HE"]*b[:,index["TRAIN_HE"]] - x["TRAIN"]["TRAIN_CO"]
        v[:,1] = torch.ones(N) * b[:,index["SM_ASC"]] + x["SM"]["SM_TT"]*b[:,index["SM_TT"]] + x["SM"]["SM_HE"]*b[:,index["SM_HE"]] + x["SM"]["SM_SEATS"]*b[:,index["SM_SEATS"]] - x["SM"]["SM_CO"]
        v[:,2] = x["CAR"]["CAR_TT"]*b[:,index["CAR_TT"]] - x["CAR"]["CAR_CO"]
        
        return v

class TasteParams(nn.Module):
    '''
    Network for tastes
    '''
    def __init__(self, layer_sizes, args):
        super(TasteParams, self).__init__()
        self.seq = nn.Sequential()
        for i, (in_size, out_size) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            self.seq.add_module(name="L%i"%(i+1), module=nn.Linear(in_size, out_size, bias=True))
            if i<len(layer_sizes)-2:
                self.seq.add_module(name="A%i"%(i+1), module=get_act(args.act_func))
        self.args = args
        
    def forward(self,z):
        '''
        Parameters:
            z: (N,D) # batch size, input dimension
        Returns:
            V: (N,K) # taste parameters 
        '''
        N,D = z.size()
        return self.seq(z) # (N,K) 
