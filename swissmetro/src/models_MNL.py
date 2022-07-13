import torch
import torch.nn as nn


class ChoiceSwissMetro(nn.Module):
    '''
    Feed-forward network without hidden layer, similar to choice model without sharing weights among alternatives
    '''
    def __init__(self, args):
        super(ChoiceSwissMetro, self).__init__()
        self.choice_set = args.choice_set
        self.choices = nn.ModuleDict()
        for alt in self.choice_set:
            self.choices.update({alt:FFW(args.K[alt])})
        self.args = args
        
    def forward(self, x, av):
        '''
        Parameters:
            x: (N,4,3) # batch size, input dimension, num of alternatives
            av: availability (N,3)
        Returns:
            prob: (N,3) # choice probability
        '''
        # Use generic utility function (input are already computed as x[alternative] for alternative 
        v = torch.zeros(len(av),len(self.choice_set))
        col = 0
        for alt in self.choice_set:
            v[:,col] = (self.choices[alt].forward(x[alt][:,1:])-x[alt][:,0].view(-1,1)).view(-1) # flatten out the result 
            col += 1
        
        exp_v = torch.exp(v)
        exp_v_av = exp_v * av
        return exp_v_av/exp_v_av.sum(dim=1).view(-1,1)  # prob (N,J)

class FFW(nn.Module):
    '''
    FFW for each utility   
    '''
    def __init__(self, K):
        super(FFW, self).__init__()
        self.linear = nn.Linear(K,1, False)  # bias is included already!! 
    
    def forward(self,x):
        return self.linear(x)
