import torch
import torch.nn as nn
import torch.nn.functional as F
import copy


class TasteNetChoice(nn.Module):
    """
    A TasteNet-MNL for A Synthetic Dataset
    """

    def __init__(self, args):
        super(TasteNetChoice, self).__init__()
        if args.separate: # two tastes coefficients are learned through 2 seperate TasteNets (instead of one TasteNet)
            self.taste = TasteNetCombo(args)
        else:
            self.taste = TasteNet(args)
        self.util = Utility(args)
        self.args = args

    def forward(self, z, x):
        b = self.taste(z)  # z: (N, D); b: (N, K) taste coefficients for K attributes (no ASC)
        v = self.util(x, b)  # x: (N, K+1, J), b: (N, K), v: (N, J)
        return v

    def getParams(self):
        """
        Get a list of the network parameters
        """
        params_list = []
        for params in self.parameters():
            params_list.append(params)
        return copy.deepcopy(params_list)

    def getCoefBias(self):
        """
        Get coefficients and bias for each part of the network
        """
        count = 0
        bias = []
        coef = []
        for params in self.parameters():
            if count % 2 == 0:
                coef.append(params)
            else:
                bias.append(params)
            count += 1
        return coef, bias


class TasteNet(nn.Module):
    """TasteNet part of the TasteNet-MNL"""
    def __init__(self, args):
        super(TasteNet, self).__init__()
        self.seq = nn.Sequential()
        for i, (in_size, out_size) in enumerate(zip(args.layer_sizes[:-1], args.layer_sizes[1:])):
            self.seq.add_module(name="L%i" % (i + 1), module=nn.Linear(in_size, out_size, bias=True))
            if i < len(args.layer_sizes) - 2:
                self.seq.add_module(name="A%i" % (i + 1), module=get_act(args.activation))
        self.args = args

    def forward(self, z):
        """
        Parameters:
            individual characteristics z: (N,D), N: batch size, D: number of characteristics
        Returns:
            taste coefficients b: (N,K-1), N: batch size, K: number of taste coefficients to learn. Cost coef is fixed to -1.
        """
        if self.args.transform == "exp":
            return -torch.exp(-1 * self.seq(z))  # (N,1)
        elif self.args.transform == "relu":
            return -F.relu(-self.seq(z))
        else:  # no transform
            return self.seq(z)


class TasteNetCombo(nn.Module):
    """
    Combine two TasteNets
    """
    def __init__(self, args):
        super(TasteNetCombo, self).__init__()
        self.taste_time = TasteNet(args)
        self.taste_wait = TasteNet(args)

    def forward(self, z):
        b_time = self.taste_time(z)
        b_wait = self.taste_wait(z)
        b = torch.cat([b_time.view(-1, 1), b_wait.view(-1, 1)], dim=1)  # (N,2)
        return b


class Utility(nn.Module):
    """MNL part of TasteNet-MNL"""

    def __init__(self):
        super(Utility, self).__init__()
        self.linear = nn.Linear(1, 1, bias=False)  # ASCs for J-1 alternatives (parameters to learn)

    def forward(self, x, b):
        """
        x: (N, K, J):
            K = 3: cost, time, wait
            J = 2: 2 alternatives 
        b: (N, K-1):
            b[:, 0] = b_time
            b[:, 1] = b_wait
            (b_cost fixed to -1)
        """
        N, _, _ = x.size()

        asc0 = torch.zeros(N, 1)  # asc0 fixed to 0
        asc1 = self.linear(torch.ones(N, 1))  # learn asc1 by the self.linear (1->1)

        b_cost = -torch.ones(N, 1)  # fixed to -1, add this constant
        b_time = b[:, 0].view(N, 1)  # (N,1)
        b_wait = b[:, 1].view(N, 1)  # (N,1)

        v = x[:, 0, :] * b_cost + x[:, 1, :] * b_time + x[:, 2, :] * b_wait + torch.cat((asc0, asc1), dim=1)
        return v  # (N,J)


def get_act(activation):
    if activation == "tanh":
        return nn.Tanh()
    elif activation == "relu":
        return nn.ReLU()
    else:
        return None