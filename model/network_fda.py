import torch
import torch.nn as nn
import numpy as np
from torch.nn.functional import normalize
import torch.nn.functional as F

class LayerNorm(nn.Module):

    def __init__(self, d, eps=1e-6):
        super().__init__()
        # d is the normalization dimension
        self.d = d
        self.eps = eps
        self.alpha = nn.Parameter(torch.randn(d))
        self.beta = nn.Parameter(torch.randn(d))

    def forward(self, x):
        # x is a torch.Tensor
        # avg is the mean value of a layer
        avg = x.mean(dim=-1, keepdim=True)
        # std is the standard deviation of a layer (eps is added to prevent dividing by zero)
        std = x.std(dim=-1, keepdim=True) + self.eps
        return (x - avg) / std * self.alpha + self.beta


class FeedForward(nn.Module):

    def __init__(self, in_d=1, hidden=[4,4,4], dropout=0.1, activation=F.relu):
        # in_d      : input dimension, integer
        # hidden    : hidden layer dimension, array of integers
        # dropout   : dropout probability, a float between 0.0 and 1.0
        # activation: activation function at each layer
        super().__init__()
        self.sigma = activation
        dim = [in_d] + hidden + [1]
        self.layers = nn.ModuleList([nn.Linear(dim[i-1], dim[i]) for i in range(1, len(dim))])
        self.ln = nn.ModuleList([LayerNorm(k) for k in hidden])
        self.dp = nn.ModuleList([nn.Dropout(dropout) for _ in range(len(hidden))])

    def forward(self, t):
        for i in range(len(self.layers)-1):
            t = self.layers[i](t)
            # skipping connection
            t = t + self.ln[i](t)
            t = self.sigma(t)
            # apply dropout
            t = self.dp[i](t)
        # linear activation at the last layer
        return self.layers[-1](t)

def _inner_product(f1, f2, h):
    """    
    f1 - (B, J) : B functions, observed at J time points,
    f2 - (B, J) : same as f1
    h  - (J-1,1): weights used in the trapezoidal rule
    pay attention to dimension
    <f1, f2> = sum (h/2) (f1(t{j}) + f2(t{j+1}))
    """
    prod = f1 * f2 # (B, J = len(h) + 1)
    return torch.matmul((prod[:, :-1] + prod[:, 1:]), h.unsqueeze(dim=-1))/2

def _l1(f, h):
    # f dimension : ( B bases, J )
    B, J = f.size()
    return _inner_product(torch.abs(f), torch.ones((B, J)), h)

def _l2(f, h):
    # f dimension : ( B bases, J )
    # output dimension - ( B bases, 1 )
    return torch.sqrt(_inner_product(f, f, h))

class FDProcessor(nn.Module):

    def __init__(self, n_base=4, base_hidden=[32, 32],
                 dropout=0.1, lambda1=0.0, lambda2=0.0,
                 device=None):
        """
        n_base      : number of basis nodes, integer
        base_hidden : hidden layers used in each basis node, array of integers
        grid        : observation time grid, array of sorted floats including 0.0 and 1.0
        sub_hidden  : hidden layers in the subsequent network, array of integers
        dropout     : dropout probability
        lambda1     : penalty of L1 regularization, a positive real number
        lambda2     : penalty of L2 regularization, a positive real number
        device      : device for the training
        """
        super().__init__()
        self.n_base = n_base
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.device = device
        # instantiate each basis node in the basis layer
        self.BL = nn.ModuleList([FeedForward(1, hidden=base_hidden, dropout=dropout, activation=F.selu)
                                 for _ in range(n_base)])
        # instantiate the subsequent network
        #self.FF = FeedForward(n_base, sub_hidden, dropout)

    def forward(self, x, grid):
        B, J = x.size()
        # grid should include both end points
        grid = np.array(grid)
        # send the time grid tensor to device
        self.t = torch.tensor(grid).float().to(x.device)
        self.h = torch.tensor(grid[1:] - grid[:-1]).float().to(x.device)
        assert J == self.h.size()[0] + 1
        T = self.t.unsqueeze(dim=-1)
        # evaluate the current basis nodes at time grid
        self.bases = [basis(T).transpose(-1, -2) for basis in self.BL]
        """
        compute each basis node's L2 norm
        normalize basis nodes
        """
        l2_norm = _l2(torch.cat(self.bases, dim=0), self.h).detach()
        self.normalized_bases = [self.bases[i] / (l2_norm[i, 0] + 1e-6) for i in range(self.n_base)]
        # compute each score <basis_i, f> 
        score = torch.cat([_inner_product(b.repeat((B, 1)), x, self.h) # (B, 1)
                           for b in self.bases], dim=-1) # score dim = (B, n_base)
        # take the tensor of scores into the subsequent network
        #out = self.FF(score)
        return score

    def R1(self, l1_k):
        """
        L1 regularization
        l1_k : number of basis nodes to regularize, integer        
        """
        if self.lambda1 == 0: return torch.zeros(1).to(self.device)
        # sample l1_k basis nodes to regularize
        selected = np.random.choice(self.n_base, min(l1_k, self.n_base), replace=False)
        selected_bases = torch.cat([self.normalized_bases[i] for i in selected], dim=0) # (k, J)
        return self.lambda1 * torch.mean(_l1(selected_bases, self.h))

    def R2(self, l2_pairs):
        """
        L2 regularization
        l2_pairs : number of pairs to regularize, integer  
        """
        if self.lambda2 == 0 or self.n_base == 1: return torch.zeros(1).to(self.device)
        k = min(l2_pairs, self.n_base * (self.n_base - 1) // 2)
        f1, f2 = [None] * k, [None] * k
        for i in range(k):
            a, b = np.random.choice(self.n_base, 2, replace=False)
            f1[i], f2[i] = self.normalized_bases[a], self.normalized_bases[b]
        return self.lambda2 * torch.mean(torch.abs(_inner_product(torch.cat(f1, dim=0),
                                                                  torch.cat(f2, dim=0),
                                                                  self.h)))


class casue_specific_hazard(nn.Module):
    def __init__(self,):
        super(casue_specific_hazard, self).__init__()
    
    def forward(self, x):
        b, v = x.shape
        x = x.reshape(b,-1,2)
        numerator = torch.exp(x)
        denominator = 1 + torch.sum(torch.exp(x), dim=-1, keepdim=True)
        return numerator / denominator

class StoNet_Survival_FDA(nn.Module):
    def __init__(self, num_hidden, hidden_dim, input_dim, output_dim, n_event,n_base,max_interval, view,
                 miss_col=None, obs_ind_node=None, miss_pattern=None):
        
        super(StoNet_Survival_FDA, self).__init__()
        self.num_hidden = num_hidden
        self.miss_col = miss_col
        self.miss_pattern = miss_pattern
        self.obs_ind_node = obs_ind_node
        self.n_event = n_event
        self.n_base = n_base
        self.max_interval = max_interval
        self.view = view
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.fd_processor = FDProcessor(n_base=n_base)
        self.module_list = []

        self.module_list.append(nn.Linear(input_dim+self.n_base*self.view, hidden_dim[0]))
        self.add_module(str(0), self.module_list[0])


        for i in range(self.num_hidden - 1):
            self.module_list.append(nn.Sequential(nn.ReLU(),
                                                  nn.Linear(hidden_dim[i], hidden_dim[i + 1])))
            self.add_module(str(i+1), self.module_list[i+1])

        self.module_list.append(nn.Sequential(nn.ReLU(),
                                              #nn.Linear(hidden_dim[-1], hidden_dim[-1]),
                                              nn.Linear(hidden_dim[-1], output_dim),
                                              casue_specific_hazard()
                                              ))

        self.add_module(str(self.num_hidden), self.module_list[self.num_hidden])

    def forward(self, input):
        regs1 = []
        regs2= []
        hid2 = []
        x1 = input[0]
        x2 = input[1]
        for v in range(self.view):
            x_fda = x2[:,:,v]
            TT = x_fda.size(-1)
            grid = torch.linspace(0,1,TT)
            #hid1 = self.module_list[0](x1)
            hid_fda = self.fd_processor.forward(x_fda, grid).unsqueeze(1)
            hid_fda = hid_fda.repeat(1,self.max_interval,1)
            hid_fda = hid_fda.reshape(-1, self.n_base)
            reg1 = self.fd_processor.R1(2).to(self.device)
            reg2 = self.fd_processor.R2(3).to(self.device)
            regs1.append(reg1)
            regs2.append(reg2)
            hid2.append(hid_fda)
        hid2 = torch.cat(hid2, dim=1)
        hid = torch.cat((x1,hid2),dim=-1)
        for layer_index in range(self.num_hidden+1):
            hid = self.module_list[layer_index](hid)
        r1 = sum(regs1)
        r2 = sum(regs2)
        return hid, r1, r2

class StoNet_Survival_FDA_sub(nn.Module):
    def __init__(self, num_hidden, hidden_dim, input_dim, output_dim, n_event,n_base,max_interval,view,
                 miss_col=None, obs_ind_node=None, miss_pattern=None):

        super(StoNet_Survival_FDA_sub, self).__init__()
        self.num_hidden = num_hidden
        self.miss_col = miss_col
        self.miss_pattern = miss_pattern
        self.obs_ind_node = obs_ind_node
        self.n_event = n_event
        self.n_base = n_base
        self.max_interval = max_interval
        self.view = view
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.fd_processor = FDProcessor(n_base=n_base)
        self.module_list = []

        self.module_list.append(nn.Linear(input_dim+self.n_base*self.view, hidden_dim[0]))
        self.add_module(str(0), self.module_list[0])

        for i in range(self.num_hidden - 1):
            self.module_list.append(nn.Sequential(nn.ReLU(),
                                                  nn.Linear(hidden_dim[i], hidden_dim[i + 1])))
            self.add_module(str(i+1), self.module_list[i+1])


        self.module_list.append(nn.Sequential(nn.ReLU(),
                                              nn.Linear(hidden_dim[-1], output_dim),
                                              nn.Sigmoid(),
                                              ))

        self.add_module(str(self.num_hidden), self.module_list[self.num_hidden])

    def forward(self, input):
        regs1 = []
        regs2= []
        hid2 = []
        x1 = input[0]
        x2 = input[1]
        for v in range(self.view):
            x_fda = x2[:,:,v]
            TT = x_fda.size(-1)
            grid = torch.linspace(0,1,TT)
            #hid1 = self.module_list[0](x1)
            hid_fda = self.fd_processor.forward(x_fda, grid).unsqueeze(1)
            hid_fda = hid_fda.repeat(1,self.max_interval,1)
            hid_fda = hid_fda.reshape(-1, self.n_base)
            reg1 = self.fd_processor.R1(2).to(self.device)
            reg2 = self.fd_processor.R2(3).to(self.device)
            regs1.append(reg1)
            regs2.append(reg2)
            hid2.append(hid_fda)
        hid2 = torch.cat(hid2, dim=1)
        hid = torch.cat((x1,hid2),dim=-1)
        for layer_index in range(self.num_hidden+1):
            hid = self.module_list[layer_index](hid)
        r1 = sum(regs1)
        r2 = sum(regs2)
        return hid, r1, r2
