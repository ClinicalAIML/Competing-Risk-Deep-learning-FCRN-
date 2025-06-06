import torch
import torch.nn as nn
import numpy as np


class casue_specific_hazard(nn.Module):
    def __init__(self,):
        super(casue_specific_hazard, self).__init__()
    
    def forward(self, x):
        b, v = x.shape
        x = x.reshape(b,-1,2)
        numerator = torch.exp(x)
        denominator = 1 + torch.sum(torch.exp(x), dim=-1, keepdim=True)
        return numerator / denominator


class StoNet_Survival(nn.Module):
    def __init__(self, num_hidden, hidden_dim, input_dim, output_dim, n_event,
                 miss_col=None, obs_ind_node=None, miss_pattern=None):

        super(StoNet_Survival, self).__init__()
        self.num_hidden = num_hidden
        self.miss_col = miss_col
        self.miss_pattern = miss_pattern
        self.obs_ind_node = obs_ind_node
        self.n_event = n_event
        self.module_list = []

        self.module_list.append(nn.Linear(input_dim, hidden_dim[0]))
        self.add_module(str(0), self.module_list[0])

        for i in range(self.num_hidden - 1):
            self.module_list.append(nn.Sequential(nn.ReLU(),
                                                  nn.Linear(hidden_dim[i], hidden_dim[i + 1])))
            self.add_module(str(i+1), self.module_list[i+1])

        self.module_list.append(nn.Sequential(nn.ReLU(),
                                              nn.Linear(hidden_dim[-1], output_dim),
                                              casue_specific_hazard()
                                              ))

        self.add_module(str(self.num_hidden), self.module_list[self.num_hidden])


        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, x):
        for layer_index in range(self.num_hidden+1):
            x = self.module_list[layer_index](x)
        return x


class StoNet_Survival_sub(nn.Module):
    def __init__(self, num_hidden, hidden_dim, input_dim, output_dim, n_event,
                 miss_col=None, obs_ind_node=None, miss_pattern=None):

        super(StoNet_Survival_sub, self).__init__()
        self.num_hidden = num_hidden
        self.miss_col = miss_col
        self.miss_pattern = miss_pattern
        self.obs_ind_node = obs_ind_node
        self.n_event = n_event
        self.module_list = []

        self.module_list.append(nn.Linear(input_dim, hidden_dim[0]))
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


        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, x):
        for layer_index in range(self.num_hidden+1):
            x = self.module_list[layer_index](x)
        return x

