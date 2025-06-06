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
                 miss_col=None, obs_ind_node=None, miss_pattern=None, graph=None):
        
        super(StoNet_Survival, self).__init__()
        self.num_hidden = num_hidden
        self.miss_col = miss_col
        self.miss_pattern = miss_pattern
        self.obs_ind_node = obs_ind_node
        self.n_event = n_event
        #self.graph = graph
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

        self.sse = nn.MSELoss(reduction='sum')

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    def likelihood_miss(self, x_impute, graph):
        likelihoods = []
        for i in range(len(self.miss_col)):
            with torch.no_grad():
                graph_idx = graph[i]
                graph_x = x_impute[:, graph_idx]
                graph_mean = graph_x.mean(dim=0)
                graph_cov = graph_x.T.cov()
                temp = torch.linalg.solve(graph_cov[1:len(graph_idx), 1:len(graph_idx)], graph_cov[1:len(graph_idx), 0])
                cond_mean = graph_mean[0] + torch.matmul(graph_x[:, 1:len(graph_idx)] -
                                                         graph_mean[1:len(graph_idx)], temp)
                cond_cov = graph_cov[0, 0] - torch.matmul(temp, graph_cov[1:len(graph_idx), 0])
            #likelihoods.append(-self.sse(x_impute[:, self.miss_col[i]], cond_mean)/(2*cond_cov))
            likelihoods.append(self.sse(x_impute[:, self.miss_col[i]], cond_mean)/(2*cond_cov))
        likelihood = sum(likelihoods)
        return likelihood

    def backward_imputation(self, alpha, outcome_loss, x, y,mhstep,
                            obs_ind_loss_weight=1, graph=None, miss_lr=None, miss_ind=None):
        
        #hidden_output = self.forward(x)
        # initialize momentum term of x imputation
        if self.miss_col is not None:
            x_miss_momentum = torch.zeros_like(x[:, self.miss_col])

        # missing value imputation
        if self.miss_col is not None:
            for step in range(mhstep):
                x_impute = torch.clone(x.detach())  # x cannot be treated as leaf variable by pytorch, so create x_impute
                x_impute.requires_grad = True
                x_impute.grad = None

                miss_likelihood1 = self.likelihood_miss(x_impute, graph)
                miss_likelihood2 = outcome_loss(self.forward(x_impute), y)

                miss_likelihood1.backward()
                miss_likelihood2.backward()

                with torch.no_grad():
                    x_miss_momentum = (1 - alpha) * x_miss_momentum + miss_lr * x_impute.grad[:, self.miss_col] + \
                                        torch.FloatTensor(x_impute[:, self.miss_col].shape).to(self.device).normal_().mul(np.sqrt(2*alpha))
                    x_miss_momentum = x_miss_momentum * miss_ind # only update the entries with missing values
                    x[:, self.miss_col] += miss_lr * x_miss_momentum

                # update the hidden nodes in the first hidden layer after missing value imputation
                #hidden_output = torch.clone(self.forward(x).detach())

        #return hidden_output
    
    def forward(self, x):
        for layer_index in range(self.num_hidden+1):
            x = self.module_list[layer_index](x)
        return x


class StoNet_Survival_Sub(nn.Module):
    def __init__(self, num_hidden, hidden_dim, input_dim, output_dim, n_event,
                 miss_col=None, obs_ind_node=None, miss_pattern=None, graph=None):
        
        super(StoNet_Survival_Sub, self).__init__()
        self.num_hidden = num_hidden
        self.miss_col = miss_col
        self.miss_pattern = miss_pattern
        self.obs_ind_node = obs_ind_node
        self.n_event = n_event
        #self.graph = graph
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

        self.sse = nn.MSELoss(reduction='sum')

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    def likelihood_miss(self, x_impute, graph):
        likelihoods = []
        for i in range(len(self.miss_col)):
            with torch.no_grad():
                graph_idx = graph[i]
                graph_x = x_impute[:, graph_idx]
                graph_mean = graph_x.mean(dim=0)
                graph_cov = graph_x.T.cov()
                temp = torch.linalg.solve(graph_cov[1:len(graph_idx), 1:len(graph_idx)], graph_cov[1:len(graph_idx), 0])
                cond_mean = graph_mean[0] + torch.matmul(graph_x[:, 1:len(graph_idx)] -
                                                         graph_mean[1:len(graph_idx)], temp)
                cond_cov = graph_cov[0, 0] - torch.matmul(temp, graph_cov[1:len(graph_idx), 0])
            #likelihoods.append(-self.sse(x_impute[:, self.miss_col[i]], cond_mean)/(2*cond_cov))
            likelihoods.append(self.sse(x_impute[:, self.miss_col[i]], cond_mean)/(2*cond_cov))
        likelihood = sum(likelihoods)
        return likelihood

    def backward_imputation(self, alpha, outcome_loss, x, y,mhstep,
                            obs_ind_loss_weight=1, graph=None, miss_lr=None, miss_ind=None):
        
        #hidden_output = self.forward(x)
        # initialize momentum term of x imputation
        if self.miss_col is not None:
            x_miss_momentum = torch.zeros_like(x[:, self.miss_col])

        # missing value imputation
        if self.miss_col is not None:
            for step in range(mhstep):
                x_impute = torch.clone(x.detach())  # x cannot be treated as leaf variable by pytorch, so create x_impute
                x_impute.requires_grad = True
                x_impute.grad = None

                miss_likelihood1 = self.likelihood_miss(x_impute, graph)
                miss_likelihood2 = outcome_loss(self.forward(x_impute), y)

                miss_likelihood1.backward()
                miss_likelihood2.backward()

                with torch.no_grad():
                    x_miss_momentum = (1 - alpha) * x_miss_momentum + miss_lr * x_impute.grad[:, self.miss_col] + \
                                        torch.FloatTensor(x_impute[:, self.miss_col].shape).to(self.device).normal_().mul(np.sqrt(2*alpha))
                    x_miss_momentum = x_miss_momentum * miss_ind # only update the entries with missing values
                    x[:, self.miss_col] += miss_lr * x_miss_momentum

                # update the hidden nodes in the first hidden layer after missing value imputation
                #hidden_output = torch.clone(self.forward(x).detach())

        #return hidden_output
    
    def forward(self, x):
        for layer_index in range(self.num_hidden+1):
            x = self.module_list[layer_index](x)
        return x
