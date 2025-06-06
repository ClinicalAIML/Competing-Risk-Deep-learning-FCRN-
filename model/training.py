import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import time
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F
from lifelines.utils import concordance_index
from sksurv.metrics import concordance_index_censored, brier_score, integrated_brier_score
from lifelines import KaplanMeierFitter



def log(x):
    return torch.log(x + 1e-10)

class CensoredLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(CensoredLoss, self).__init__()
        self.reduction = reduction

    def forward(self, outputs, y):
        targets = y
        #censors = y[1]
        batch_size, num_intervals, v = targets.shape
        outputs = outputs.reshape(batch_size, num_intervals, -1)
        loss = 0.0
        total_count = 0

        # Iterate over each individual in the batch
        for i in range(batch_size):
            for t in range(num_intervals):
                # Check if the event is censored after interval t
                if torch.sum(targets[i, t:]) == 0:
                    # Ignore the rest of the intervals after censoring/event
                    break
                # Calculate cross-entropy loss for the observed interval
                censor_p = 1 - torch.sum(outputs[i,t], dim=-1)
                loss1 = targets[i,t,0] * log(censor_p)

                loss2 = torch.sum(targets[i,t,1:] * log(outputs[i,t]))
                total_loss = loss1 + loss2
                loss += total_loss

                #loss += F.cross_entropy(outputs[i, t].unsqueeze(0), targets[i, t].argmax().unsqueeze(0))
                total_count += 1


        if self.reduction == 'sum':
            return -loss
        elif self.reduction == 'mean':
            #total_elements = uncensored_mask.float().sum()
            return -loss / total_count if total_count > 0 else torch.tensor(1)
        elif self.reduction == 'none':
            return -loss / batch_size
        else:
            raise ValueError("Unsupported reduction type. Choose 'mean', 'sum', or 'none'.")


class CensoredLoss_Sub(nn.Module):
    def __init__(self, reduction='mean'):
        super(CensoredLoss_Sub, self).__init__()
        self.reduction = reduction

    def forward(self, outputs, y):
        targets = y[0]
        weights = y[1]
        batch_size, num_intervals, v = targets.shape
        outputs = outputs.reshape(batch_size, num_intervals, -1)
        loss = 0.0
        total_count = 0
        #outputs = outputs.reshape(batch_size, num_intervals, v)
        # Iterate over each individual in the batch
        for i in range(batch_size):
            for t in range(num_intervals):
                # Check if the event is censored after interval t
                censor_p = 1 - outputs[i,t]
                loss1 = targets[i,t,0] * log(censor_p)
                loss2 = targets[i,t,1] * log(outputs[i,t])
                total_loss = (loss1 + loss2) * weights[i,t]
                loss += total_loss

                #loss += F.cross_entropy(outputs[i, t].unsqueeze(0), targets[i, t].argmax().unsqueeze(0))
                total_count += 1

        if self.reduction == 'sum':
            return -loss
        elif self.reduction == 'mean':
            #total_elements = uncensored_mask.float().sum()
            return -loss / total_count if total_count > 0 else torch.tensor(1)
        else:
            raise ValueError("Unsupported reduction type. Choose 'mean', 'sum', or 'none'.")


def training_survival(mode, net, train_data, val_data, epochs, batch_size, optimizer,base_path,
             outcome_cat=False, CE_weight=None,
             graph=None, miss_lr=None):

    # save training and validation performance
    out_train_loss_path = []
    out_val_loss_path = []
    c1_path = []
    c2_path = []
    ibs1_path = []
    ibs2_path = []
    best_val_path = []

    performance = dict(out_train_loss=out_train_loss_path, out_val_loss=out_val_loss_path, cc1=c1_path, cc2=c2_path,
                       ibs1=ibs1_path,ibs2=ibs2_path,best_val=best_val_path)

    if outcome_cat:
        out_train_acc_path = []
        out_val_acc_path = []
        performance.update([('out_train_acc', out_train_acc_path), ('out_val_acc', out_val_acc_path)])

    para_path = {}
    para_grad_path = {}
    para_gamma_path = {}

    if mode == "train":
        # save parameter values, indicator variables, and selected input variables for each epoch
        input_gamma_path = dict(var_selected_out={}, var_selected_treat={}, num_selected_out=[],
                                num_selected_treat=[])


    out_loss = CensoredLoss(reduction='mean')
    out_loss_sum = CensoredLoss(reduction='sum')

    # training
    best_val = 100
    for epoch in range(epochs):
        print("Epoch" + str(epoch))


        total_loss = 0
        #for y, treat, x, *rest in train_data:
        for target, x, actual_time, surv_time, indicator in train_data:

            v = x.size()[-1]
            x = x.reshape(-1, v)
            y = target
            #y=(target,indicator,surv_time,mask[0],mask[1])
            
            pred = net.forward(x)

            loss = out_loss(pred, y)
            optimizer.zero_grad()

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print('Epoch {}'.format(epoch), 'Loss:{:.6f}'.format(total_loss/len(train_data)))


        # calculate validation performance
        out_val_loss = 0
        with torch.no_grad():
            #for y, treat, x, *rest in val_data:
            for target, x, actual_time, surv_time, indicator in val_data:
                v = x.size()[-1]
                x = x.reshape(-1, v)
                y = target
                #y=(target,indicator,surv_time,mask[0],mask[1])
                pred = net.forward(x)
                out_val_loss += out_loss(pred, y).item()


        if outcome_cat is False:
            # use RMSE as the model performance metric for regression tasks
            out_val_loss = np.sqrt(out_val_loss/len(val_data))
        else:
            out_val_loss /= len(val_data)
        out_val_loss_path.append(out_val_loss)
        print(f"Avg val loss: {out_val_loss:>8f} \n")
        if out_val_loss < best_val:
            best_val = out_val_loss
            torch.save(net.state_dict(), os.path.join(base_path, 'best_model' + '.pt'))
        best_val_path.append(best_val)
        print(f"Best val loss: {best_val:>8f} \n")



    if mode == "pretrain":
        output = dict(para_path=para_path, para_grad_path=para_grad_path, para_gamma_path=para_gamma_path,
                      performance=performance)
    else:
        output = dict(para_path=para_path, para_grad_path=para_grad_path, para_gamma_path=para_gamma_path,
                      input_gamma_path=input_gamma_path, performance=performance,
                    )

    return output


def training_survival_sub(mode, net, train_data, val_data, epochs, batch_size, optimizer,base_path,
             outcome_cat=False, CE_weight=None,
             graph=None, miss_lr=None):

    # save training and validation performance
    out_train_loss_path = []
    out_val_loss_path = []
    c1_path = []
    c2_path = []
    ibs1_path = []
    ibs2_path = []
    best_val_path = []

    performance = dict(out_train_loss=out_train_loss_path, out_val_loss=out_val_loss_path, cc1=c1_path, cc2=c2_path,
                       ibs1=ibs1_path,ibs2=ibs2_path,best_val=best_val_path)

    if outcome_cat:
        out_train_acc_path = []
        out_val_acc_path = []
        performance.update([('out_train_acc', out_train_acc_path), ('out_val_acc', out_val_acc_path)])

    para_path = {}
    para_grad_path = {}
    para_gamma_path = {}

    if mode == "train":
        # save parameter values, indicator variables, and selected input variables for each epoch
        input_gamma_path = dict(var_selected_out={}, var_selected_treat={}, num_selected_out=[],
                                num_selected_treat=[])


    out_loss = CensoredLoss_Sub(reduction='mean')
    out_loss_sum = CensoredLoss_Sub(reduction='sum')

    # training
    best_val = 100
    for epoch in range(epochs):
        print("Epoch" + str(epoch))


        total_loss = 0
        #for y, treat, x, *rest in train_data:
        for target, weight, x, actual_time, surv_time, indicator in train_data:

            v = x.size()[-1]
            x = x.reshape(-1, v)
            y = (target, weight)
            #y=(target,indicator,surv_time,mask[0],mask[1])
            
            pred = net.forward(x)

            loss = out_loss(pred, y)
            optimizer.zero_grad()

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print('Epoch {}'.format(epoch), 'Loss:{:.6f}'.format(total_loss/len(train_data)))


        # calculate validation performance
        out_val_loss = 0
        with torch.no_grad():
            #for y, treat, x, *rest in val_data:
            for target, weight, x, actual_time, surv_time, indicator in val_data:
                v = x.size()[-1]
                x = x.reshape(-1, v)
                y = (target,weight)
                #y=(target,indicator,surv_time,mask[0],mask[1])
                pred = net.forward(x)
                out_val_loss += out_loss(pred, y).item()


        if outcome_cat is False:
            # use RMSE as the model performance metric for regression tasks
            out_val_loss = np.sqrt(out_val_loss/len(val_data))
        else:
            out_val_loss /= len(val_data)
        out_val_loss_path.append(out_val_loss)
        print(f"Avg val loss: {out_val_loss:>8f} \n")
        if out_val_loss < best_val:
            best_val = out_val_loss
            torch.save(net.state_dict(), os.path.join(base_path, 'best_model' + '.pt'))
        best_val_path.append(best_val)
        print(f"Best val loss: {best_val:>8f} \n")



    if mode == "pretrain":
        output = dict(para_path=para_path, para_grad_path=para_grad_path, para_gamma_path=para_gamma_path,
                      performance=performance)
    else:
        output = dict(para_path=para_path, para_grad_path=para_grad_path, para_gamma_path=para_gamma_path,
                      input_gamma_path=input_gamma_path, performance=performance,
                    )

    return output


def training_survival_fda(mode, net, train_data, val_data, epochs, batch_size, optimizer,base_path,
             outcome_cat=False, CE_weight=None,
             graph=None, miss_lr=None):

    # save training and validation performance
    out_train_loss_path = []
    out_val_loss_path = []
    c1_path = []
    c2_path = []
    ibs1_path = []
    ibs2_path = []
    best_val_path = []
    performance = dict(out_train_loss=out_train_loss_path, out_val_loss=out_val_loss_path, cc1=c1_path, cc2=c2_path,
                       ibs1=ibs1_path,ibs2=ibs2_path,best_val=best_val_path)

    if outcome_cat:
        out_train_acc_path = []
        out_val_acc_path = []
        performance.update([('out_train_acc', out_train_acc_path), ('out_val_acc', out_val_acc_path)])

    para_path = {}
    para_grad_path = {}
    para_gamma_path = {}

    if mode == "train":
        # save parameter values, indicator variables, and selected input variables for each epoch
        input_gamma_path = dict(var_selected_out={}, var_selected_treat={}, num_selected_out=[],
                                num_selected_treat=[])


    out_loss = CensoredLoss(reduction='mean')
    out_loss_sum = CensoredLoss(reduction='sum')

    # training
    best_val = 100
    for epoch in range(epochs):
        print("Epoch" + str(epoch))


        total_loss = 0
        #for y, treat, x, *rest in train_data:
        for target, x, x_fda, actual_time, surv_time, indicator in train_data:

            v = x.size()[-1]
            x = x.reshape(-1, v)
            y = target
            #y=(target,indicator,surv_time,mask[0],mask[1])
            
            pred, reg1, reg2 = net.forward((x,x_fda))

            loss = out_loss(pred, y)
            optimizer.zero_grad()

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print('Epoch {}'.format(epoch), 'Loss:{:.6f}'.format(total_loss/len(train_data)))
        out_train_loss_path.append(total_loss/len(train_data))


        # calculate validation performance
        out_val_loss = 0
        with torch.no_grad():
            #for y, treat, x, *rest in val_data:
            for target, x, x_fda, actual_time, surv_time, indicator in val_data:
                v = x.size()[-1]
                x = x.reshape(-1, v)
                y = target
                #y=(target,indicator,surv_time,mask[0],mask[1])
                pred, _, _ = net.forward((x,x_fda))
                out_val_loss += out_loss(pred, y).item()


        if outcome_cat is False:
            # use RMSE as the model performance metric for regression tasks
            out_val_loss = np.sqrt(out_val_loss/len(val_data))
        else:
            out_val_loss /= len(val_data)
        out_val_loss_path.append(out_val_loss)
        print(f"Avg val loss: {out_val_loss:>8f} \n")
        if out_val_loss < best_val:
            best_val = out_val_loss
            torch.save(net.state_dict(), os.path.join(base_path, 'best_model' + '.pt'))
        best_val_path.append(best_val)
        print(f"Best val loss: {best_val:>8f} \n")



    if mode == "pretrain":
        output = dict(para_path=para_path, para_grad_path=para_grad_path, para_gamma_path=para_gamma_path,
                      performance=performance)
    else:
        output = dict(para_path=para_path, para_grad_path=para_grad_path, para_gamma_path=para_gamma_path,
                      input_gamma_path=input_gamma_path, performance=performance,
                    )

    return output


def training_survival_fda_sub(mode, net, train_data, val_data, epochs, batch_size, optimizer,base_path,
             outcome_cat=False, CE_weight=None,
             graph=None, miss_lr=None):

    # save training and validation performance
    out_train_loss_path = []
    out_val_loss_path = []
    c1_path = []
    c2_path = []
    ibs1_path = []
    ibs2_path = []
    best_val_path = []

    performance = dict(out_train_loss=out_train_loss_path, out_val_loss=out_val_loss_path, cc1=c1_path, cc2=c2_path,
                       ibs1=ibs1_path,ibs2=ibs2_path,best_val=best_val_path)

    if outcome_cat:
        out_train_acc_path = []
        out_val_acc_path = []
        performance.update([('out_train_acc', out_train_acc_path), ('out_val_acc', out_val_acc_path)])

    para_path = {}
    para_grad_path = {}
    para_gamma_path = {}

    if mode == "train":
        # save parameter values, indicator variables, and selected input variables for each epoch
        input_gamma_path = dict(var_selected_out={}, var_selected_treat={}, num_selected_out=[],
                                num_selected_treat=[])


    out_loss = CensoredLoss_Sub(reduction='mean')
    out_loss_sum = CensoredLoss_Sub(reduction='sum')

    # training
    best_val = 100
    for epoch in range(epochs):
        print("Epoch" + str(epoch))


        total_loss = 0
        #for y, treat, x, *rest in train_data:
        for target, weight, x, x_fda, actual_time, surv_time, indicator in train_data:

            v = x.size()[-1]
            x = x.reshape(-1, v)
            y = (target, weight)
            #y=(target,indicator,surv_time,mask[0],mask[1])
            
            pred, reg1, reg2 = net.forward((x,x_fda))

            loss = out_loss(pred, y)
            optimizer.zero_grad()

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print('Epoch {}'.format(epoch), 'Loss:{:.6f}'.format(total_loss/len(train_data)))
        out_train_loss_path.append(total_loss/len(train_data))

        # calculate validation performance
        out_val_loss= 0
        with torch.no_grad():
            #for y, treat, x, *rest in val_data:
            for target, weight, x, x_fda, actual_time, surv_time, indicator in val_data:
                v = x.size()[-1]
                x = x.reshape(-1, v)
                y = (target, weight)
                #y=(target,indicator,surv_time,mask[0],mask[1])
                pred, _, _ = net.forward((x,x_fda))
                out_val_loss += out_loss(pred, y).item()


        if outcome_cat is False:
            # use RMSE as the model performance metric for regression tasks
            out_val_loss = np.sqrt(out_val_loss/len(val_data))
        else:
            out_val_loss /= len(val_data)
        out_val_loss_path.append(out_val_loss)
        print(f"Avg val loss: {out_val_loss:>8f} \n")
        if out_val_loss < best_val:
            best_val = out_val_loss
            torch.save(net.state_dict(), os.path.join(base_path, 'best_model' + '.pt'))
        best_val_path.append(best_val)
        print(f"Best val loss: {best_val:>8f} \n")



    if mode == "pretrain":
        output = dict(para_path=para_path, para_grad_path=para_grad_path, para_gamma_path=para_gamma_path,
                      performance=performance)
    else:
        output = dict(para_path=para_path, para_grad_path=para_grad_path, para_gamma_path=para_gamma_path,
                      input_gamma_path=input_gamma_path, performance=performance,
                    )

    return output