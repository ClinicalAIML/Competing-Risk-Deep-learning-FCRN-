import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "6"
import sys
sys.path.insert(0, '.')
from model.network_fda import StoNet_Survival_FDA
from model.training import training_survival_fda
from data_fda import competing_risk_inter
from torch.utils.data import DataLoader
import torch
import numpy as np
import argparse

import errno
from torch.optim import SGD
from pickle import dump
import json

from model.metric2 import cs_metric



parser = argparse.ArgumentParser(description='Run Simulation for Causal StoNet')
# simulation setting
parser.add_argument('--data_num', default=1, type=int, help='number of simulation')
parser.add_argument('--num_sim', default=7, type=int, help='number of simulation runs')
parser.add_argument('--interval_length', default=5, type=int, help='number of simulation')
parser.add_argument('--num_event', default=2, type=int, help='number of competing events')
parser.add_argument('--n_base', default=4, type=int, help='number of base layer')
parser.add_argument('--view', default=3, type=int, help='number of fda series')
parser.add_argument('--learning_rate', default=1e-3, type=float, nargs='+',
                    help='step size for parameter update during training stage')
parser.add_argument('--weight_decay', default=0.01, type=float, help='decay factor for para')

parser.add_argument('--partition_seed', default=1563, type=int, help='set seed for dataset partition')

# Parameter for StoNet
# model
parser.add_argument('--layer', default=3, type=int, help='number of hidden layers')
parser.add_argument('--unit', default=[32, 64, 32], type=int, nargs='+', help='number of hidden unit in each layer')
parser.add_argument('--regression', dest='classification_flag', action='store_false', help='false for regression')
parser.add_argument('--classification', dest='classification_flag', action='store_true', help='true for classification')

# training setting
parser.add_argument('--train_epoch', default=100, type=int, help='total number of training epochs')
parser.add_argument('--mh_step', default=1, type=int, help='number of SGHMC step for imputation')
parser.add_argument('--impute_lr', default=[1e-2, 1e-4], type=float, nargs='+', help='step size for SGHMC')
parser.add_argument('--impute_alpha', default=0.1, type=float, help='momentum weight for SGHMC')
parser.add_argument('--para_momentum', default=0.9, type=float, help='momentum weight for parameter update')
parser.add_argument('--para_lr_decay', default=0.8, type=float, help='decay factor for para_lr')
parser.add_argument('--impute_lr_decay', default=0.6, type=float, help='decay factor for impute_lr')

# Parameters for Sparsity
parser.add_argument('--num_run', default=1, type=int, help='Number of different initialization used to train the model')

args = parser.parse_args()


def fit_and_evaluate_Causal_StoNet(data_seed, save_results=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # task
    classification_flag = args.classification_flag
    

    train_path = f'./raw_data/fda_view3/complete_train/train_data_{data_seed}.csv'
    test_path = f'./raw_data/fda_view3/complete_test/test_data_{data_seed}.csv'
    train_fda_path = f'./raw_data/fda_view3/complete_train/train_fda_{data_seed}.h5'
    test_fda_path = f'./raw_data/fda_view3/complete_test/test_fda_{data_seed}.h5'
    train_set = competing_risk_inter(train_path, train_fda_path, args.interval_length, args.view)
    test_set = competing_risk_inter(test_path, test_fda_path, args.interval_length, args.view)

    max_interval = train_set.get_max_interval()

    # load training data and validation data
    batch_size = 64
    test_size = len(test_set)
    train_data = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_data = DataLoader(test_set, batch_size=test_size, shuffle=False)
    test_data = DataLoader(test_set, batch_size=test_size, shuffle=False)
    #in_sample_data = DataLoader(in_sample_set, batch_size=batch_size)

    # network setup
    net_args = dict(num_hidden=args.layer,
                    hidden_dim=args.unit,
                    input_dim=16,
                    output_dim=2 if classification_flag else 1,
                    n_event=args.num_event,
                    n_base=args.n_base,
                    max_interval=max_interval,
                    view=args.view,
                    #treat_layer=args.depth,
                    #treat_node=args.treat_node
                    )

    # set number of independent runs for sparsity
    num_seed = args.num_run

    # training setting
    lr = args.learning_rate
    training_epochs = args.train_epoch
    para_lr_decay = args.para_lr_decay


    # training results containers
    results = dict(dim=0,
                   BIC=0,
                   num_selection_out=0,
                   num_selection_treat=0,
                   out_train_loss=0,
                   out_val_loss=0
                   )

    if classification_flag:
        results.update([('out_train_acc', 0), ('out_val_acc', 0)])

    # path to save the result
    base_path = os.path.join(f'./competing_risk_results/cause_specific/complete_simple_fda_view3/exp_{args.num_sim}', str(data_seed))
    if not os.path.exists(base_path):
        os.makedirs(base_path)

    with open(base_path + '/' + 'args_output_vari.json', "w") as file:
        json.dump(vars(args), file, indent=4)

    for prune_seed in range(num_seed):
        print('number of runs', prune_seed)

        PATH = base_path
        if not os.path.isdir(PATH):
            try:
                os.makedirs(PATH)
            except OSError as exc:  # Python >2.5
                if exc.errno == errno.EEXIST and os.path.isdir(PATH):
                    pass
                else:
                    raise

        # initialize network
        np.random.seed(prune_seed)
        torch.manual_seed(prune_seed)
        net = StoNet_Survival_FDA(**net_args)
        net.to(device)

        optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=args.weight_decay)


        # train
        print("Train")
        output_train = training_survival_fda(mode="train",
                                net=net,
                                epochs=training_epochs,
                                optimizer=optimizer,
                                train_data=train_data,
                                val_data=val_data,
                                batch_size=batch_size,
                                outcome_cat=classification_flag,
                                base_path=base_path
                                )
        #para_train = output_train["para_path"]
        performance_fine_tune = output_train["performance"]

        if save_results:
            torch.save(net.state_dict(), os.path.join(base_path, 'model' + str(data_seed) + '.pt'))


        # calculate performance
        net.load_state_dict(torch.load(os.path.join(base_path,'best_model.pt')))
        net.eval()

        IBS1, IBS2 = 0, 0
        
        with torch.no_grad():
            #for y, treat, x, *rest in val_data:
            count = 0
            for target, x, x_fda, actual_time, test_time, test_indicator in val_data:
                surv_time = train_set.data['time']
                indicator = train_set.data['cause']
                v = x.size()[-1]
                x = x.reshape(-1, v)
                #y = (target, censor)
                pred, _, _ = net.forward((x,x_fda))

                IBS = cs_metric(pred.cpu().numpy(), target.cpu().numpy(), surv_time, indicator)
                IBS1 += IBS[0]
                IBS2 += IBS[1]

                count += 1
        
            IBS1 /= count
            IBS2 /= count

        if True:
            results['out_val_loss'] = performance_fine_tune['out_val_loss'][-1]
            results['best_val_loss'] = performance_fine_tune['best_val'][-1]

            results['test_IBS1'] = IBS1
            results['test_IBS2'] = IBS2


    # save overall performance
    if save_results:
        with open(os.path.join(base_path, 'competing_stoNet_results_' + str(data_seed) + '.json'), "w") as f:
            json.dump(results, f, indent=4)
    return results

result = fit_and_evaluate_Causal_StoNet(args.data_num, save_results=True)

