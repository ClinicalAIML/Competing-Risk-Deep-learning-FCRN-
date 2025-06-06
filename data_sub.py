from scipy.stats import truncnorm, bernoulli, beta, norm
from torch.utils.data import Dataset, random_split, ConcatDataset, Subset
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
import numpy as np
import pandas as pd
import torch
import os
from sklearn.model_selection import train_test_split
from lifelines import KaplanMeierFitter



class sub_competing_complete(Dataset):

    def __init__(self, data_path, interval_length, focus_event):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.data = pd.read_csv(data_path)
        #self.data = self.data.dropna()
        self.data_size = len(self.data.index)
        nid = np.arange(1,self.data_size+1)
        self.data['nid'] = nid
        #max_time = self.data['time'].max()
        #min_time = self.data['time'].min()
        max_time = 100
        min_time = 0
        self.interval_length = interval_length
        self.max_interval = max_time // self.interval_length + (1 if max_time % self.interval_length != 0 else 0)
        #self.cat_col = ['binary1', 'binary2', 'binary3','binary4','binary5']
        self.num_col = ['continuous1', 'continuous2', 'continuous3', 'continuous4','continuous5','continuous6',
                        'continuous7','continuous8','continuous9','continuous10']
        self.focus_event = focus_event

        self.preprocess_data()
    
    # def get_miss_col(self):
    #     return self.missing_columns_indices

    def get_max_interval(self):
        return self.max_interval

    def preprocess_data(self):
        # Expanding DataFrame to long format

        # Estimate G(t) using Kaplan-Meier on censoring times
        time_points = np.arange(0, 101, 5)
        km_data = pd.DataFrame({
            'event_time': self.data['time'],
            'event_observed': (self.data['cause'] == 0).astype(int)  # 1 if censored, 0 otherwise
        })
        kmf = KaplanMeierFitter()
        kmf.fit(durations=km_data['event_time'], event_observed=km_data['event_observed'])
        # Obtain G_t at each time point
        G_t = kmf.survival_function_at_times(time_points).values
        # Ensure G_t is not zero to avoid division by zero
        G_t[G_t == 0] = np.finfo(float).eps + 0.5  # Replace zeros with a very small number
        #G_t[G_t == 0] = np.inf

        # Step 3: Compute weight_i(t_k)

        expanded_data = []
        for i, row in self.data.iterrows():
            T_i = row['time']
            num_Ti = int(T_i // self.interval_length - (1 if T_i % self.interval_length == 0 else 0))
            for interval in range(self.max_interval):  # Generate intervals
                start_time = interval * self.interval_length
                end_time = start_time + self.interval_length
                mid_time = (start_time + end_time) / 2
                
                if end_time < T_i:
                    actual_time = end_time
                    weight_t = 1
                    status = 0

                if T_i <= start_time:
                    actual_time = 0
                    status = 0
                    if row['cause']==0 or row['cause']==self.focus_event:
                        weight_t = 0
                    else:
                        weight_t = G_t[interval]/G_t[num_Ti]                    

                # Check if the event or censoring falls within this interval
                if start_time < T_i <= end_time:
                    weight_t = 1
                    actual_time = T_i
                    if row['cause']==self.focus_event:
                        status = 1
                    else:
                        status = 0
                    #event_occurred = row['cause'] if status == 1 else 0
                    
                expanded_data.append({
                    'nid': row['nid'],
                    'time': mid_time,
                    'actual_time': actual_time,
                    'status': status,
                    'censor': 1-status,
                    'weight': weight_t,
                    'continuous1': row['continuous1'],
                    'continuous2': row['continuous2'],
                    'continuous3': row['continuous3'],
                    'continuous4': row['continuous4'],
                    'continuous5': row['continuous5'],
                    'continuous6': row['continuous6'],
                    'continuous7': row['continuous7'],
                    'continuous8': row['continuous8'],
                    'continuous9': row['continuous9'],
                    'continuous10': row['continuous10']
                    # 'binary1': row['binary1'],
                    # 'binary2': row['binary2'],
                    # 'binary3': row['binary3'],
                    # 'binary4': row['binary4'],
                    # 'binary5': row['binary5']
                })

        # Convert list to DataFrame
        self.long_form_data = pd.DataFrame(expanded_data)
      
    def __len__(self):
        return int(self.data_size)

    def __getitem__(self, idx):
        batch_data = self.long_form_data[self.long_form_data['nid']==(idx+1)]
        sur_batch_data = self.data[self.data['nid']==(idx+1)]
        target = torch.FloatTensor(np.array(batch_data[['censor', 'status']], dtype=np.float32)).to(self.device)
        weight = torch.FloatTensor(np.array(batch_data['weight'], dtype=np.float32)).to(self.device)
        covariate = self.num_col + ['time']
        x = torch.FloatTensor(np.array(batch_data[covariate], dtype=np.float32)).to(self.device)
        actual_time = torch.FloatTensor(np.array(batch_data['actual_time'], dtype=np.float32)).to(self.device)
        surv_time = torch.FloatTensor(np.array(sur_batch_data['time'], dtype=np.float32)).to(self.device)
        indicator = torch.FloatTensor(np.array(sur_batch_data['cause'], dtype=np.float32)).to(self.device)
        return target, weight, x, actual_time, surv_time, indicator


class sub_competing_missing(Dataset):

    def __init__(self, data_path, interval_length, focus_event):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.data = pd.read_csv(data_path)
        self.data_size = len(self.data.index)
        #max_time = self.data['time'].max()
        #min_time = self.data['time'].min()
        nid = np.arange(1,self.data_size+1)
        self.data['nid'] = nid
        #max_time = self.data['time'].max()
        #min_time = self.data['time'].min()
        max_time = 100
        min_time = 0
        self.interval_length = interval_length
        self.max_interval = max_time // self.interval_length + (1 if max_time % self.interval_length != 0 else 0)
        #self.cat_col = ['binary1', 'binary2', 'binary3','binary4','binary5']
        self.num_col = ['continuous1', 'continuous2', 'continuous3', 'continuous4','continuous5','continuous6',
                        'continuous7','continuous8','continuous9','continuous10']
        self.num_var = self.data[self.num_col]
        self.missing_indicator = self.num_var.isnull().astype(int)
        self.miss_columns = self.num_var.columns[self.num_var.isnull().any()].to_list()
        self.missing_columns_indices = [self.num_var.columns.get_loc(col) for col in self.miss_columns]
        self.miss_rows = self.num_var.index[self.num_var.isnull().any(axis=1)].tolist()
        self.miss_ind = np.array(self.missing_indicator.iloc[:, self.missing_columns_indices], dtype=np.float32)

        num_var = self.data[self.num_col]
        num_imputer = SimpleImputer(strategy='median')
        num_var = num_imputer.fit_transform(num_var)
        self.data[self.num_col] = num_var
        x_scalar = StandardScaler()
        num_var = self.data[self.num_col]
        num_var = x_scalar.fit_transform(num_var)
        self.data[self.num_col] = num_var

        self.focus_event = focus_event
        self.preprocess_data()
    
    def get_miss_col(self):
        return self.missing_columns_indices

    def get_max_interval(self):
        return self.max_interval

    def preprocess_data(self):
        # Estimate G(t) using Kaplan-Meier on censoring times
        time_points = np.arange(0, 101, 5)
        km_data = pd.DataFrame({
            'event_time': self.data['time'],
            'event_observed': (self.data['cause'] == 0).astype(int)  # 1 if censored, 0 otherwise
        })
        kmf = KaplanMeierFitter()
        kmf.fit(durations=km_data['event_time'], event_observed=km_data['event_observed'])
        # Obtain G_t at each time point
        G_t = kmf.survival_function_at_times(time_points).values
        # Ensure G_t is not zero to avoid division by zero
        G_t[G_t == 0] = np.finfo(float).eps + 0.5  # Replace zeros with a very small number
        #G_t[G_t == 0] = np.inf

        # Expanding DataFrame to long format

        expanded_data = []
        for i, row in self.data.iterrows():
            T_i = row['time']
            num_Ti = int(T_i // self.interval_length - (1 if T_i % self.interval_length == 0 else 0))
            for interval in range(self.max_interval):  # Generate intervals
                start_time = interval * self.interval_length
                end_time = start_time + self.interval_length
                mid_time = (start_time + end_time) / 2
                
                if end_time < T_i:
                    actual_time = end_time
                    weight_t = 1
                    status = 0

                if T_i <= start_time:
                    actual_time = 0
                    status = 0
                    if row['cause']==0 or row['cause']==self.focus_event:
                        weight_t = 0
                    else:
                        weight_t = G_t[interval]/G_t[num_Ti]                    

                # Check if the event or censoring falls within this interval
                if start_time < T_i <= end_time:
                    weight_t = 1
                    actual_time = T_i
                    if row['cause']==self.focus_event:
                        status = 1
                    else:
                        status = 0
                    #event_occurred = row['cause'] if status == 1 else 0

                expanded_data1={
                    'nid': row['nid'],
                    'time': mid_time,
                    'actual_time': actual_time,
                    'status': status,
                    'censor': 1-status,
                    'weight': weight_t,
                    'continuous1': row['continuous1'],
                    'continuous2': row['continuous2'],
                    'continuous3': row['continuous3'],
                    'continuous4': row['continuous4'],
                    'continuous5': row['continuous5'],
                    'continuous6': row['continuous6'],
                    'continuous7': row['continuous7'],
                    'continuous8': row['continuous8'],
                    'continuous9': row['continuous9'],
                    'continuous10': row['continuous10']
                    # 'binary1': row['binary1'],
                    # 'binary2': row['binary2'],
                    # 'binary3': row['binary3'],
                    # 'binary4': row['binary4'],
                    # 'binary5': row['binary5']
                }
                for num_miss in range(len(self.missing_columns_indices)):
                    idd = self.missing_columns_indices[num_miss]
                    expanded_data1.update({f'miss_ind{idd}': self.miss_ind[i,num_miss]})
                expanded_data.append(expanded_data1)

        # Convert list to DataFrame
        self.long_form_data = pd.DataFrame(expanded_data)
                
    def __len__(self):
        return int(self.data_size)

    def __getitem__(self, idx):
        batch_data = self.long_form_data[self.long_form_data['nid']==(idx+1)]
        sur_batch_data = self.data[self.data['nid']==(idx+1)]
        target = torch.FloatTensor(np.array(batch_data[['censor', 'status']], dtype=np.float32)).to(self.device)
        weight = torch.FloatTensor(np.array(batch_data['weight'], dtype=np.float32)).to(self.device)
        covariate = self.num_col + ['time']
        x = torch.FloatTensor(np.array(batch_data[covariate], dtype=np.float32)).to(self.device)
        actual_time = torch.FloatTensor(np.array(batch_data['actual_time'], dtype=np.float32)).to(self.device)
        surv_time = torch.FloatTensor(np.array(sur_batch_data['time'], dtype=np.float32)).to(self.device)
        indicator = torch.FloatTensor(np.array(sur_batch_data['cause'], dtype=np.float32)).to(self.device)
        miss_name = []
        for num in range(len(self.missing_columns_indices)):
            idd = self.missing_columns_indices[num]
            name = f'miss_ind{idd}'
            miss_name.append(name)
        miss_ind = torch.FloatTensor(np.array(batch_data[miss_name], dtype=np.float32)).to(self.device)
        return target, weight, x, miss_ind, actual_time, surv_time, indicator
