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
import h5py



class competing_risk_inter(Dataset):

    def __init__(self, data_path, fda_path, interval_length, view):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.data = pd.read_csv(data_path)
        #self.fda_data = pd.read_csv(fda_path)
        with h5py.File(fda_path, 'r') as file:
            matrices = {name: np.array(file[name]) for name in file.keys()}
        self.fda_data = []
        for i in range(view):
            fda_x = np.array(matrices[f'fda{i+1}']).astype(np.float32).T
            self.fda_data.append(fda_x)
        self.data_size = len(self.data.index)
        nid = np.arange(1,self.data_size+1)
        self.data['nid'] = nid
        #max_time = self.data['time'].max()
        #min_time = self.data['time'].min()
        max_time = 100
        min_time = 0
        self.interval_length = interval_length
        self.max_interval = max_time // self.interval_length + (1 if max_time % self.interval_length != 0 else 0)
        self.cat_col = ['binary1', 'binary2', 'binary3','binary4','binary5']
        self.num_col = ['continuous1', 'continuous2', 'continuous3', 'continuous4','continuous5','continuous6',
                        'continuous7','continuous8','continuous9','continuous10']
        x_scalar = StandardScaler()
        num_var = self.data[self.num_col]
        num_var = x_scalar.fit_transform(num_var)
        self.data[self.num_col] = num_var
        self.fda = np.transpose(np.array(self.fda_data, dtype=np.float32),(1,2,0))


        self.preprocess_data()
    
    # def get_miss_col(self):
    #     return self.missing_columns_indices

    def get_max_interval(self):
        return self.max_interval

    def preprocess_data(self):
        # Expanding DataFrame to long format
        expanded_data = []
        for _, row in self.data.iterrows():
            T_i = row['time']
            num_intervals = T_i // self.interval_length + (1 if T_i % self.interval_length != 0 else 0)
            for interval in range(self.max_interval):  # Generate intervals
                start_time = interval * self.interval_length
                end_time = start_time + self.interval_length
                mid_time = (start_time + end_time) / 2
                
                if end_time < T_i:
                    actual_time = end_time
                    censor = 1
                    event_occurred = 0

                if T_i <= start_time:
                    actual_time = 0
                    censor = 0
                    event_occurred = 0

                # Check if the event or censoring falls within this interval
                if start_time < T_i <= end_time:
                    censor = 1 - row['status']
                    event_occurred = row['cause'] if row['status'] == 1 else 0
                    actual_time = T_i

                expanded_data.append({
                    'nid': row['nid'],
                    'time': mid_time,
                    'actual_time': actual_time,
                    'censored': censor,
                    'event_occurred': event_occurred,
                    'continuous1': row['continuous1'],
                    'continuous2': row['continuous2'],
                    'continuous3': row['continuous3'],
                    'continuous4': row['continuous4'],
                    'continuous5': row['continuous5'],
                    'continuous6': row['continuous6'],
                    'continuous7': row['continuous7'],
                    'continuous8': row['continuous8'],
                    'continuous9': row['continuous9'],
                    'continuous10': row['continuous10'],
                    'binary1': row['binary1'],
                    'binary2': row['binary2'],
                    'binary3': row['binary3'],
                    'binary4': row['binary4'],
                    'binary5': row['binary5']
                })

        # Convert list to DataFrame
        long_form_data = pd.DataFrame(expanded_data)

        # # Creating dummy variables for each type of event
        max_cause = self.data['cause'].max()
        for i in range(1, max_cause + 1):
            long_form_data[f'event_{i}'] = (long_form_data['event_occurred'] == i).astype(float)

        self.final_data = long_form_data



        
    def __len__(self):
        return int(self.data_size)

    def __getitem__(self, idx):
        batch_data = self.final_data[self.final_data['nid']==(idx+1)]
        sur_batch_data = self.data[self.data['nid']==(idx+1)]
        target = torch.FloatTensor(np.array(batch_data[['censored', 'event_1', 'event_2']], dtype=np.float32)).to(self.device)
        covariate = self.num_col + self.cat_col + ['time']
        x = torch.FloatTensor(np.array(batch_data[covariate], dtype=np.float32)).to(self.device)
        actual_time = torch.FloatTensor(np.array(batch_data['actual_time'], dtype=np.float32)).to(self.device)
        surv_time = torch.FloatTensor(np.array(sur_batch_data['time'], dtype=np.float32)).to(self.device)
        indicator = torch.FloatTensor(np.array(sur_batch_data['cause'], dtype=np.float32)).to(self.device)
        fda_data = torch.FloatTensor(self.fda[idx]).to(self.device)
        return target, x, fda_data, actual_time, surv_time, indicator


class sub_competing_complete(Dataset):

    def __init__(self, data_path, fda_path, interval_length, focus_event,view):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.data = pd.read_csv(data_path)
        #self.fda_data = pd.read_csv(fda_path)
        with h5py.File(fda_path, 'r') as file:
            matrices = {name: np.array(file[name]) for name in file.keys()}
        self.fda_data = []
        for i in range(view):
            fda_x = np.array(matrices[f'fda{i+1}']).astype(np.float32).T
            self.fda_data.append(fda_x)
        self.data_size = len(self.data.index)
        nid = np.arange(1,self.data_size+1)
        self.data['nid'] = nid
        #max_time = self.data['time'].max()
        #min_time = self.data['time'].min()
        max_time = 100
        min_time = 0
        self.interval_length = interval_length
        self.max_interval = max_time // self.interval_length + (1 if max_time % self.interval_length != 0 else 0)
        self.cat_col = ['binary1', 'binary2', 'binary3','binary4','binary5']
        self.num_col = ['continuous1', 'continuous2', 'continuous3', 'continuous4','continuous5','continuous6',
                        'continuous7','continuous8','continuous9','continuous10']
        
        x_scalar = StandardScaler()
        num_var = self.data[self.num_col]
        num_var = x_scalar.fit_transform(num_var)
        self.data[self.num_col] = num_var
        self.focus_event = focus_event

        self.fda = np.transpose(np.array(self.fda_data, dtype=np.float32),(1,2,0))

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
                    'continuous10': row['continuous10'],
                    'binary1': row['binary1'],
                    'binary2': row['binary2'],
                    'binary3': row['binary3'],
                    'binary4': row['binary4'],
                    'binary5': row['binary5']
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
        covariate = self.num_col + self.cat_col + ['time']
        x = torch.FloatTensor(np.array(batch_data[covariate], dtype=np.float32)).to(self.device)
        actual_time = torch.FloatTensor(np.array(batch_data['actual_time'], dtype=np.float32)).to(self.device)
        surv_time = torch.FloatTensor(np.array(sur_batch_data['time'], dtype=np.float32)).to(self.device)
        indicator = torch.FloatTensor(np.array(sur_batch_data['cause'], dtype=np.float32)).to(self.device)
        fda_data = torch.FloatTensor(self.fda[idx]).to(self.device)
        return target, weight, x, fda_data, actual_time, surv_time, indicator

