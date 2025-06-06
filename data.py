from scipy.stats import truncnorm, bernoulli, beta, norm
from torch.utils.data import Dataset, random_split, ConcatDataset, Subset
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
import numpy as np
import pandas as pd
import torch
import os
from sklearn.model_selection import train_test_split


class competing_risk_inter2(Dataset):

    def __init__(self, data_path, interval_length):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.data = pd.read_csv(data_path)
        self.data = self.data.dropna()
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
        x_scalar = StandardScaler()
        num_var = self.data[self.num_col]
        num_var = x_scalar.fit_transform(num_var)
        self.data[self.num_col] = num_var

        self.preprocess_data()
    
    def get_miss_col(self):
        return self.missing_columns_indices

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
                    'continuous10': row['continuous10']
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
        covariate = self.num_col + ['time']
        x = torch.FloatTensor(np.array(batch_data[covariate], dtype=np.float32)).to(self.device)
        actual_time = torch.FloatTensor(np.array(batch_data['actual_time'], dtype=np.float32)).to(self.device)
        surv_time = torch.FloatTensor(np.array(sur_batch_data['time'], dtype=np.float32)).to(self.device)
        indicator = torch.FloatTensor(np.array(sur_batch_data['cause'], dtype=np.float32)).to(self.device)
        return target, x, actual_time, surv_time, indicator

class competing_risk_missing_inter(Dataset):

    def __init__(self, data_path, interval_length):
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

        self.preprocess_data()

    def get_max_interval(self):
        return self.max_interval
    
    def get_miss_col(self):
        return self.missing_columns_indices

    def preprocess_data(self):
        # Expanding DataFrame to long format
        expanded_data = []
        for i, row in self.data.iterrows():
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

                expanded_data1={
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
                    'continuous10': row['continuous10']
                }
                for num_miss in range(len(self.missing_columns_indices)):
                    idd = self.missing_columns_indices[num_miss]
                    expanded_data1.update({f'miss_ind{idd}': self.miss_ind[i,num_miss]})
                expanded_data.append(expanded_data1)
                

        # Convert list to DataFrame
        long_form_data = pd.DataFrame(expanded_data)

        # Creating dummy variables for each type of event
        max_cause = self.data['cause'].max()
        for i in range(1, int(max_cause + 1)):
            long_form_data[f'event_{i}'] = (long_form_data['event_occurred'] == i).astype(float)

        self.final_data = long_form_data



        
    def __len__(self):
        return int(self.data_size)

    def __getitem__(self, idx):
        batch_data = self.final_data[self.final_data['nid']==(idx+1)]
        sur_batch_data = self.data[self.data['nid']==(idx+1)]
        target = torch.FloatTensor(np.array(batch_data[['censored', 'event_1', 'event_2']], dtype=np.float32)).to(self.device)
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
        return target, x, miss_ind, actual_time, surv_time, indicator


class mimic_half(Dataset):

    def __init__(self, data_path, fda_path, interval_length, view):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.data = pd.read_pickle(data_path)
        self.fda_data = []
        with pd.HDFStore(fda_path, 'r') as store:
            for v in range(view):
                fda = store[f'fda{v+1}'].iloc[:,1:]
                self.fda_data.append(fda)
        self.fda = np.transpose(np.array(self.fda_data, dtype=np.float32),(1,2,0))

        self.data_size = len(self.data.index)
        nid = np.arange(1,self.data_size+1)
        self.data['nid'] = nid

        max_time = 183
        min_time = 0
        self.interval_length = interval_length
        self.max_interval = int(max_time // self.interval_length + (1 if max_time % self.interval_length != 0 else 0))
        self.cat_col = ['gender','race_clean_factor','admission_type_factor']
        self.num_col = ['anchor_age','bmi','icu_los','cci']
        x_scalar = StandardScaler()
        num_var = self.data[self.num_col]
        num_var = x_scalar.fit_transform(num_var)
        self.data[self.num_col] = num_var

        self.preprocess_data()
    
    # def get_miss_col(self):
    #     return self.missing_columns_indices

    def get_max_interval(self):
        return self.max_interval

    def preprocess_data(self):
        # Expanding DataFrame to long format
        expanded_data = []
        for _, row in self.data.iterrows():
            T_i = row['day_to_event_halfyear']
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
                    censor = 1 if row['event_type_halfyear']==0 else 0
                    event_occurred = row['event_type_halfyear']
                    actual_time = T_i

                expanded_data.append({
                    'nid': row['nid'],
                    'time': mid_time,
                    'actual_time': actual_time,
                    'censored': censor,
                    'event_occurred': event_occurred,
                    'anchor_age': row['anchor_age'],
                    'bmi': row['bmi'],
                    'icu_los': row['icu_los'],
                    'cci': row['cci'],
                    #'spo2': row['spo2'],
                    'gender': row['gender'],
                    'race_clean_factor': row['race_clean_factor'],
                    'admission_type_factor': row['admission_type_factor'],
                })

        # Convert list to DataFrame
        long_form_data = pd.DataFrame(expanded_data)

        # # Creating dummy variables for each type of event
        max_cause = self.data['event_type_halfyear'].max()
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
        surv_time = torch.FloatTensor(np.array(sur_batch_data['day_to_event_halfyear'], dtype=np.float32)).to(self.device)
        indicator = torch.FloatTensor(np.array(sur_batch_data['event_type_halfyear'], dtype=np.float32)).to(self.device)
        fda_data = torch.FloatTensor(self.fda[idx]).to(self.device)
        return target, x, fda_data, actual_time, surv_time, indicator

class mimic_whole(Dataset):

    def __init__(self, data_path, fda_path, interval_length, view):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.data = pd.read_pickle(data_path)
        #self.data = pd.DataFrame(data)
        self.fda_data = []
        with pd.HDFStore(fda_path, 'r') as store:
            for v in range(view):
                fda = store[f'fda{v+1}'].iloc[:,1:]
                self.fda_data.append(fda)
        self.fda = np.transpose(np.array(self.fda_data, dtype=np.float32),(1,2,0))

        self.data_size = len(self.data.index)
        nid = np.arange(1,self.data_size+1)
        self.data['nid'] = nid

        max_time = 365
        min_time = 0
        self.interval_length = interval_length
        self.max_interval = int(max_time // self.interval_length + (1 if max_time % self.interval_length != 0 else 0))
        self.cat_col = ['gender','race_clean_factor','admission_type_factor']
        self.num_col = ['anchor_age','bmi','icu_los','cci']
        x_scalar = StandardScaler()
        num_var = self.data[self.num_col]
        num_var = x_scalar.fit_transform(num_var)
        self.data[self.num_col] = num_var

        self.preprocess_data()
    
    # def get_miss_col(self):
    #     return self.missing_columns_indices

    def get_max_interval(self):
        return self.max_interval

    def preprocess_data(self):
        # Expanding DataFrame to long format
        expanded_data = []
        for _, row in self.data.iterrows():
            T_i = row['day_to_event_oneyear']
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
                    censor = 1 if row['event_type_oneyear']==0 else 0
                    event_occurred = row['event_type_oneyear']
                    actual_time = T_i

                expanded_data.append({
                    'nid': row['nid'],
                    'time': mid_time,
                    'actual_time': actual_time,
                    'censored': censor,
                    'event_occurred': event_occurred,
                    'anchor_age': row['anchor_age'],
                    'bmi': row['bmi'],
                    'icu_los': row['icu_los'],
                    'cci': row['cci'],
                    #'spo2': row['spo2'],
                    'gender': row['gender'],
                    'race_clean_factor': row['race_clean_factor'],
                    'admission_type_factor': row['admission_type_factor'],
                })

        # Convert list to DataFrame
        long_form_data = pd.DataFrame(expanded_data)

        # # Creating dummy variables for each type of event
        max_cause = self.data['event_type_oneyear'].max()
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
        surv_time = torch.FloatTensor(np.array(sur_batch_data['day_to_event_oneyear'], dtype=np.float32)).to(self.device)
        indicator = torch.FloatTensor(np.array(sur_batch_data['event_type_oneyear'], dtype=np.float32)).to(self.device)
        fda_data = torch.FloatTensor(self.fda[idx]).to(self.device)
        return target, x, fda_data, actual_time, surv_time, indicator




