import numpy as np
import pandas as pd
from lifelines.utils import concordance_index
from pycox.evaluation import EvalSurv
from sksurv.metrics import concordance_index_censored, brier_score, integrated_brier_score
from lifelines import KaplanMeierFitter
import torch
from scipy.special import softmax
from sksurv.nonparametric import kaplan_meier_estimator
from sksurv.metrics import cumulative_dynamic_auc

   
    
def cs_metric(predicted_hazards, targets, surv_time, cause):

    n_samples, n_intervals, v = targets.shape
    n_events = v - 1
    predicted_hazards = predicted_hazards.reshape(n_samples,n_intervals,n_events)

    # Step 1: Estimate G(t) using Kaplan-Meier on censoring times
    time_points = np.arange(0, 101, 5)
    km_data = pd.DataFrame({
        'event_time': surv_time,
        'event_observed': (cause == 0).astype(int)  # 1 if censored, 0 otherwise
    })
    kmf = KaplanMeierFitter()
    kmf.fit(durations=km_data['event_time'], event_observed=km_data['event_observed'])

    # Obtain G_t at each time point
    G_t = kmf.survival_function_at_times(time_points).values
    # Ensure G_t is not zero to avoid division by zero
    #G_t[G_t == 0] = np.finfo(float).eps + 0.5  # Replace zeros with a very small number
    G_t[G_t == 0] = np.inf

    # reform targets
    observed = targets.copy()
    for i in range(n_samples):
        for e in range(n_events):
            for t in range(n_intervals):
                if observed[i,t,e+1]==1:
                    observed[i,t+1:,e+1]=1
                    break
                

    # Step 2: Compute CIFs
    predicted_cifs = np.zeros((n_samples, n_intervals, n_events))
    wr_t_hat = np.zeros((n_intervals, n_events))
    for e in range(n_events):
        survival_prob = np.ones(n_samples)
        for k in range(n_intervals):
            hazard = predicted_hazards[:, k, e]
            cumulative_hazard = np.sum(predicted_hazards[:, k], axis=-1)
            predicted_cifs[:, k, e] = predicted_cifs[:, k-1, e] + survival_prob * hazard
            # if k==0:
            #     predicted_cifs[:, k, e] = predicted_cifs[:, 0, e] + survival_prob * hazard
            # else:
            #     predicted_cifs[:, k, e] = predicted_cifs[:, k-1, e] + survival_prob * hazard
            event_prob = survival_prob * hazard
            survival_prob *= (1 - cumulative_hazard)
            wr_t = event_prob*survival_prob
            wr_t_mean = np.mean(wr_t)
            wr_t_hat[k,e] = wr_t_mean


    # # Step 3: Compute delta_i(t_k)
    # delta = np.ones((n_samples, n_intervals))
    # for i in range(n_samples):
    #     for tt in range(n_intervals):
    #         if np.sum(targets[i, tt:]) == 0:
    #             delta[i, tt] = 0

    # Step 4: Compute BS_IPCW at each time point for each event
    BS_IPCW = np.zeros((n_intervals, n_events))
    for e in range(n_events):
        for k in range(n_intervals):
            # numerator = delta[:, k] * (predicted_cifs[:, k, e] - observed[:, k, e+1])**2
            # null_numrator = delta[:, k] * (null_cifs[:, k, e] - observed[:, k, e+1])**2
            numerator = (predicted_cifs[:, k, e] - observed[:, k, e+1])**2
            #G_t_k = G_t[k]
            # BS_IPCW[k, e] = np.mean(numerator / G_t_k)
            # null_BS_IPCW[k, e] = np.mean(null_numrator / G_t_k)
            BS_IPCW[k, e] = np.mean(numerator)

    # Step 5: Compute IBS for each event
    IBS = np.zeros(n_events)
    for e in range(n_events):
        IBS[e] = np.mean(BS_IPCW[:, e])

    return IBS


def cs_metric_sub(predicted_hazards, targets, surv_time, cause):

    n_samples, n_intervals, v = targets.shape
    n_events = v - 1
    predicted_hazards = predicted_hazards.reshape(n_samples,n_intervals,n_events)


    # Step 1: Estimate G(t) using Kaplan-Meier on censoring times
    time_points = np.arange(0, 101, 5)
    #time_points = np.arange(0, 140, 5)
    km_data = pd.DataFrame({
        'event_time': surv_time,
        'event_observed': (cause == 0).astype(int)  # 1 if censored, 0 otherwise
    })
    kmf = KaplanMeierFitter()
    kmf.fit(durations=km_data['event_time'], event_observed=km_data['event_observed'])

    # Obtain G_t at each time point
    G_t = kmf.survival_function_at_times(time_points).values
    # Ensure G_t is not zero to avoid division by zero
    #G_t[G_t == 0] = np.finfo(float).eps + 0.5  # Replace zeros with a very small number
    G_t[G_t == 0] = np.inf

    # reform targets
    observed = targets.copy()
    for i in range(n_samples):
        for e in range(n_events):
            for t in range(n_intervals):
                if observed[i,t,e+1]==1:
                    observed[i,t+1:,e+1]=1
                    break
                

    # Step 2: Compute CIFs
    predicted_cifs = np.zeros((n_samples, n_intervals, n_events))
    wr_t_hat = np.zeros((n_intervals, n_events))
    for e in range(n_events):
        survival_prob = np.ones(n_samples)
        for k in range(n_intervals):
            hazard = predicted_hazards[:, k, e]
            cumulative_hazard = hazard
            predicted_cifs[:, k, e] = predicted_cifs[:, k-1, e] + survival_prob * hazard
            # if k==0:
            #     predicted_cifs[:, k, e] = predicted_cifs[:, 0, e] + survival_prob * hazard
            # else:
            #     predicted_cifs[:, k, e] = predicted_cifs[:, k-1, e] + survival_prob * hazard
            event_prob = survival_prob * hazard
            survival_prob *= (1 - cumulative_hazard)
            wr_t = event_prob*survival_prob
            wr_t_mean = np.mean(wr_t)
            wr_t_hat[k,e] = wr_t_mean


    # # Step 3: Compute delta_i(t_k)
    # delta = np.ones((n_samples, n_intervals))
    # for i in range(n_samples):
    #     for tt in range(n_intervals):
    #         if np.sum(targets[i, tt:]) == 0:
    #             delta[i, tt] = 0

    # Step 4: Compute BS_IPCW at each time point for each event
    BS_IPCW = np.zeros((n_intervals, n_events))
    for e in range(n_events):
        for k in range(n_intervals):
            # numerator = delta[:, k] * (predicted_cifs[:, k, e] - observed[:, k, e+1])**2
            numerator = (predicted_cifs[:, k, e] - observed[:, k, e+1])**2
            #G_t_k = G_t[k]
            # BS_IPCW[k, e] = np.mean(numerator / G_t_k)
            BS_IPCW[k, e] = np.mean(numerator)


    # Step 5: Compute IBS for each event
    IBS = np.zeros(n_events)
    IBS = np.mean(BS_IPCW)

    return IBS

