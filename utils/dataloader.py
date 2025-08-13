import torch
import torch.nn as nn
from torch.utils.data import DataLoader , TensorDataset
from torch.optim import Adam
import torch.nn.init as init

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import MultipleLocator
import matplotlib.cm as cm

import copy
import seaborn as sns

from scipy.stats import norm
from sklearn.neighbors import KernelDensity, LocalOutlierFactor

import tqdm

import optuna
from optuna.trial import TrialState

def load_data():
    # model hyperparameters
    cuda = torch.cuda.is_available()
    DEVICE = torch.device("cuda" if cuda else "cpu")

    num_seeds = 30
    seed = 0

    all_state_dim = 64
    all_action_dim = 19
    training_seed = 21

    # Load fullstate
    data_fullstate = np.empty(num_seeds, dtype=object)
    data_no_joint_pos = np.empty(num_seeds, dtype=object)
    data_no_joint_vel = np.empty(num_seeds, dtype=object)
    data_no_action = np.empty(num_seeds, dtype=object)
    data_no_imu = np.empty(num_seeds, dtype=object)
    data_no_fc = np.empty(num_seeds, dtype=object)
    for i in range(num_seeds):
        data_fullstate[i] = np.load(f"data/HEBB-FULL-STATE_seed-{seed}-fullstate-rand-{i}.npz")    
        data_no_joint_pos[i] = np.load(f"data/HEBB-FULL-STATE_seed-{seed}-no_joint_pos-rand-{i}.npz")
        data_no_joint_vel[i] = np.load(f"data/HEBB-FULL-STATE_seed-{seed}-no_joint_vel-rand-{i}.npz")
        data_no_action[i] = np.load(f"data/HEBB-FULL-STATE_seed-{seed}-no_action-rand-{i}.npz")
        data_no_imu[i] = np.load(f"data/HEBB-FULL-STATE_seed-{seed}-no_imu-rand-{i}.npz")
        data_no_fc[i] = np.load(f"data/HEBB-FULL-STATE_seed-{seed}-no_fc-rand-{i}.npz")
        
    train_x = torch.empty((0, all_state_dim), dtype=torch.float32 ,device=DEVICE)
    train_y = torch.empty((0, all_action_dim), dtype=torch.float32,device=DEVICE)
    test_x = torch.empty((0, all_state_dim), dtype=torch.float32,device=DEVICE)
    test_y = torch.empty((0, all_action_dim), dtype=torch.float32,device=DEVICE)
    for i in range(training_seed):
        train_x = torch.cat((train_x, torch.tensor(data_fullstate[i]["state"].reshape(data_fullstate[i]["state"].shape[0], -1), dtype=torch.float32,device=DEVICE)), dim=0)
        train_y = torch.cat((train_y, torch.tensor(data_fullstate[i]["action_lowpass"].reshape(data_fullstate[i]["action_lowpass"].shape[0], -1), dtype=torch.float32,device=DEVICE)), dim=0)
    for j in range(training_seed, num_seeds):
        test_x = torch.cat((test_x, torch.tensor(data_no_joint_pos[j]["state"].reshape(data_no_joint_pos[j]["state"].shape[0], -1), dtype=torch.float32,device=DEVICE)), dim=0)
        test_y = torch.cat((test_y, torch.tensor(data_no_joint_pos[j]["action_lowpass"].reshape(data_no_joint_pos[j]["action_lowpass"].shape[0], -1), dtype=torch.float32,device=DEVICE)), dim=0)