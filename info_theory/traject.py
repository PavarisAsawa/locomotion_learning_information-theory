import torch
import torch.nn as nn
from torch.utils.data import DataLoader , TensorDataset
from torchvision.utils import save_image, make_grid
from torch.optim import Adam
import torch.nn.init as init

import numpy as np
import math

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import MultipleLocator
import matplotlib.cm as cm
from matplotlib.cm import get_cmap

import copy
import seaborn as sns

from scipy.stats import norm
from sklearn.neighbors import KernelDensity, LocalOutlierFactor
from sklearn.decomposition import PCA

import tqdm
import pickle

# MI estimators
# from utils.estimators import *


def import_data(mode:list , num_policy:int , case:str, slope_level:int=5):
    dataset = {key : [] for key in mode}
    for i in dataset.keys():   # run all mode
        for policy in range(num_policy): # each run have 3 policies
            if case == "slope":
                dataset[i].append(np.load(f"/home/workstation/pavaris_ws/locomotion_learning_information-theory/data{case}/{case}-{i}-ANGLE{str(int)}-policy_{policy}.npy" , allow_pickle=True).item())
            else:
                dataset[i].append(np.load(f"/home/workstation/pavaris_ws/locomotion_learning_information-theory/data/{case}/{case}-{i}-policy_{policy}.npy" , allow_pickle=True).item())
    for key, value in dataset.items():
        dataset[key] = np.array(value)
    return dataset

def plot_trajectory(dataset, keys:str , joint_arr:list ,pol_num:int=0 ,env_num:int=0, t=None):
    trajectory_dataset = dataset["full"][pol_num]["data"][keys][: , env_num , joint_arr] # [timestep , num_joint]
    num_row = trajectory_dataset.shape[1]
    if t is None:
        timestep = np.arange(trajectory_dataset.shape[0])
    else:
        t0, t1 = int(t[0]), int(t[1])                 # minimal: just use provided window
        timestep = np.arange(t0, t1)
        trajectory_dataset = trajectory_dataset[t0:t1, :]   # <-- NEW: slice data to match x


    fig, axes = plt.subplots(nrows=num_row, ncols=1, figsize=(10, 3*num_row), sharex=True, sharey=False)
    for i, ax in enumerate(axes):
        joint_idx = joint_arr[i]
        ax.plot(timestep, trajectory_dataset[:, i])
        ax.set_ylabel(f"j{joint_idx}")
        ax.grid(True, alpha=0.3)
        ax.set_title(f"{keys} • policy {pol_num}, env {env_num} • joint[{joint_idx}]", fontsize=10)
    plt.show()  # <-- important


def main():
    NUM_POLICY = 5
    matplotlib.use('Qt5Agg')
    mode=["full","vel_act", "pos_act", "pos_vel_act", "act_FC", "act_IMU_FC", "pos_act_IMU", "pos_act_FC", "vel_act_IMU", "vel_act_FC"]
    dataset  = import_data(mode , NUM_POLICY , case="flat")

    # [case:str][number_file:int][data/brain:str]
    selected_dataset = dataset
    plot_trajectory(dataset=dataset, pol_num=0,keys="state" , joint_arr=[0,1,2] , t=[0,500])

if __name__ == '__main__':
    main()