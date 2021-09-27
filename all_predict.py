from matplotlib import pyplot
from pandas import read_csv
import time
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from torch import nn
import torch
import os

trans_data=[]

col_name=['H2','CH4','C2H4','C2H6','C2H2']

def create_data(in_data):
    data_x = []
    data_y = []


all_data=read_csv("./data/data.csv")

for i in col_name:
    tmp = list(all_data[i])
    trans_data.append(tmp)

t_data = np.array(trans_data)

