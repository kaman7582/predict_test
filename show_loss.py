import matplotlib.pyplot as plt
import numpy as np
from torch import nn

loss_path="./data/loss.npy"

def plot_loss():
    y = []
    y = list(np.load(loss_path))
    x = range(0,len(y))
    plt.plot(x, y, '.-')
    #plt_title = 'BATCH_SIZE = 32; LEARNING_RATE:0.001'
    #plt.title(plt_title)
    plt.xlabel('per 1000 times')
    plt.ylabel('LOSS')
    # plt.savefig(file_name)
    plt.show()

if __name__=="__main__":
    plot_loss()