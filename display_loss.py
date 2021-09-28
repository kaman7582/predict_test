import matplotlib.pyplot as plt
import torch
import numpy as np

def plot_loss():
    y = []
    y = list(np.load("./models/loss_{}.npy".format))
    x = range(0,len(y))
    plt.plot(x, y, '.-')
    #plt_title = 'BATCH_SIZE = 32; LEARNING_RATE:0.001'
    #plt.title(plt_title)
    plt.xlabel('per 1000 times')
    plt.ylabel('LOSS')
    # plt.savefig(file_name)
    plt.show()


col_name=['H2','CH4','C2H4','C2H6','C2H2']


def display_loss_data():
    plt.figure()
    i=1
    for name in col_name:
        y = []
        y = list(np.load("./models/loss_{}.npy".format(name)))
        plt.subplot(5,1,i)
        plt.plot( y, linestyle = 'solid',label=name,linewidth = '1')
        plt.ylabel('LOSS')
        plt.legend(shadow=True)
        if(i < 5):
            plt.xticks([])
        i += 1
        # plt.savefig(file_name)
    plt.xlabel('Train 1000 times')
    plt.savefig('loss.png',dpi=3840,format='png')
    plt.show()



if __name__=="__main__":
    display_loss_data()