import matplotlib.pyplot as plt
from pandas import read_csv
import numpy as np
import torch
from torch import nn

epochs=1000
SAMPLE_COL = 5
look_back = 5
train_percent=0.7
dataset_x=[]
dataset_y=[]
col_name=['H2','CH4','C2H4','C2H6','C2H2']
trans_data=[]

raw_data_path="./data/data.csv"
save_module_path="./models/"

mon_list =[12,1,2,3,4,5,6,7,8,9]
mon_local=[0,14,46,73,103,123,136,166,196,229]
mon_name=[]
for mon in mon_list:
    if(mon == 12):
        mon_name.append("2020-"+str(mon))
    else:
        mon_name.append("2021-"+str(mon))


class LSTM_Module(nn.Module):
    """
        使用LSTM进行回归
        
        参数：
        - input_size: feature size
        - hidden_size: number of hidden units
        - output_size: number of output
        - num_layers: layers of LSTM to stack
    """
    def __init__(self, input_size, hidden_size, output_size=1, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)
        self.fc = nn.Linear(hidden_size, output_size)
    def forward(self, _x):
        x, _ = self.lstm(_x)  # _x is input, size (seq_len, batch, input_size)
        s, b, h = x.shape  # x is output, size (seq_len, batch, hidden_size)
        x = x.view(s*b, h)
        x = self.fc(x)
        x = x.view(s, b, -1)  # 把形状改回来
        return x

def create_data(dataset):
    data_x = []
    data_y = []
    for i in range(len(dataset)-look_back):
        data_x.append(dataset[i:i+look_back])
        data_y.append(dataset[i+look_back])
    return data_x,data_y #转为ndarray数据

def dataset_parser():
    all_data=read_csv(raw_data_path)
    # get all the information
    for i in col_name:
        tmp = list(all_data[i])
        trans_data.append(tmp)
    #convert all 
    for i in range(SAMPLE_COL):
        tmpx,tmpy = create_data(trans_data[i])
        dataset_x.append(tmpx)
        dataset_y.append(tmpy)

def predict_all():
    dataset_parser()
    data_trainx = np.asarray(dataset_x)
    total_len=data_trainx.shape[1]
    train_size = int((total_len) * train_percent)
    predict_size = total_len-train_size
    plt.figure()
    for i in range(SAMPLE_COL):
        load_path="./models/{}.pt".format(col_name[i])
        model = torch.load(load_path)
        model = model.eval()
        train_X = data_trainx[i].reshape(-1, 1, look_back)
        train_tx = torch.from_numpy(train_X).to(torch.float32)
        pred_test = model(train_tx)
        pred_test = pred_test.view(-1).data.numpy()
        pred_test = pred_test[train_size:]
        xrange=range(train_size,train_size+predict_size)
        plt.subplot(5,1,i+1)
        plt.plot(trans_data[i],'b',label=col_name[i])
        plt.plot(xrange,pred_test, 'r', label='prediction')
        plt.legend(loc="upper left",fontsize=6,shadow=True)
        plt.xticks([])
        max_val =max(np.max(trans_data[i]),1)
        plt.plot((train_size, train_size),(0,max_val), 'g--')  # 分割线 左边是训练数据 右边是测试数据的输出
        if(col_name[i] == 'C2H2'):
            plt.plot((0,total_len),(0.5,0.5),'g--',linewidth = 1)
            plt.annotate('Alarm', xy=(40, 0.5), xytext=(50, 0.8),
            xycoords='data',
            arrowprops=dict(arrowstyle="->")
            )
            #plt.set_ylim(-2, 2)
    plt.xticks(mon_local, mon_name,rotation=45)
    #plt.savefig("result.png")
    plt.savefig('predict.png',dpi=3840,format='png')
    plt.show()
    

if __name__=="__main__":
    predict_all()