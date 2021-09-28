from matplotlib import pyplot
from pandas import read_csv
import numpy as np
from torch import nn
import torch

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
    
def train_dataset(input_size):
    data_trainx = np.asarray(dataset_x)
    data_trainy = np.asarray(dataset_y)
    total_len=data_trainx.shape[1]
    train_size = int((total_len) * train_percent)

    train_slice_x = data_trainx[:,:train_size]
    train_slice_y = data_trainy[:,:train_size]
    
    lstm_module = LSTM_Module(look_back, 8, output_size=1, num_layers=2)
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(lstm_module.parameters(), lr=1e-2)

    for i in range(SAMPLE_COL):
        print("Start to train %s"%(col_name[i]))
        loss_list=[]
        tmp_trainx = train_slice_x[i]
        train_X = tmp_trainx.reshape(-1, 1, input_size)
        tmp_trainy = train_slice_y[i]
        train_Y = tmp_trainy.reshape(-1, 1, 1)
        #convert to tensor
        train_tx = torch.from_numpy(train_X).to(torch.float32)
        train_ty = torch.from_numpy(train_Y).to(torch.float32)
        for idx in range(epochs):
            out = lstm_module(train_tx)
            loss = loss_function(out, train_ty)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss_list.append(loss.item())
            if (idx+1) % 100 == 0:
                print('Epoch: {}, Loss:{:.5f}'.format(idx+1, loss.item()))
        print("Finish training")
        torch.save(lstm_module,"./models/{}.pt".format(col_name[i]))
        np.save("./models/loss_{}.npy".format(col_name[i]),np.array(loss_list))

if __name__=="__main__":
    dataset_parser()
    train_dataset(look_back)
    #model = torch.load("./models/lstm.pt")