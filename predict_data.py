from matplotlib import pyplot
from pandas import read_csv
import time
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from torch import nn
import torch
import os

#parameters setting
DAY_FOR_TRAIN=5
epochs=1000

#multivariate-lstm,load csv file
date_list=[]
c2h2_list=[]
h2_list=[]
ch4_list=[]
c2h4_list=[]
c2h6_list=[]

mon_name=[]
mon_list =[12,1,2,3,4,5,6,7,8,9]
mon_local=[0,14,46,73,103,123,136,166,196,229]

for mon in mon_list:
    if(mon == 12):
        mon_name.append("2020-"+str(mon))
    else:
        mon_name.append("2021-"+str(mon))

def data_create(file_name):
    dataset_train = read_csv(file_name,dtype=str)
    datelist_train = list(dataset_train['Date'])
    #print(len(datelist_train))
    year=2020
    # format the date
    for dt in  datelist_train:
        mon_day=time.strptime(dt,"%m.%d")
        month=mon_day.tm_mon
        idx = mon_list.index(month)
        if(mon_day.tm_mon == 1 and mon_day.tm_mday ==1):
            year=2021
        stamp = datetime(year, mon_day.tm_mon, mon_day.tm_mday)
        date_list.append(stamp.strftime('%Y-%m-%d'))

    #start to process chemistry data
    c2h2_train = list(dataset_train['C2H2'])
    h2_list = list(dataset_train['H2'])
    ch4_list = list(dataset_train['CH4'])
    c2h4_list = list(dataset_train['C2H4'])
    for ch in c2h2_train:
        c2h2_list.append(float(ch))
    
def draw_data_gram():
    #x = range(len(step_place))
    plt.xticks(mon_local, mon_name,rotation=45)
    plt.plot(c2h2_list,'g',label="C2H2")
    plt.show()


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


def creat_dataset(dataset,look_back):
    """
        根据给定的序列data，生成数据集。
        数据集分为输入和输出，每一个输入的长度为days_for_train，每一个输出的长度为1。
        也就是说用days_for_train天的数据，对应下一天的数据。
        若给定序列的长度为d，将输出长度为(d-days_for_train)个输入/输出对
    """
    data_x = []
    data_y = []
    for i in range(len(dataset)-look_back):
        data_x.append(dataset[i:i+look_back])
        data_y.append(dataset[i+look_back])
    return np.asarray(data_x), np.asarray(data_y) #转为ndarray数据

lstm_module = LSTM_Module(DAY_FOR_TRAIN, 8, output_size=1, num_layers=2)

def train_dataset(trainx,trainy,input_size):
    train_X = trainx.reshape(-1, 1, input_size)
    train_Y = trainy.reshape(-1, 1, 1)
    
    train_tx = torch.from_numpy(train_X).to(torch.float32)
    train_ty = torch.from_numpy(train_Y).to(torch.float32)

    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(lstm_module.parameters(), lr=1e-2)
    for i in range(epochs):
        out = lstm_module(train_tx)
        loss = loss_function(out, train_ty)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if (i+1) % 100 == 0:
            print('Epoch: {}, Loss:{:.5f}'.format(i+1, loss.item()))
    torch.save(lstm_module,"./models/lstm.pt")

def predict_date_set(all_data,raw_data,train_size):
    #model = lstm_module.eval()
    # find test size
    model = torch.load('./models/lstm.pt')
    dataset_x = all_data.reshape(-1, 1, DAY_FOR_TRAIN)
    dataset_x = torch.from_numpy(dataset_x).to(torch.float32)
    pred_test = model(dataset_x)
    pred_test = pred_test.view(-1).data.numpy()
    #real_part = raw_data[:train_size]
    #real_rest = raw_data[train_size:]
    #find out the rest part
    pred_rest = pred_test[train_size:]
    predict_size = len(all_data) - train_size
    xrange=range(train_size,train_size+predict_size)
    # find out the predict part
    plt.title('Predcitions C2H2', family='Arial', fontsize=12)
    plt.xlabel('Timeline', family='Arial', fontsize=10)
    plt.ylabel('C2H2', family='Arial', fontsize=10)

    plt.xticks(mon_local, mon_name,rotation=45)
    plt.plot(xrange,pred_rest, 'r', label='prediction')
    plt.plot(raw_data, 'b', label='real curve')
    plt.plot((train_size, train_size), (0, 1), 'g--')  # 分割线 左边是训练数据 右边是测试数据的输出
    plt.plot((0,len(all_data)),(0.5,0.5),'g--')
    plt.legend(shadow=True)
    plt.show()

if __name__=="__main__":
    data_create("./data/data.csv")
    #draw_data_gram()
    if len(c2h2_list) == 0:
        print("data error")
        exit(0)
    #conver data to numpy array
    c2h2_np=np.array(c2h2_list)
    '''
    # 准备归一化数据格式
    c2h2_np = c2h2_np.reshape((len(c2h2_np), 1))
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler = scaler.fit(c2h2_np)
    normalized = scaler.transform(c2h2_np)
    print(normalized)
    '''
    #using 70% data to train and 30% data to test
    #sequence = [i/float(length) for i in range(length)]
    '''
    如果是自然语言处理 (NLP) ，那么：
    seq_len 将对应句子的长度
    batch_size 同个批次中输入的句子数量
    inp_dim 句子中用来表示每个单词（中文分词）的矢量维度
    再举个例子，比如现在有5个句子，每个句子由3个单词组成，
    每个单词用10维的向量组成，这样参数为：seq_len=3, batch=5, input_size=10.
    '''
    
    data_X, data_Y =creat_dataset(c2h2_np,DAY_FOR_TRAIN)

    train_size = int(len(data_X) * 0.7)
    
    #data for training
    train_X = data_X[:train_size]
    train_Y = data_Y[:train_size]
    #data for testing
    test_X = data_X[train_size:]
    test_Y = data_Y[train_size:]
    if os.path.exists("./models/lstm.pt") == False:
        train_dataset(train_X,train_Y,DAY_FOR_TRAIN)
    
    predict_date_set(data_X,c2h2_np,train_size)
    



