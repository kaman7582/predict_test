import pandas as pd
from pandas import read_csv
from datetime import datetime
from pandas import read_csv
from matplotlib import pyplot
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from pandas import DataFrame
from pandas import concat
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from numpy import concatenate
from keras.layers import LSTM
from math import sqrt

data = pd.read_csv("https://raw.githubusercontent.com/jbrownlee/Datasets/master/pollution.csv",index_col=0)
data.head()
data.to_csv("raw.csv")

# 加载数据
def parse(x):
	return datetime.strptime(x, '%Y %m %d %H')
dataset = read_csv('https://raw.githubusercontent.com/jbrownlee/Datasets/master/pollution.csv',  parse_dates = [['year', 'month', 'day', 'hour']], index_col=0, date_parser=parse)
#删除No列
dataset.drop('No', axis=1, inplace=True)
# 修改剩余列名称
dataset.columns = ['pollution', 'dew', 'temp', 'press', 'wnd_dir', 'wnd_spd', 'snow', 'rain']
dataset.index.name = 'date'
# 将所有空值替换为0
dataset['pollution'].fillna(0, inplace=True)
# 删除前24小时行
dataset = dataset[24:]
# 打印前5行
print(dataset.head(5))
# 保存数据到pollution.csv
dataset.to_csv('pollution.csv')


#方便在浏览器中显示图标
#%matplotlib inline
# 加载数据
dataset = read_csv('pollution.csv', header=0, index_col=0)
values = dataset.values
# 选择指定列绘图
groups = [0, 1, 2, 3, 5, 6, 7]
i = 1
# 为每一列绘制图表
pyplot.figure()
for group in groups:
	pyplot.subplot(len(groups), 1, i)
	pyplot.plot(values[:, group])
	pyplot.title(dataset.columns[group], y=0.5, loc='right')
	i += 1
pyplot.show()


# 将数据转换成监督学习问题
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols, names = list(), list()
	# 输入序列(t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# 预测序列(t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# 把所有放在一起
	agg = concat(cols, axis=1)
	agg.columns = names
	# 删除空值行
	if dropnan:
		agg.dropna(inplace=True)
	return agg
 
# 加载数据
dataset = read_csv('pollution.csv', header=0, index_col=0)
values = dataset.values
# 对风向特征整数标签化
encoder = LabelEncoder()
values[:,4] = encoder.fit_transform(values[:,4])
#print(values)
#exit(0)
# 确保所有数据是浮点数类型
values = values.astype('float32')
# 对特征标准化
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)
# 构建成监督学习问题
reframed = series_to_supervised(scaled, 1, 1)
# 删除我们不想预测的天气数据列，只留下pollution列
reframed.drop(reframed.columns[[9,10,11,12,13,14,15]], axis=1, inplace=True)
print(reframed.head())

# 切分训练集和测试机
values = reframed.values
n_train_hours = 365 * 24
train = values[:n_train_hours, :]
test = values[n_train_hours:, :]
# 切分输入和输出
train_X, train_y = train[:, :-1], train[:, -1]
print(train_X.shape)
test_X, test_y = test[:, :-1], test[:, -1]
# 将输入转换为三维格式 [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

# 设计模型
model = Sequential()
model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')
# 拟合模型
history = model.fit(train_X, train_y, epochs=2, batch_size=72, validation_data=(test_X, test_y), verbose=2, shuffle=False)
# 绘制损失趋势线
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()

# 开始预测
yhat = model.predict(test_X)
test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
# 预测值反转缩放
inv_yhat = concatenate((yhat, test_X[:, 1:]), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:,0]
# 实际值反转缩放
test_y = test_y.reshape((len(test_y), 1))
inv_y = concatenate((test_y, test_X[:, 1:]), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:,0]
# 计算均方根误差
rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
print('Test RMSE: %.3f' % rmse)
