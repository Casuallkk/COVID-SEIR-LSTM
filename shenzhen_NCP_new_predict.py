import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt

# 数据预处理
import pandas as pd
df = pd.read_excel('real_data.xlsx')
value = df['深圳新增确诊'].values[0:74]
print(len(value))
x = []
y = []
seq = 3
for i in range(len(value)-seq):
    x.append(value[i:i+seq])
    y.append(value[i+seq])
print(x, '\n', y)
print(len(x))   # 58

train_x = (torch.tensor(x[:60]).float()/100000.).reshape(-1, seq, 1)
train_y = (torch.tensor(y[:60]).float()/100000.).reshape(-1, 1)
test_x = (torch.tensor(x[60:]).float()/100000.).reshape(-1, seq, 1)
test_y = (torch.tensor(y[60:]).float()/100000.).reshape(-1, 1)
print(len(train_x))
print(len(test_x))
# 模型训练
class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=20, num_layers=1, batch_first=True)
        self.linear = nn.Linear(20 * seq, 1)
    def forward(self, x):
        x, (h, c) = self.lstm(x)
        x = x.reshape(-1, 20 * seq)
        x = self.linear(x)
        return x

# 模型训练
model = LSTM()
optimzer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_func = nn.MSELoss()
model.train()

for epoch in range(20000):
    output = model(train_x)
    loss = loss_func(output, train_y)
    optimzer.zero_grad()
    loss.backward()
    optimzer.step()
    if epoch % 20== 0:
        tess_loss = loss_func(model(test_x), test_y)
        print("epoch:{}, train_loss:{}, test_loss:{}".format(epoch, loss, tess_loss))

# 模型预测、画图
model.eval()
prediction = list((model(train_x).data.reshape(-1))*100000) + list((model(test_x).data.reshape(-1))*100000)
plt.plot(value[3:], label='True Value')
plt.plot(prediction[:59], label='LSTM fit')
plt.plot(np.arange(58, 71, 1), prediction[58:71], label='LSTM pred')
print(len(value[3:]))
print(len(prediction))
plt.legend(loc='best')
plt.title('New infections prediction(ShenZhen Municipality)')
plt.xlabel('Day')
plt.ylabel('New Cases')
plt.show()
