# 基于LSTM的疫情预测


import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=16, num_layers=1, batch_first=True)
        self.linear = nn.Linear(16 * seq, 1)

    def forward(self, x):
        x, (h, c) = self.lstm(x)
        x = x.reshape(-1, 16 * seq)
        x = self.linear(x)
        return x


if __name__ == '__main__':
    df = pd.read_excel('shanghai.xlsx')  # 这个会直接默认读取到这个Excel的第一个表单
    value = df['上海新增确诊'].values[0:67]
    df1 = pd.read_excel('result_data.xlsx')
    value_1 = df1['新增确诊'].values[0:67]

    x = []
    y = []
    seq = 3
    for i in range(len(value) - seq - 1):
        x.append(value[i:i + seq])
        y.append(value[i + seq])
    # print(x, '\n', y)

    train_x = (torch.tensor(x[:50]).float() / 1000.).reshape(-1, seq, 1)
    train_y = (torch.tensor(y[:50]).float() / 1000.).reshape(-1, 1)
    test_x = (torch.tensor(x[50:]).float() / 1000.).reshape(-1, seq, 1)
    test_y = (torch.tensor(y[50:]).float() / 1000.).reshape(-1, 1)
    # 模型训练
    model = LSTM()
    optimzer = torch.optim.Adam(model.parameters(), lr=0.005)
    loss_func = nn.MSELoss()
    model.train()
    l = []
    for epoch in range(600):
        output = model(train_x)
        loss = loss_func(output, train_y)
        l.append(loss)
        optimzer.zero_grad()
        loss.backward()
        optimzer.step()
        if epoch % 20 == 0:
            tess_loss = loss_func(model(test_x), test_y)
            print("epoch:{}, train_loss:{}, test_loss:{}".format(epoch, loss, tess_loss))

    # 模型预测、画图
    model.eval()
    prediction = list((model(train_x).data.reshape(-1)) * 1000) + list((model(test_x).data.reshape(-1)) * 1000)
    plt.figure(1)
    plt.plot(value[3:], label='True Value')
    plt.plot(prediction[:46], label='LSTM fit')
    plt.plot(np.arange(45, 61, 1), prediction[45:64], label='LSTM pred')
    plt.plot(value_1, label='SEIR')
    plt.legend(loc='best')
    plt.title('New daily infections prediction(Shanghai)')
    plt.xlabel('Day')
    plt.ylabel('New Confirmed Cases')
    plt.show()