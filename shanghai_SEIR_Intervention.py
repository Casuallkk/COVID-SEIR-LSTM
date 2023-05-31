# 上海SEIR传染病模型仿真
######################################
# N: 区域内总人口                      #
# S: 易感者                           #
# E: 潜伏者                           #
# I: 感染者                           #
# R: 康复者                           #
# r: 每天接触的人数                    #
# r2: 潜伏者每天接触的人数              #
# beta1: 感染者传染给易感者的概率, S——>I #
# beta2: 潜伏者感染易感者的概率, E——>S   #
# sigma: 潜伏者转化为感染者的概率, E——>I #
# gama: 康复概率, I——>R                #
# T: 传播时间                          #
#######################################

import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as spi
import pandas as pd


def getpre(i_t, increase):
    for i in range(len(i_t)):
        if i <= 0:
            increase[i] = i_t[i]
        else:
            increase.append(i_t[i] - i_t[i - 1])
    return increase


def getpre1(i_t, increase):
    for i in range(len(i_t)):
        if i <= 0:
            pass
        else:
            tmp = i_t[i] - i_t[i - 1]
            if tmp <= 0:
                increase.append(0)
            else:
                increase.append(tmp)
    return increase


def SEIR1(inivalue, _):
    X = inivalue
    Y = np.zeros(4)
    # S数量
    Y[0] = - (r * beta1 * X[0] * X[2]) / N - (r * beta2 * X[0] * X[1]) / N
    # E数量
    Y[1] = (r * beta1 * X[0] * X[2]) / N + (r * beta2 * X[0] * X[1]) / N - sigma * X[1]
    # I数量
    Y[2] = sigma * X[1] - gamma * X[2]
    # R数量
    Y[3] = gamma * X[2]
    return Y


def SEIR2(inivalue, _):
    X = inivalue
    Y = np.zeros(4)
    # S数量
    Y[0] = - (r2 * beta1 * X[0] * X[2]) / N - (r2 * beta2 * X[0] * X[1]) / N
    # E数量
    Y[1] = (r2 * beta1 * X[0] * X[2]) / N + (r2 * beta2 * X[0] * X[1]) / N - sigma2 * X[1]
    # I数量
    Y[2] = sigma2 * X[1] - gamma * X[2]
    # R数量
    Y[3] = gamma * X[2]
    return Y


if __name__ == '__main__':
    N = 24894300
    E_0 = 0
    I_0 = 38
    R_0 = 8
    S_0 = N - E_0 - I_0 - R_0
    beta1 = 0.025
    beta2 = 0.021 / 3
    sigma = 1 / 5.5
    gamma = 1 / 7
    r = 16.5
    T = 41

    # ode求解
    INI = [S_0, E_0, I_0, R_0]
    T_range = np.arange(0, T + 1)
    Res = spi.odeint(SEIR1, INI, T_range)
    S_t = Res[:, 0]
    E_t = Res[:, 1]
    I_t = Res[:, 2]
    R_t = Res[:, 3]
    I_t1_1 = [0]
    I_t1_1 = getpre(I_t, I_t1_1)

    # 阶段二，3.28后
    S_2 = S_t[T]
    E_2 = E_t[T]
    I_2 = I_t[T]
    R_2 = R_t[T]
    I_1 = I_t1_1[T]

    beta1 = 0.002
    beta2 = 0.018 / 3
    sigma2 = 1 / 5.5
    r2 = 1
    T2 = 60 - T

    # ode求解
    INI = [S_2, E_2, I_2, R_2]
    T_range = np.arange(0, T2 + 1)
    Res = spi.odeint(SEIR2, INI, T_range)
    S_t2 = Res[:, 0]
    E_t2 = Res[:, 1]
    I_t2 = Res[:, 2]
    R_t2 = Res[:, 3]
    I_t2_2 = [I_1]
    I_t2_2 = getpre1(I_t2, I_t2_2)
    # 显示日期
    plt.figure(figsize=(10, 6))

    xs = pd.date_range(start='20220301', periods=T + 1, freq='1D')  # 生成2020-02-11类型的日期数组（）
    # print(xs)
    xs2 = pd.date_range(start='20220411', periods=T2 + 1, freq='1D')

    plt.plot(xs, E_t, color='grey', label='Exposed', marker='.')
    plt.plot(xs2, E_t2, color='grey', label='Exposed Prediction')
    plt.plot(xs, I_t, color='red', label='Infected', marker='.')
    plt.plot(xs, I_t1_1, color='blue', label='Infected_increase', marker='.')
    plt.plot(xs2, I_t2, color='red', label='Infected Prediction')
    plt.plot(xs2, I_t2_2, color='blue', label='Infected increase Prediction')
    plt.plot(xs, I_t + R_t, color='green', label='Infected + Removed', marker='.')
    plt.plot(xs2, I_t2 + R_t2, color='green', label='Cumulative Infections Prediction')
    plt.xlabel('Date')
    plt.ylabel('Number')
    plt.title('SEIR Prediction(Shanghai, 3.28 Intervention)')
    plt.legend()
    plt.show()
