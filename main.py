# 多元線性回歸
from matplotlib.font_manager import fontManager
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mlp


# 引入中文字體
fontManager.addfont('ChineseFont.ttf')
mlp.rc('font', family='ChineseFont')

data = pd.read_csv('./Salary_Data2.csv')

# label encoding: 如果特徵存在高低關係，可以使用label encoding做分類
data['EducationLevel'] = data['EducationLevel'].map({'高中以下': 0, '大學': 1, '碩士以上': 2})

# one-hot encoding: 如果特徵不存在高低關係，可以使用one-hot encoding做分類，可以省略最後一個特徵來節省分類
# 使用sklearn的OneHotEncoder來自動分類沒有高低關係的特徵
onehotEncoder = OneHotEncoder()
onehotEncoder.fit(data[['City']])
cityEncoded = onehotEncoder.transform(data[['City']]).toarray()
data[['CityA', 'CityB', 'CityC']] = cityEncoded

# 移除多餘的資料和特徵
# axis: 1是列, 0是行
data = data.drop(['City', 'CityC'], axis=1)
# print(data)

# 訓練集、測試集
# 將資料分成兩份: 一份拿來測試、一份拿來訓練
# 通常會分測試20%、訓練80%
x = data[['YearsExperience', 'EducationLevel', 'CityA', 'CityB']]
y = data['Salary']

# test_size: 指定測試集為20%訓練集為80%, random_state: 固定資料排序
xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.2, random_state=87)

# 將pandas的資料轉換成numpy的格式以便後續計算
xTrain = xTrain.to_numpy()
xTest = xTest.to_numpy()
yTrain = yTrain.to_numpy()
yTest = yTest.to_numpy()

# 特徵縮放: 提升梯度下降的速度
# 特徵縮放: 標準化
# 利用StandardScaler來實現自動特徵縮放
scaler = StandardScaler()
scaler.fit(xTrain)
xTrain = scaler.transform(xTrain)
xTest = scaler.transform(xTest)

# 模型公式: y = w1 * x1 + w2 * x2 + w3 * x3 ... + b
# 套用到data時: 月薪 = w1 * 年資 + w2 * 學歷 + w3 * 城市 + b
# w = np.array([1, 2, 3, 4]) # [w1, w2, w3, w4]
# b = 0

# 取得w1 * x1 + w2 * x2 + w3 * x3 + w4 * x4的資料
# axis: 1是行, 0是列
# yPred = (xTrain * w).sum(axis=1) + b
# print(yPred)

# cost function: cost = (真實數據 - 預測值) ** 2
# 目前使訓練階段，所以使用yTrain來計算cost
# print((yTrain - yPred) ** 2)
def computeCost(x, y, w, b):
    yPred = (x * w).sum(axis=1) + b
    cost = ((y - yPred) ** 2).mean() # 計算cost平均
    return cost
# w = np.array([0, 2, 2, 4])
# b = 0
# print(computeCost(xTrain, yTrain, w, b))

# optimizer-gradient descent
"""
多元線性回歸 梯度下降公式: (w: 特徵)
w1 gradient = 2 * x1 (yPred - y)
w2 gradient = 2 * x2 (yPred - y)
w3 gradient = 2 * x3 (yPred - y)
w4 gradient = 2 * x4 (yPred - y)
b gradient = 2 (yPred - y)
"""
# yPred = (xTrain * w).sum(axis=1) + b
# bGradient = (yPred - yTrain).mean()
# w1 gradient = x1 (yPred - yTrain)
# : 表示第一維度的資料全都要，0表示第二微度的第一筆資料
# w1Gradient = (xTrain[:, 0] * (yPred - yTrain)).mean()
# w2Gradient = (xTrain[:, 1] * (yPred - yTrain)).mean()
# w3Gradient = (xTrain[:, 2] * (yPred - yTrain)).mean()
# w4Gradient = (xTrain[:, 3] * (yPred - yTrain)).mean()
# print(w1Gradient, w2Gradient, w3Gradient, w4Gradient)
# 將w1至w4特徵的斜率資料放入w gradient
# .shape: 取得指定陣列微度的長度
# wGradient = np.zeros(xTrain.shape[1])
# for i in range(xTrain.shape[1]):
#     wGradient[i] = (xTrain[:, i] * (yPred - yTrain)).mean()

def computeGradient(x, y, w, b):
    yPred = (x * w).sum(axis=1) + b
    wGradient = np.zeros(x.shape[1])
    bGradient = (yPred - y).mean()
    wGradient = np.zeros(xTrain.shape[1])
    for i in range(xTrain.shape[1]):
        wGradient[i] = (xTrain[:, i] * (yPred - yTrain)).mean()

    return wGradient, bGradient

# 梯度下降
# 設定np float format
np.set_printoptions(formatter={'float': '{: .2e}'.format})
def gradientDescent(x, y, wInit, bInit, learningRate, costFunction, gradientFunction, runIter, pIter=1000):

    # 紀錄cost, w, b
    cHist = []
    wHist = []
    bHist = []

    # 初始 w, b
    w = wInit
    b = bInit
    for i in range(runIter):
        wGradient, bGradient = gradientFunction(x, y, w, b)
        w = w - wGradient * learningRate
        b = b - bGradient * learningRate
        cost = costFunction(x, y, w, b)

        cHist.append(cost)
        wHist.append(w)
        bHist.append(b)

        # 每一千次print資料
        if i % pIter == 0:
            print(f'Ieration: {i:5}, Cost: {cost:.2e}, w: {w}, b: {b:.2e}, w gradient: {wGradient}, b gradient: {bGradient:.2e}')

    return w, b, wHist, bHist, cHist

wInit = np.array([1, 2, 3, 4])
bInit = 0
learningRate = 1.0e-3
runIter = 20000
wFinal, bFinal, wHist, bHist, cHist = gradientDescent(xTrain, yTrain, wInit, bInit, learningRate, computeCost, computeGradient, runIter)

# 測試階段
yPred = (wFinal * xTest).sum(axis=1) + bFinal
# 列出預測結果與實際結果
print(pd.DataFrame({
    'y pred': yPred,
    'y test': yTest
}))

# final cost
print(computeCost(xTest, yTest, wFinal, bFinal))

# 5.3年資 碩士以上 城市A
# 7.2年資 高中以下 城市B
xReal = np.array([[5.3, 2, 1, 0], [7.2, 0, 0, 1]])
# 特徵縮放
xReal = scaler.transform(xReal)
yRead = (wFinal * xReal).sum(axis=1) + bFinal
print(yRead)