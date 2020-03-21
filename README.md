# My-MachineLearning-python
机器学习算法汇总 包括找到的代码及自己编写的代码
## 1.学习笔记
机器学习，也称为统计学习，是人工智能的一个分支，研究如何从数据中提取知识。早期建立人工智能系统除了需要构建推理系统，还需要知识作为推理的基础。但目前海量知识如何让机器识别是一个难题。人工总结知识点传给机器相对缓慢，速度也较慢。让机器自己从大量数据中学习规律，从而习得经验是早期人工智能专家理想的方案，借助统计学，概率学，机器学习开始飞速发展。对数据的处理和学习方案也变得越来越高效。
## 2.顺序
按照李航《统计学习方法》第二版目录，将各代码及算法模型一一归纳整理出来。
# 第一章 统计学习及监督学习概论
# 第二章 感知机
* [如何调用相应的库]
from sklearn import datasets #可导入如iris之类数据库
import numpy as np
from sklearn.cross_validation import train_test_split #划分数据集
from sklearn.preprocessing import StandardScaler #标准化
from sklearn.linear_model import Perceptron  #感知机
from sklearn.metrics import accuracy_score #模型评分
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt  #作图
* [机器学习中一些函数的使用]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)  #X_train 占 0.8

Perceptron类有fit_transform()和predict方法，Perceptro类还提供了partial_fit()方法，允许分类器训练流式数据，并作出预测
clf=Perceptron(fit_intercept=True,n_iter=30,shuffle=False) #不需要设置学习率，大规模学习计算的简单算法
#使用训练数据进行训练
clf.fit(x_data_train,y_data_train)
#得到训练结果，权重矩阵
print(clf.coef_)
print(clf.intercept_)
#利用测试数据进行验证
acc = clf.score(x_data_test,y_data_test)
print(acc)
#画出数据点及超平面
from matplotlib import pyplot as plt
#画出正例和反例的散点图
plt.scatter(positive_x1,positive_x2,c='red')
plt.scatter(negetive_x1,negetive_2,c='blue')
#画出超平面（在本例中即是一条直线）
line_x = np.arange(-4,4)
line_y = line_x * (-clf.coef_[0][0] / clf.coef_[0][1]) - clf.intercept_
plt.plot(line_x,line_y)
plt.show()



