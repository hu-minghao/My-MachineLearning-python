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
## clf=Perceptron(fit_intercept=True,n_iter=30,shuffle=False) #不需要设置学习率，大规模学习计算的简单算法
#使用训练数据进行训练
## clf.fit(x_data_train,y_data_train)
#得到训练结果，权重矩阵
## print(clf.coef_)
## print(clf.intercept_)
#利用测试数据进行验证
## acc = clf.score(x_data_test,y_data_test)
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

# 第三章 K临近分类算法
基于给于的数据集，对需要预测的数据，按照周围最近的k个样本点中，最多类别的点来划分。
最临近算法为一种非泛式学习法，是将带标记的数据集简单的记住，然后在最近的样本中投票，选出待测点的分类。
* 优点：精度高、对异常值不敏感、无数据输入假定。
* 缺点：计算复杂度高、空间复杂度高。 适用数据范围：数值型和标称型。
其中k为超参数，另外影响该算法结果的有是否考虑距离，记录数据的算法

*algorithm：快速k近邻搜索算法，默认参数为auto，可以理解为算法自己决定合适的搜索算法。除此之外，用户也可以自己指定搜索算法ball_tree、kd_tree、brute方法进行搜索，brute是蛮力搜索，也就是线性扫描，当训练集很大时，计算非常耗时。kd_tree，构造kd树存储数据以便对其进行快速检索的树形数据结构，kd树也就是数据结构中的二叉树。以中值切分构造的树，每个结点是一个超矩形，在维数小于20时效率高。ball tree是为了克服kd树高纬失效而发明的，其构造过程是以质心C和半径r分割样本空间，每个节点是一个超球体。 

# 第四章 朴素贝叶斯模型
利用样本构造一个统计概率模型，认为样本各属性间是独立的。
* sklearn中有高斯贝叶斯模型，多项式贝叶斯模型等
>>> from sklearn import datasets
>>> iris = datasets.load_iris()
>>> from sklearn.naive_bayes import GaussianNB
>>> gnb = GaussianNB()
>>> y_pred = gnb.fit(iris.data, iris.target).predict(iris.data)
>>> print("Number of mislabeled points out of a total %d points : %d"
...       % (iris.data.shape[0],(iris.target != y_pred).sum()))

# 第五章 决策树
决策树的优点
* Decision Trees (DTs) 是一种用来 classification 和 regression 的无参监督学习方法。其目的是创建一种模型从数据特征中学习简单的决策规则来预测一个目标变量的值。
* 使用白盒模型。如果某种给定的情况在该模型中是可以观察的，那么就可以轻易的通过布尔逻辑来解释这种情况。相比之下，在黑盒模型中的结果就是很难说明清楚地。
决策树的缺点
* 决策树模型容易产生一个过于复杂的模型,这样的模型对数据的泛化性能会很差。这就是所谓的过拟合.一些策略像剪枝、设置叶节点所需的最小样本数或设置数的最大深度是避免出现 该问题最为有效地方法。

DecisionTreeClassifier 是能够在数据集上执行多分类的类,与其他分类器一样，DecisionTreeClassifier 采用输入两个数组：数组X，用 [n_samples, n_features] 的方式来存放训练样本。整数值数组Y，用 [n_samples] 来保存训练样本的类标签:

* from sklearn import tree
*  X = [[0, 0], [1, 1]]
*  Y = [0, 1]
*  clf = tree.DecisionTreeClassifier()
*  clf = clf.fit(X, Y)
*  clf.predict([[2., 2.]])

另外，也可以预测每个类的概率，这个概率是叶中相同类的训练样本的分数:
## clf.predict_proba([[2., 2.]])
* DecisionTreeClassifier 既能用于二分类（其中标签为[-1,1]）也能用于多分类（其中标签为[0,…,k-1]）。使用Lris数据集，我们可以构造一个决策树
经过训练，我们可以使用 export_graphviz 导出器以 Graphviz 格式导出决策树. 如果你是用 conda 来管理包，那么安装 graphviz 二进制文件和 python 包可以用以下指令安装

conda install python-graphviz
或者，可以从 graphviz 项目主页下载 graphviz 的二进制文件，并从 pypi 安装: Python 包装器，并安装 pip install graphviz .以下是在整个 iris 数据集上训练的上述树的 graphviz 导出示例; 其结果被保存在 iris.pdf 中:

*  import graphviz
*  dot_data = tree.export_graphviz(clf, out_file=None)
*  graph = graphviz.Source(dot_data)
*  graph.render("iris")

决策树通过使用 DecisionTreeRegressor 类也可以用来解决回归问题。如在分类设置中，拟合方法将数组X和数组y作为参数，只有在这种情况下，y数组预期才是浮点值:

*  from sklearn import tree
*  X = [[0, 0], [2, 2]]
*  y = [0.5, 2.5]
*  clf = tree.DecisionTreeRegressor()
*  clf = clf.fit(X, y)
*  clf.predict([[1, 1]])
array([ 0.5])

一个多值输出问题是一个类似当 Y 是大小为 [n_samples, n_outputs] 的2d数组时，有多个输出值需要预测的监督学习问题。

当输出值之间没有关联时，一个很简单的处理该类型的方法是建立一个n独立模型，即每个模型对应一个输出，然后使用这些模型来独立地预测n个输出中的每一个。然而，由于可能与相同输入相关的输出值本身是相关的，所以通常更好的方法是构建能够同时预测所有n个输出的单个模型。首先，因为仅仅是建立了一个模型所以训练时间会更短。第二，最终模型的泛化性能也会有所提升。对于决策树，这一策略可以很容易地用于多输出问题。 这需要以下更改：

在叶中存储n个输出值，而不是一个;
通过计算所有n个输出的平均减少量来作为分裂标准.
该模块通过在 DecisionTreeClassifier 和 DecisionTreeRegressor 中实现该策略来支持多输出问题。如果决策树与大小为 [n_samples, n_outputs] 的输出数组Y向匹配，则得到的估计器:

predict 是输出n_output的值
在 predict_proba 上输出 n_output 数组列表
用多输出决策树进行回归分析 Multi-output Decision Tree Regression 。 在该示例中，输入X是单个实数值，并且输出Y是X的正弦和余弦。

* CART（Classification and Regression Trees （分类和回归树））与 C4.5 非常相似，但它不同之处在于它支持数值目标变量（回归），并且不计算规则集。CART 使用在每个节点产生最大信息增益的特征和阈值来构造二叉树。scikit-learn 使用 CART 算法的优化版本。
## 使用技巧
对于拥有大量特征的数据决策树会出现过拟合的现象。获得一个合适的样本比例和特征数量十分重要，因为在高维空间中只有少量的样本的树是十分容易过拟合的。\
考虑事先进行降维( PCA , ICA ，使您的树更好地找到具有分辨性的特征。\
通过 export 功能可以可视化您的决策树。使用 max_depth=3 作为初始树深度，让决策树知道如何适应您的数据，然后再增加树的深度。\
请记住，填充树的样本数量会增加树的每个附加级别。使用 max_depth 来控制输的大小防止过拟合。\
通过使用 min_samples_split 和 min_samples_leaf 来控制叶节点上的样本数量。当这个值很小时意味着生成的决策树将会过拟合，然而当这个值很大时将会不利于决策树的对样本的学习。所以尝试 min_samples_leaf=5 作为初始值。如果样本的变化量很大，可以使用浮点数作为这两个参数中的百分比。两者之间的主要区别在于 min_samples_leaf 保证叶结点中最少的采样数，而 min_samples_split 可以创建任意小的叶子，尽管在文献中 min_samples_split 更常见。\
在训练之前平衡您的数据集，以防止决策树偏向于主导类.可以通过从每个类中抽取相等数量的样本来进行类平衡，或者优选地通过将每个类的样本权重 (sample_weight) 的和归一化为相同的值。还要注意的是，基于权重的预修剪标准\
* (min_weight_fraction_leaf) 对于显性类别的偏倚偏小，而不是不了解样本权重的标准，如 min_samples_leaf 。
如果样本被加权，则使用基于权重的预修剪标准 min_weight_fraction_leaf 来优化树结构将更容易，这确保叶节点包含样本权重的总和的至少一部分。\
所有的决策树内部使用 np.float32 数组 ，如果训练数据不是这种格式，将会复制数据集。\
如果输入的矩阵X为稀疏矩阵，建议您在调用fit之前将矩阵X转换为稀疏的csc_matrix ,在调用predict之前将 csr_matrix 稀疏。当特征在大多数样本中具有零值时，与密集矩阵相比，稀疏矩阵输入的训练时间可以快几个数量级。\

# 第六章 逻辑回归与最大熵模型
## from sklearn.linear_model import LogisticRegression as LR
* help(LR)
sklearn.linear_model.LogisticRegression(  
    penalty=’l2’, dual=False,   
    tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1,   
    class_weight=None, random_state=None, solver=’warn’,  
    max_iter=100, multi_class=’warn’, verbose=0,   
    warm_start=False, n_jobs=None, l1_ratio=None)
   
* penalty：惩罚项
* str类型，默认为l2。newton-cg、sag和lbfgs求解算法只支持L2规范,L2假设的模型参数满足高斯分布。
l1:L1G规范假设的是模型的参数满足拉普拉斯分布.
* dual：对偶或原始方法，bool类型，默认为False。对偶方法只用在求解线性多核(liblinear)的L2惩罚项上。当样本数量>样本特征的时候，dual通常设置为False。
* tol：停止求解的标准，float类型，默认为1e-4。就是求解到多少的时候，停止，认为已经求出最优解。
* c：正则化系数λ的倒数，float类型，默认为1.0。必须是正浮点型数。像SVM一样，越小的数值表示越强的正则化。
* fit_intercept：是否存在截距或偏差，bool类型，默认为True。
* intercept _ scaling：仅在正则化项为”liblinear”，且fit_intercept设置为True时有用。float类型，默认为1。
* class_ weight：用于标示分类模型中各种类型的权重，可以是一个字典或者’balanced’字符串，默认为不输入，也就是不考虑权重，即为None。
如果选择输入的话，可以选择balanced让类库自己计算类型权重，或者自己输入各个类型的权重。
举个例子，比如对于0,1的二元模型，我们可以定义class_ weight = {0:0.9,1:0.1}，这样类型0的权重为90%，而类型1的权重为10%。如果class_ weight选择balanced，那么类库会根据训练样本量来计算权重。
某种类型样本量越多，则权重越低，样本量越少，则权重越高。
当class_ weight为balanced时，类权重计算方法如下：n_samples / (n_classes * np.bincount(y))。n_samples为样本数，n_classes为类别数量，np.bincount(y)会输出每个类的样本数，例如y=[1,0,0,1,1],则np.bincount(y)=[2,3]。
其实这个数据平衡的问题我们有专门的解决办法：重采样
* random_state：随机数种子，int类型，可选参数，默认为无，仅在正则化优化算法为sag,liblinear时有用。
* solver：优化算法选择参数
* liblinear：使用了开源的liblinear库实现，内部使用了坐标轴下降法来迭代优化损失函数。
* lbfgs：拟牛顿法的一种，利用损失函数二阶导数矩阵即海森矩阵来迭代优化损失函数。
* newton-cg：也是牛顿法家族的一种，利用损失函数二阶导数矩阵即海森矩阵来迭代优化损失函数。只用于L2
* sag：即随机平均梯度下降，是梯度下降法的变种，和普通梯度下降法的区别是每次迭代仅仅用一部分的样本来计算梯度，适合于样本数据多的时候。只用于L2
* saga：线性收敛的随机优化算法的的变重。只用于L2
* max_iter：算法收敛最大迭代次数，int类型，默认为10。仅在正则化优化算法为newton-cg, sag和lbfgs才有用，算法收敛的最大迭代次数。
* multi_class：分类方式选择参数，str类型，可选参数为ovr和multinomial，默认为ovr。ovr即前面提到的one-vs-rest(OvR)，而multinomial即前面提到的many-vs-many(MvM)。如果是二元逻辑回归，ovr和multinomial并没有任何区别，区别主要在多元逻辑回归上。
* verbose：日志冗长度，int类型。默认为0。就是不输出训练过程，1的时候偶尔输出结果，大于1，对于每个子模型都输出。
* warm_start：热启动参数，bool类型。默认为False。如果为True，则下一次训练是以追加树的形式进行（重新使用上一次的调用作为初始化）。
* n_jobs：并行数。int类型，默认为1。为-1的时候，用所有CPU的内核运行程序。
## 例子
from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

iris = datasets.load_iris()
X = iris.data[:,[2,3]]  ## 选取花瓣长度和花瓣宽度两个特征
y = iris.target
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.3,train_size=0.7,random_state=0)
sc = StandardScaler()
sc.fit(x_train)  # 计算均值和方差
x_train_std = sc.transform(x_train) #利用计算好的方差和均值进行Z分数标准化
x_test_std = sc.transform(x_test)

lr = LR(C=1000,random_state=123)
lr.fit(x_train_std,y_train)
lr.score(u"正确率：",x_test_std,y_test) # 输出一个正确率
y_pred = lr.predict(x_test_std)
print(u"混淆矩阵",confusion_matrix(y_true=y_test,y_pred=y_pred))
print("正确率：",AC(y_test,y_pred))

lr.coef_

## sklearn中暂时没有最大熵模型

# 第七章 支持向量机
例子
from sklearn import metrics
## from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import numpy as np

## 读取数据
X = []
Y = []
fr = open("testSetRBF.txt")
index = 0
for line in fr.readlines():
    line = line.strip()
    line = line.split('\t')
    X.append(line[:2])
    Y.append(line[-1])

plt.scatter(np.array(X)[:,0],np.array(X)[:,1])
plt.show()
#归一化
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# 交叉分类
train_X,test_X, train_y, test_y = train_test_split(X,
                                                   Y,
                                                   test_size=0.2) # test_size:测试集比例20%

## SVM模型，选择3个邻居
## model = SVC(kernel='rbf', degree=2, gamma=1.7)
## model.fit(train_X, train_y)
print(model)

expected = test_y
## predicted = model.predict(test_X)
print(metrics.classification_report(expected, predicted))       # 输出分类信息
label = list(set(Y))    # 去重复，得到标签类别
### print(metrics.confusion_matrix(expected, predicted, labels=label))  # 输出混淆矩阵信息

## 第二个例子，sklearn中支持向量机分线性核，多项式核函数和径向核
'''
    支持向量机：
        支持向量机原理：
            ** 分类原则：寻求最优分类边界
                1.正确：对大部分样本可以正确地划分类别。
                2.泛化：最大化支持向量间距。
                3.公平：与支持向量等距。
                4.简单：线性，直线或平面，分割超平面。

        ** 基于核函数的升维变换:通过名为核函数的特征变换，增加新的特征，使得低维度空间中的线性不可分问题变为高维度空间中的线性可分问题。

            1>线性核函数：linear，不通过核函数进行维度提升，仅在原始维度空间中寻求线性分类边界。

            2>多项式核函数：poly，通过多项式函数增加原始样本特征的高次方幂
                    y = x_1+x_2
                    y = x_1^2 + 2x_1x_2 + x_2^2
                    y = x_1^3 + 3x_1^2x_2 + 3x_1x_2^2 + x_2^3

            3>径向基核函数：rbf，通过高斯分布函数增加原始样本特征的分布概率

        基于线性核函数的SVM分类相关API：
                model = svm.SVC(kernel='linear')
                model.fit(train_x, train_y)

        案例，基于径向基核函数训练sample2.txt中的样本数据。
            步骤：
                1.读取文件，绘制样本点的分布情况
                2.拆分测试集合训练集
                3.基于svm训练分类模型
                4.输出分类效果，绘制分类边界
'''
import numpy as np
import sklearn.model_selection as ms
## import sklearn.svm as svm
## import sklearn.metrics as sm（计算模型精度）
## import matplotlib.pyplot as mp

data = np.loadtxt('./ml_data/multiple2.txt', delimiter=',', unpack=False, dtype='f8')
x = data[:, :-1]
y = data[:, -1]

## 才分训练集和测试集
train_x, test_x, train_y, test_y = ms.train_test_split(x, y, test_size=0.25, random_state=5)

## 训练svm模型---基于线性核函数
## model = svm.SVC(kernel='linear')
## model.fit(train_x, train_y)

## 训练svm模型---基于多项式核函数
## model = svm.SVC(kernel='poly', degree=3)
## model.fit(train_x, train_y)

## 训练svm模型---基于径向基核函数
model = svm.SVC(kernel='rbf', C=600)
model.fit(train_x, train_y)

## 预测
pred_test_y = model.predict(test_x)

## 计算模型精度
## bg = sm.classification_report(test_y, pred_test_y)
print('分类报告：', bg, sep='\n')

## 绘制分类边界线
l, r = x[:, 0].min() - 1, x[:, 0].max() + 1
b, t = x[:, 1].min() - 1, x[:, 1].max() + 1
n = 500
grid_x, grid_y = np.meshgrid(np.linspace(l, r, n), np.linspace(b, t, n))
bg_x = np.column_stack((grid_x.ravel(), grid_y.ravel()))
bg_y = model.predict(bg_x)
grid_z = bg_y.reshape(grid_x.shape)

## 画图显示样本数据
mp.figure('SVM Classification', facecolor='lightgray')
mp.title('SVM Classification', fontsize=16)
mp.xlabel('X', fontsize=14)
mp.ylabel('Y', fontsize=14)
mp.tick_params(labelsize=10)
mp.pcolormesh(grid_x, grid_y, grid_z, cmap='gray')
mp.scatter(test_x[:, 0], test_x[:, 1], s=80, c=test_y, cmap='jet', label='Samples')

mp.legend()
mp.show()



输出结果：
分类报告：
              precision    recall  f1-score   support

         0.0       0.91      0.87      0.89        45
         1.0       0.81      0.87      0.84        30

    accuracy                           0.87        75
   macro avg       0.86      0.87      0.86        75
weighted avg       0.87      0.87      0.87        75

''' sklearn中支持向量机算法很丰富，这里就不一一描述了。
