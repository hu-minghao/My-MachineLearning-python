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
# 第八章 提升方法

* 1.scikit-learn中Adaboost类库比较直接，就是AdaBoostClassifier和AdaBoostRegressor两个，从名字就可以看出AdaBoostClassifier用于分类，AdaBoostRegressor用于回归。

　　　　AdaBoostClassifier使用了两种Adaboost分类算法的实现，SAMME和SAMME.R。而AdaBoostRegressor则使用了我们原理篇里讲到的Adaboost回归算法的实现，即Adaboost.R2。
* 2. AdaBoostClassifier和AdaBoostRegressor框架参数
我们首先来看看AdaBoostClassifier和AdaBoostRegressor框架参数。两者大部分框架参数相同
        
        1）base_estimator：AdaBoostClassifier和AdaBoostRegressor都有，即我们的弱分类学习器或者弱回归学习器。理论上可以选择任何一个分类或者回归学习器，不过需要支持样本权重。我们常用的一般是CART决策树或者神经网络MLP。默认是决策树，即AdaBoostClassifier默认使用CART分类树DecisionTreeClassifier，而AdaBoostRegressor默认使用CART回归树DecisionTreeRegressor。另外有一个要注意的点是，如果我们选择的AdaBoostClassifier算法是SAMME.R，则我们的弱分类学习器还需要支持概率预测，也就是在scikit-learn中弱分类学习器对应的预测方法除了predict还需要有predict_proba。

 

　　　　2）algorithm：这个参数只有AdaBoostClassifier有。主要原因是scikit-learn实现了两种Adaboost分类算法，SAMME和SAMME.R。两者的主要区别是弱学习器权重的度量，SAMME使用了和我们的原理篇里二元分类Adaboost算法的扩展，即用对样本集分类效果作为弱学习器权重，而SAMME.R使用了对样本集分类的预测概率大小来作为弱学习器权重。由于SAMME.R使用了概率度量的连续值，迭代一般比SAMME快，因此AdaBoostClassifier的默认算法algorithm的值也是SAMME.R。我们一般使用默认的SAMME.R就够了，但是要注意的是使用了SAMME.R， 则弱分类学习器参数base_estimator必须限制使用支持概率预测的分类器。SAMME算法则没有这个限制。

 

　　　　3）loss：这个参数只有AdaBoostRegressor有，Adaboost.R2算法需要用到。有线性‘linear’, 平方‘square’和指数 ‘exponential’三种选择, 默认是线性，一般使用线性就足够了，除非你怀疑这个参数导致拟合程度不好。这个值的意义在原理篇我们也讲到了，它对应了我们对第k个弱分类器的中第i个样本的误差的处理，即：如果是线性误差，则eki=|yi−Gk(xi)|Ek；如果是平方误差，则eki=(yi−Gk(xi))2E2k，如果是指数误差，则eki=1−exp（−yi+Gk(xi))Ek），Ek为训练集上的最大误差Ek=max|yi−Gk(xi)|i=1,2...m
 

 　　　　4) n_estimators： AdaBoostClassifier和AdaBoostRegressor都有，就是我们的弱学习器的最大迭代次数，或者说最大的弱学习器的个数。一般来说n_estimators太小，容易欠拟合，n_estimators太大，又容易过拟合，一般选择一个适中的数值。默认是50。在实际调参的过程中，我们常常将n_estimators和下面介绍的参数learning_rate一起考虑。

 

　　　　5) learning_rate:  AdaBoostClassifier和AdaBoostRegressor都有，即每个弱学习器的权重缩减系数ν，在原理篇的正则化章节我们也讲到了，加上了正则化项，我们的强学习器的迭代公式为fk(x)=fk−1(x)+ναkGk(x)。ν的取值范围为0<ν≤1。对于同样的训练集拟合效果，较小的ν意味着我们需要更多的弱学习器的迭代次数。通常我们用步长和迭代最大次数一起来决定算法的拟合效果。所以这两个参数n_estimators和learning_rate要一起调参。一般来说，可以从一个小一点的ν开始调参，默认是1。
    
 ## 3. AdaBoostClassifier和AdaBoostRegressor弱学习器参数
 这里我们再讨论下AdaBoostClassifier和AdaBoostRegressor弱学习器参数，由于使用不同的弱学习器，则对应的弱学习器参数各不相同。这里我们仅仅讨论默认的决策树弱学习器的参数。即CART分类树DecisionTreeClassifier和CART回归树DecisionTreeRegressor。

　　　　DecisionTreeClassifier和DecisionTreeRegressor的参数基本类似，在scikit-learn决策树算法类库使用小结这篇文章中我们对这两个类的参数做了详细的解释。这里我们只拿出调参数时需要尤其注意的最重要几个的参数再拿出来说一遍：

　　　　1) 划分时考虑的最大特征数max_features: 可以使用很多种类型的值，默认是"None",意味着划分时考虑所有的特征数；如果是"log2"意味着划分时最多考虑log2N个特征；如果是"sqrt"或者"auto"意味着划分时最多考虑N−−√个特征。如果是整数，代表考虑的特征绝对数。如果是浮点数，代表考虑特征百分比，即考虑（百分比xN）取整后的特征数。其中N为样本总特征数。一般来说，如果样本特征数不多，比如小于50，我们用默认的"None"就可以了，如果特征数非常多，我们可以灵活使用刚才描述的其他取值来控制划分时考虑的最大特征数，以控制决策树的生成时间。

　　　　2) 决策树最大深max_depth: 默认可以不输入，如果不输入的话，决策树在建立子树的时候不会限制子树的深度。一般来说，数据少或者特征少的时候可以不管这个值。如果模型样本量多，特征也多的情况下，推荐限制这个最大深度，具体的取值取决于数据的分布。常用的可以取值10-100之间。

　　　　3) 内部节点再划分所需最小样本数min_samples_split: 这个值限制了子树继续划分的条件，如果某节点的样本数少于min_samples_split，则不会继续再尝试选择最优特征来进行划分。 默认是2.如果样本量不大，不需要管这个值。如果样本量数量级非常大，则推荐增大这个值。

　　　　4) 叶子节点最少样本数min_samples_leaf: 这个值限制了叶子节点最少的样本数，如果某叶子节点数目小于样本数，则会和兄弟节点一起被剪枝。 默认是1,可以输入最少的样本数的整数，或者最少样本数占样本总数的百分比。如果样本量不大，不需要管这个值。如果样本量数量级非常大，则推荐增大这个值。

　　　　5）叶子节点最小的样本权重和min_weight_fraction_leaf：这个值限制了叶子节点所有样本权重和的最小值，如果小于这个值，则会和兄弟节点一起被剪枝。 默认是0，就是不考虑权重问题。一般来说，如果我们有较多样本有缺失值，或者分类树样本的分布类别偏差很大，就会引入样本权重，这时我们就要注意这个值了。

　　　　6) 最大叶子节点数max_leaf_nodes: 通过限制最大叶子节点数，可以防止过拟合，默认是"None”，即不限制最大的叶子节点数。如果加了限制，算法会建立在最大叶子节点数内最优的决策树。如果特征不多，可以不考虑这个值，但是如果特征分成多的话，可以加以限制，具体的值可以通过交叉验证得到。
    
## 例子
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
* from sklearn.ensemble import AdaBoostClassifier
* from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_gaussian_quantiles
 
## 生成2维正态分布，生成的数据按分位数分为两类，500个样本,2个样本特征，协方差系数为2
X1, y1 = make_gaussian_quantiles(cov=2.0,n_samples=500, n_features=2,n_classes=2, random_state=1)
## 生成2维正态分布，生成的数据按分位数分为两类，400个样本,2个样本特征均值都为3，协方差系数为2
X2, y2 = make_gaussian_quantiles(mean=(3, 3), cov=1.5,n_samples=400, n_features=2, n_classes=2, random_state=1)
#讲两组数据合成一组数据
X = np.concatenate((X1, X2))
y = np.concatenate((y1, - y2 + 1))　

        bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2, min_samples_split=20, min_samples_leaf=5),
                         algorithm="SAMME",
                         n_estimators=200, learning_rate=0.8)
        bdt.fit(X, y)
## 做出图
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

Z = bdt.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
cs = plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)
plt.scatter(X[:, 0], X[:, 1], marker='o', c=y)
plt.show()

## GBDT算法（梯度提升树）

* 在sacikit-learn中，GradientBoostingClassifier为GBDT的分类类， 而GradientBoostingRegressor为GBDT的回归类。两者的参数类型完全相同，当然有些参数比如损失函数loss的可选择项并不相同。这些参数中，类似于Adaboost，我们把重要参数分为两类，第一类是Boosting框架的重要参数，第二类是弱学习器即CART回归树的重要参数。

2. GBDT类库boosting框架参数
　　　　首先，我们来看boosting框架相关的重要参数。由于GradientBoostingClassifier和GradientBoostingRegressor的参数绝大部分相同，我们下面会一起来讲，不同点会单独指出。

　　　　    1) n_estimators: 也就是弱学习器的最大迭代次数，或者说最大的弱学习器的个数。一般来说n_estimators太小，容易欠拟合，n_estimators太大，又容易过拟合，一般选择一个适中的数值。默认是100。在实际调参的过程中，我们常常将n_estimators和下面介绍的参数learning_rate一起考虑。

　　　　   2) learning_rate: 即每个弱学习器的权重缩减系数ν，也称作步长，在原理篇的正则化章节我们也讲到了，加上了正则化项，我们的强学习器的迭代公式为fk(x)=fk−1(x)+νhk(x)。ν的取值范围为0<ν≤1。对于同样的训练集拟合效果，较小的ν意味着我们需要更多的弱学习器的迭代次数。通常我们用步长和迭代最大次数一起来决定算法的拟合效果。所以这两个参数n_estimators和learning_rate要一起调参。一般来说，可以从一个小一点的ν开始调参，默认是1。

　　　　   3) subsample: 即我们在原理篇的正则化章节讲到的子采样，取值为(0,1]。注意这里的子采样和随机森林不一样，随机森林使用的是放回抽样，而这里是不放回抽样。如果取值为1，则全部样本都使用，等于没有使用子采样。如果取值小于1，则只有一部分样本会去做GBDT的决策树拟合。选择小于1的比例可以减少方差，即防止过拟合，但是会增加样本拟合的偏差，因此取值不能太低。推荐在[0.5, 0.8]之间，默认是1.0，即不使用子采样。

　　　　   4) init: 即我们的初始化的时候的弱学习器，拟合对应原理篇里面的f0(x)，如果不输入，则用训练集样本来做样本集的初始化分类回归预测。否则用init参数提供的学习器做初始化分类回归预测。一般用在我们对数据有先验知识，或者之前做过一些拟合的时候，如果没有的话就不用管这个参数了。

　　　　   5) loss: 即我们GBDT算法中的损失函数。分类模型和回归模型的损失函数是不一样的。

　　　　　　  对于分类模型，有对数似然损失函数"deviance"和指数损失函数"exponential"两者输入选择。默认是对数似然损失函数"deviance"。在原理篇中对这些分类损失函数有详细的介绍。一般来说，推荐使用默认的"deviance"。它对二元分离和多元分类各自都有比较好的优化。而指数损失函数等于把我们带到了Adaboost算法。

　　　　　　  对于回归模型，有均方差"ls", 绝对损失"lad", Huber损失"huber"和分位数损失“quantile”。默认是均方差"ls"。一般来说，如果数据的噪音点不多，用默认的均方差"ls"比较好。如果是噪音点较多，则推荐用抗噪音的损失函数"huber"。而如果我们需要对训练集进行分段预测的时候，则采用“quantile”。

　　　　   6) alpha：这个参数只有GradientBoostingRegressor有，当我们使用Huber损失"huber"和分位数损失“quantile”时，需要指定分位数的值。默认是0.9，如果噪音点较多，可以适当降低这个分位数的值。
       
* 3. GBDT类库弱学习器参数
　　　　这里我们再对GBDT的类库弱学习器的重要参数做一个总结。由于GBDT使用了CART回归决策树，因此它的参数基本来源于决策树类，也就是说，和DecisionTreeClassifier和DecisionTreeRegressor的参数基本类似。如果你已经很熟悉决策树算法的调参，那么这一节基本可以跳过。不熟悉的朋友可以继续看下去。

　　　　   1) 划分时考虑的最大特征数max_features: 可以使用很多种类型的值，默认是"None",意味着划分时考虑所有的特征数；如果是"log2"意味着划分时最多考虑log2N个特征；如果是"sqrt"或者"auto"意味着划分时最多考虑N−−√个特征。如果是整数，代表考虑的特征绝对数。如果是浮点数，代表考虑特征百分比，即考虑（百分比xN）取整后的特征数。其中N为样本总特征数。一般来说，如果样本特征数不多，比如小于50，我们用默认的"None"就可以了，如果特征数非常多，我们可以灵活使用刚才描述的其他取值来控制划分时考虑的最大特征数，以控制决策树的生成时间。

　　　　   2) 决策树最大深度max_depth: 默认可以不输入，如果不输入的话，默认值是3。一般来说，数据少或者特征少的时候可以不管这个值。如果模型样本量多，特征也多的情况下，推荐限制这个最大深度，具体的取值取决于数据的分布。常用的可以取值10-100之间。

　　　　   3) 内部节点再划分所需最小样本数min_samples_split: 这个值限制了子树继续划分的条件，如果某节点的样本数少于min_samples_split，则不会继续再尝试选择最优特征来进行划分。 默认是2.如果样本量不大，不需要管这个值。如果样本量数量级非常大，则推荐增大这个值。

　　　　   4) 叶子节点最少样本数min_samples_leaf: 这个值限制了叶子节点最少的样本数，如果某叶子节点数目小于样本数，则会和兄弟节点一起被剪枝。 默认是1,可以输入最少的样本数的整数，或者最少样本数占样本总数的百分比。如果样本量不大，不需要管这个值。如果样本量数量级非常大，则推荐增大这个值。

　　　　   5）叶子节点最小的样本权重和min_weight_fraction_leaf：这个值限制了叶子节点所有样本权重和的最小值，如果小于这个值，则会和兄弟节点一起被剪枝。 默认是0，就是不考虑权重问题。一般来说，如果我们有较多样本有缺失值，或者分类树样本的分布类别偏差很大，就会引入样本权重，这时我们就要注意这个值了。

　　　　   6) 最大叶子节点数max_leaf_nodes: 通过限制最大叶子节点数，可以防止过拟合，默认是"None”，即不限制最大的叶子节点数。如果加了限制，算法会建立在最大叶子节点数内最优的决策树。如果特征不多，可以不考虑这个值，但是如果特征分成多的话，可以加以限制，具体的值可以通过交叉验证得到。

　　　　  7) 节点划分最小不纯度min_impurity_split:  这个值限制了决策树的增长，如果某节点的不纯度(基于基尼系数，均方差)小于这个阈值，则该节点不再生成子节点。即为叶子节点 。一般不推荐改动默认值1e-7。

## 实例
import pandas as pd
import numpy as np
## from sklearn.ensemble import GradientBoostingClassifier
from sklearn import cross_validation, metrics
from sklearn.grid_search import GridSearchCV

import matplotlib.pylab as plt
%matplotlib inline

train = pd.read_csv('train_modified.csv')
target='Disbursed' # Disbursed的值就是二元分类的输出
IDcol = 'ID'
train['Disbursed'].value_counts() 

* gbm0 = GradientBoostingClassifier(random_state=10)
* gbm0.fit(X,y)
y_pred = gbm0.predict(X)
y_predprob = gbm0.predict_proba(X)[:,1]
print "Accuracy : %.4g" % metrics.accuracy_score(y.values, y_pred)
print "AUC Score (Train): %f" % metrics.roc_auc_score(y, y_predprob)

* 首先我们从步长(learning rate)和迭代次数(n_estimators)入手。一般来说,开始选择一个较小的步长来网格搜索最好的迭代次数。这里，我们将步长初始值设置为0.1。对于迭代次数进行网格搜索如下：
        
        param_test1 = {'n_estimators':range(20,81,10)}
        gsearch1 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, min_samples_split=300,
                                  min_samples_leaf=20,max_depth=8,max_features='sqrt', subsample=0.8,random_state=10), 
                       param_grid = param_test1, scoring='roc_auc',iid=False,cv=5)
        gsearch1.fit(X,y)
        gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_
* 找到了一个合适的迭代次数，现在我们开始对决策树进行调参。首先我们对决策树最大深度max_depth和内部节点再划分所需最小样本数min_samples_split进行网格搜索。
        
        param_test2 = {'max_depth':range(3,14,2), 'min_samples_split':range(100,801,200)}
        gsearch2 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, n_estimators=60, min_samples_leaf=20, 
      max_features='sqrt', subsample=0.8, random_state=10), 
      param_grid = param_test2, scoring='roc_auc',iid=False, cv=5)
        gsearch2.fit(X,y)
        gsearch2.grid_scores_, gsearch2.best_params_, gsearch2.best_score_

* 由于决策树深度7是一个比较合理的值，我们把它定下来，对于内部节点再划分所需最小样本数min_samples_split，我们暂时不能一起定下来，因为这个还和决策树其他的参数存在关联。下面我们再对内部节点再划分所需最小样本数min_samples_split和叶子节点最少样本数min_samples_leaf一起调参。
 
      param_test3 = {'min_samples_split':range(800,1900,200), 'min_samples_leaf':range(60,101,10)}
        gsearch3 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, n_estimators=60,max_depth=7,
                                     max_features='sqrt', subsample=0.8, random_state=10), 
                       param_grid = param_test3, scoring='roc_auc',iid=False, cv=5)
        gsearch3.fit(X,y)
        gsearch3.grid_scores_, gsearch3.best_params_, gsearch3.best_score_
        
 * 最终参数
 
       gbm1 = GradientBoostingClassifier(learning_rate=0.1, n_estimators=60,max_depth=7, min_samples_leaf =60, 
               min_samples_split =1200, max_features='sqrt', subsample=0.8, random_state=10)
        gbm1.fit(X,y)
        y_pred = gbm1.predict(X)
        y_predprob = gbm1.predict_proba(X)[:,1]
        print "Accuracy : %.4g" % metrics.accuracy_score(y.values, y_pred)
        print "AUC Score (Train): %f" % metrics.roc_auc_score(y, y_predprob)

# 第九章 EM算法
* Em算法为含有未知参数的概率模型的极大似然估计算法，为一种迭代算法。
# 第十章 隐马尔科夫模型
* 隐马尔科夫模型是对时序数据进行建模的模型，建立状态序列与观测序列的联合分布。其中状态序列是不可见的，为隐藏的。其中某一状态只和前一状态有关，且观测值之间相互独立。

## 隐马尔科夫模型的python实现
import numpy as np
## from hmmlearn import hmm

states = ["A", "B", "C"]
n_states = len(states)

observations = ["down","up"]
n_observations = len(observations)

p = np.array([0.7, 0.2, 0.1])
a = np.array([
  [0.5, 0.2, 0.3],
  [0.3, 0.5, 0.2],
  [0.2, 0.3, 0.5]
])
b = np.array([
  [0.6, 0.2],
  [0.3, 0.3],
  [0.1, 0.5]
])
o = np.array([[1, 0, 1, 1, 1]]).T

## model = hmm.MultinomialHMM(n_components=n_states)
## model.startprob_= p
## model.transmat_= a
## model.emissionprob_= b
## logprob, h = model.decode(o, algorithm="viterbi")
## print("The hidden h", ", ".join(map(lambda x: states[x], h)))

这里我们使用了Python的马尔可夫库hmmlearn，可通过命令 $ pip install hmmlearn安装（sklearn的hmm已停止更新，无法正常使用，所以用了hmmlearn库）
 马尔可夫模型λ=(A,B,Π)，A,B,Π是模型的参数，此例中我们直接给出，并填充到模型中，通过观测值和模型的参数，求取隐藏状态。


    EM（Expectation Maximization）最大期望算法是十大数据挖掘经典算法之一。之前一直没见过EM的实现工具和应用场景，直到看见HMM的具体算法。HMM的核心算法是通过观测值计算模型参数，具体使用Baum-Welch算法，它是EM的具体实现，下面来看看EM算法。
    假设条件是X，结果是Y，条件能推出结果X->Y，但结果推不出条件，现在手里有一些对结果Y的观测值，想求X，那么我们举出X的所有可能性，再使用X->Y的公式求Y，看哪个X计算出的Y和当前观测最契合，就选哪个X。这就是最大似然的原理。在数据多的情况下，穷举因计算量太大而无法实现，最大期望EM是通过迭代逼近方式求取最大似然。
    EM算法分为两个步骤：Ｅ步骤是求在当前参数值和样本下的期望函数，M步骤利用期望函数调整模型中的估计值，循环执行E和M直到参数收敛。
    RNN是循环神经网络，LSTM是RNN的一种优化算法，近年来，RNN在很多领域取代了HMM。下面我们来看看它们的异同。
    首先，RNN和HMM解决的都是基于序列的问题，也都有隐藏层的概念，它们都通过隐藏层的状态来生成可观测状态。
    
## 隐马可夫模型常用领域
1.语音识别，使用的就是第一问题，解码问题
2.股票预测，使用的问题2，预测概率问题
3.XSS攻击检测，使用的问题2，预测概率问题
对于XSS攻击，首先我们需要对数据进行泛化,比如：
[a-zA-Z]泛化为A
[0-9]泛化为N
[-_]泛化为C
其他字符泛化为T
其中ANCT为可观测离散值，则对于URL中有字符串uid=admin123，则有:
admin123->AAAAANNN，而uid=%3Cscript->TNAAAAAAA。
假设我们只训练白样本，生成模型，则当识别一个白样本时score值就很高，然后拿去识别XSS，带有XSS黑样本的score值就会很低。

# 第十一章 条件随机场

# 第十二章 监督学习方法总结
监督学习主要方法有感知机（二分类），支持向量机，朴素贝叶斯，决策树，隐马尔科夫模型，提升方法，EM算法，逻辑回归。
监督学习模型主要分为判别模型与生成模型。判别模型生成判别式，生成模型构建输入与输出的联合概率分布，使用函数式表示则为P(x|Y),y=f(x)。
监督学习主要用于分类，标注，回归问题。其中分类问题是根据输入数据，将实例分为二类或多类，分类问题可以看做标注问题的一种特殊形式，标注问题可能需要给出所有标注序列。

# 第十三章 无监督学习
无监督学习使用没有标注的数据进行学习，主要解决问题为聚类，降维，概率估计。
聚类可以理解为数据的纵向比较，即数据可以分为几个类别。
降维可以理解为数据的横向比较。
概率估计则是数据纵向与横向的综合比较。
因此无监督学习常用于数据分析或监督学习之前。

# 第十四章 聚类
* 层次聚类
将样本每个样点分为一类，（单独的一类），将两两最近的类合并，合并依据是以距离度量相似度。常用的距离有闵科夫斯基距离。只到类别数达到要求。-合并法
分裂法则是将所有样本化为一个类，然后按距离最远的两个类分裂，直到类别数量达到要求。这里的距离可以是最远距离，平均距离，中心距离。
* k均值聚类
设几个类（假设k个中心点），然后计算样本到这k个中心点的距离，将样本归到最近中心点代表的类。划分完后重新计算类的中心，再重新划分类，直到中心不再变动。

## 例子层次聚类
import numpy as np
## from sklearn.cluster import AgglomerativeClustering
import random
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

point_random = []
for i in range(10):
    rx = random.uniform(0, 10)
    ry = random.uniform(0, 10)
    point_random.append([rx, ry])

#  簇族 分组量
group_size = 3

## cls = AgglomerativeClustering(n_clusters=group_size,linkage='ward')

## cluster_group = cls.fit(np.array(point_random))

## cnames = ['black', 'blue', 'red']

for point, gp_id in zip(point_random, cluster_group.labels_):
    # 放到 plt 中展示
    plt.scatter(point[0], point[1], s=5, c=cnames[gp_id], alpha=1)

plt.show()

## 进一步可视化
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.cluster import AgglomerativeClustering

from scipy.cluster.hierarchy import dendrogram, linkage

data = np.random.randint(0,10,size=[10,2])

Z = linkage(data)

dendrogram(Z)  

import matplotlib.pyplot as plt  
import pandas as pd  
import numpy as np
import scipy.cluster.hierarchy as shc
customer_data = pd.read_csv(r'.\shopping_data.csv')  


# 获取收入和支出
data = customer_data.iloc[:, 3:5].values  


plt.figure(figsize=(10, 7))  
plt.title("Customer Dendograms")  
dend = shc.dendrogram(shc.linkage(data, method='ward'))  
plt.savefig("./收支-01.jpg")
plt.show()
#%%

from sklearn.cluster import AgglomerativeClustering

cluster = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')  
cluster.fit_predict(data) 



plt.figure(figsize=(10, 7))  
plt.scatter(data[:,0], data[:,1], c=cluster.labels_, cmap='rainbow')  
plt.savefig("./收支-02.jpg")
plt.show()

## 层次聚类算法练习
    import numpy as np
    import pandas as pd
    from sklearn.cluster import AgglomerativeClustering
    import matplotlib.pyplot as plt
    from scipy.spatial.distance import pdist,squareform
    from scipy.cluster.hierarchy import linkage
    from scipy.cluster.hierarchy import dendrogram
    import matplotlib as mpl
    from mpl_toolkits.mplot3d import Axes3D

    np.random.seed(130)
    x=np.random.random_sample([5,3])*10# 生成的随机小于1的数乘以10
    label=['ID_0','ID_1','ID_2','ID_3','ID_4']
    variables=['X','Y','Z']
    df=pd.DataFrame(x,index=label,columns=variables)
    ac = AgglomerativeClustering(n_clusters=2,affinity='euclidean',linkage='complete')
    labels=ac.fit_predict(x)
    df['target']=labels
    print(df)
    row_clusters = linkage(pdist(df,metric='euclidean'),method='complete')#计算欧式距离
    #row_clusters = linkage(df.values,method='complete',metric='euclidean')
    print (pd.DataFrame(row_clusters,columns=['row label1','row label2','distance','no. of items in clust.'],index=['cluster %d'%(i+1) for i in range(row_clusters.shape[0])]))
    #层次聚类树
    row_dendr = dendrogram(row_clusters,labels=labels)
    plt.tight_layout()
    plt.ylabel('Euclidean distance')
    plt.savefig('./001.png')
    plt.show()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(df['X'], df['Y'], df['Z'], c=df['target'])
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()
    
## k均值聚类
接下来学习一下聚类的方法，scikit-learn 关于聚类的算法都包括在了sklearn.cluster方法下面，我们使用k-Means方法来实现找中心。代码如下：

-*- coding: utf-8 -*
## from matplotlib import pyplot as plt 
## from sklearn.cluster import k_means 
## import pandas as pd
## from sklearn.metrics import silhouette_score
## model = k_means(file,n_clusters = 3)
 
cluster_centers = model[0] # 聚类中心数组
 
cluster_labels = model[1] # 聚类标签数组
 
plt.scatter(x, y, c=cluster_labels) # 绘制样本并按聚类标签标注颜色
 
# 绘制聚类中心点，标记成五角星样式，以及红色边框
 
for center in cluster_centers:
 
    plt.scatter(center[0], center[1], marker="p", edgecolors="red")
 
 
 
plt.show() # 显示图#print model


    
    









     
