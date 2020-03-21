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
* 使用白盒模型。如果某种给定的情况在该模型中是可以观察的，那么就可以轻易的通过布尔逻辑来解释这种情况。相比之下，在黑盒模型中的结果就是很难说明清 楚地。
决策树的缺点
* 决策树模型容易产生一个过于复杂的模型,这样的模型对数据的泛化性能会很差。这就是所谓的过拟合.一些策略像剪枝、设置叶节点所需的最小样本数或设置数的最大深度是避免出现 该问题最为有效地方法。

DecisionTreeClassifier 是能够在数据集上执行多分类的类,与其他分类器一样，DecisionTreeClassifier 采用输入两个数组：数组X，用 [n_samples, n_features] 的方式来存放训练样本。整数值数组Y，用 [n_samples] 来保存训练样本的类标签:

>>> from sklearn import tree
>>> X = [[0, 0], [1, 1]]
>>> Y = [0, 1]
>>> clf = tree.DecisionTreeClassifier()
>>> clf = clf.fit(X, Y)
>>> clf.predict([[2., 2.]])

另外，也可以预测每个类的概率，这个概率是叶中相同类的训练样本的分数:
## clf.predict_proba([[2., 2.]])
* DecisionTreeClassifier 既能用于二分类（其中标签为[-1,1]）也能用于多分类（其中标签为[0,…,k-1]）。使用Lris数据集，我们可以构造一个决策树
经过训练，我们可以使用 export_graphviz 导出器以 Graphviz 格式导出决策树. 如果你是用 conda 来管理包，那么安装 graphviz 二进制文件和 python 包可以用以下指令安装

conda install python-graphviz
或者，可以从 graphviz 项目主页下载 graphviz 的二进制文件，并从 pypi 安装: Python 包装器，并安装 pip install graphviz .以下是在整个 iris 数据集上训练的上述树的 graphviz 导出示例; 其结果被保存在 iris.pdf 中:

>>> import graphviz
>>> dot_data = tree.export_graphviz(clf, out_file=None)
>>> graph = graphviz.Source(dot_data)
>>> graph.render("iris")

决策树通过使用 DecisionTreeRegressor 类也可以用来解决回归问题。如在分类设置中，拟合方法将数组X和数组y作为参数，只有在这种情况下，y数组预期才是浮点值:

>>> from sklearn import tree
>>> X = [[0, 0], [2, 2]]
>>> y = [0.5, 2.5]
>>> clf = tree.DecisionTreeRegressor()
>>> clf = clf.fit(X, y)
>>> clf.predict([[1, 1]])
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
对于拥有大量特征的数据决策树会出现过拟合的现象。获得一个合适的样本比例和特征数量十分重要，因为在高维空间中只有少量的样本的树是十分容易过拟合的。
考虑事先进行降维( PCA , ICA ，使您的树更好地找到具有分辨性的特征。
通过 export 功能可以可视化您的决策树。使用 max_depth=3 作为初始树深度，让决策树知道如何适应您的数据，然后再增加树的深度。
请记住，填充树的样本数量会增加树的每个附加级别。使用 max_depth 来控制输的大小防止过拟合。
通过使用 min_samples_split 和 min_samples_leaf 来控制叶节点上的样本数量。当这个值很小时意味着生成的决策树将会过拟合，然而当这个值很大时将会不利于决策树的对样本的学习。所以尝试 min_samples_leaf=5 作为初始值。如果样本的变化量很大，可以使用浮点数作为这两个参数中的百分比。两者之间的主要区别在于 min_samples_leaf 保证叶结点中最少的采样数，而 min_samples_split 可以创建任意小的叶子，尽管在文献中 min_samples_split 更常见。
在训练之前平衡您的数据集，以防止决策树偏向于主导类.可以通过从每个类中抽取相等数量的样本来进行类平衡，或者优选地通过将每个类的样本权重 (sample_weight) 的和归一化为相同的值。还要注意的是，基于权重的预修剪标准 (min_weight_fraction_leaf) 对于显性类别的偏倚偏小，而不是不了解样本权重的标准，如 min_samples_leaf 。
如果样本被加权，则使用基于权重的预修剪标准 min_weight_fraction_leaf 来优化树结构将更容易，这确保叶节点包含样本权重的总和的至少一部分。
所有的决策树内部使用 np.float32 数组 ，如果训练数据不是这种格式，将会复制数据集。
如果输入的矩阵X为稀疏矩阵，建议您在调用fit之前将矩阵X转换为稀疏的csc_matrix ,在调用predict之前将 csr_matrix 稀疏。当特征在大多数样本中具有零值时，与密集矩阵相比，稀疏矩阵输入的训练时间可以快几个数量级。

