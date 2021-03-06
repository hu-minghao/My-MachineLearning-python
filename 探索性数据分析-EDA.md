### 基于python的探索性数据分析
**Pedro Marcelino - February 2017**
## 1.探索数据中我们应该做什么
理解问题：查看每一个变量，对于进行预先经验的判断，特征的意思及它是否对预测问题重要。
单变量研究：对目标变量进行观察，尽量对目标值理解深入
多变量研究：观察自变量（特征）与变量之间关系
基础的清理工作：处理缺失值，离群值和分类变量
验证假设：对数据进行验证，查看分布是否满足多变量间的假设关系

* 为了使我们的分析更具纪律性，我们可以使用以下几列创建一个Excel电子表格：
该表包含以下几个列：
1.变量名称
2.变量类型，一般可以分为两个类型，数值变量或者类别变量
3.变量分割的类别，在房价预测中，将特征分割为建筑因素，空间因素和位置因素
4.对该变量的期望，该特征对目标值影响，或者关联程度的大小的判断，可以分为高，中等，低等类别
5.结论，看完数据后对该特征的评价。
6.意见，任何能想到的普通意见

## 2.分析目标值
看目标值分布，并分析目标值与你首先假设相关的几个特征之间的关系，对目标预测值有个大概的认识。


## 3.保持冷静，聪明地工作
到目前为止，我们只是按照直觉来分析我们认为重要的变量。尽管我们努力使我们的分析具有客观性，但我们必须说我们的出发点是主观的。
作为工程师，我对主观分析事物的方法不满意。我所有的教育都是关于培养纪律严明，能够承受主观之风的。这是有原因的。在真实世界中，把主观结构尝试加到物理工程中，您将看到物理使它坍塌。这会让你十分难受。
因此，让我们克服惯性并进行更客观的分析。

* 等离子汤
**“一开始除了等离子汤外什么都没有。在我们的宇宙学研究开始时，这些短暂的时刻在很大程度上是推测的。但是，科学根据对当今宇宙的了解，为可能发生的事情设计了一些草图。**

我开始考虑“ YearBuilt”，这使我感到害怕，因为我开始觉得我们应该做一些时间序列分析以实现这一目标。我将把它作为作业留给您。

为您即将看到的东西做好准备。我必须承认，当我第一次看到这些散点图时，我完全被震撼了！在如此短的空间中提供了如此多的信息……真是太神奇了。再次感谢@seaborn！你让我“像老虎一样移动”！

关于“ SalePrice”和“ YearBuilt”的图也可以使我们思考。在“点云”的底部，我们看到了几乎是一个害羞的指数函数（富有创造力）。我们还可以在“点云”的上限中看到这种趋势（甚至更具创造力）。另外，请注意关于过去几年的点集如何趋于保持在此限制之上（我只是想说现在价格上涨得更快）

出于实际原因，这些问题的答案很重要，因为缺少数据可能意味着样本量减少。这可能会阻止我们继续进行分析。而且，从实质的角度来看，我们需要确保丢失的数据过程不存在偏见并掩盖不便的事实

在剩下的情况下，我们可以看到'GarageX'变量具有相同数量的丢失数据。我敢打赌，缺失的数据指的是同一组观察结果（尽管我不会对其进行检查；仅为5％，我们不应该花费20 in5的问题）。由于有关车库的最重要信息是由“ GarageCars”表示的，并且考虑到我们只是在谈论丢失的数据的5％，因此我将删除提及的“ GarageX”变量。相同的逻辑适用于“ BsmtX”变量。这里是说有一组变量丢失值数量一样，那么这组变量是同一个系列的观察值，其中最主要的信息是包含在GarageCars里，所以把这些变量都删除掉。

关于“ MasVnrArea”和“ MasVnrType”，我们可以认为这些变量不是必需的。此外，它们与已经考虑的“ YearBuilt”和“ OverallQual”有很强的相关性。因此，如果删除“ MasVnrArea”和“ MasVnrType”，我们将不会丢失信息。

最后，我们在“电气”中有一个缺失的观察。由于这只是一个观察值，因此我们将删除该观察值并保留变量。

总之，要处理丢失的数据，我们将删除所有带有丢失数据的变量，但变量“ Electrical”除外。在“电子”中，我们将删除缺少数据的观测值。
离群值也是我们应注意的事情。为什么？因为离群值会明显影响我们的模型，并且会成为有价值的信息来源，从而为我们提供了有关特定行为的见解。

目前，我们不会将这些值中的任何一个视为异常值，但应谨慎使用这两个7.something值。

我们已经了解了以下散点图。但是，当我们从新的角度看待事物时，总会有发现的地方。正如艾伦·凯（Alan Kay）所说，“改变观点值得80智商点”。

我们可能会想消除一些观察结果（例如TotalBsmtSF> 3000），但我认为这样做是不值得的。我们可以忍受，所以我们什么也不会做。

请记住，单变量正态性不能确保多元正态性（这是我们希望拥有的），但确实有帮助。

如果我们解决正态性问题，就可以避免很多其他问题（例如异方差问题），这就是我们进行此分析的主要原因。

我只是希望我写对了。同方性是指“假设因变量在预测变量范围内表现出相等的方差水平”
同调性是可取的，因为我们希望误差项在自变量的所有值上都相同。

评估线性的最常见方法是检查散点图并搜索线性模式。如果模式不是线性的，则值得探索数据转换。但是，由于我们所看到的大多数散点图似乎都具有线性关系，因此我们不会对此进行讨论。

缺少相关错误-如定义所示，相关错误发生在一个错误与另一个错误相关时。例如，如果一个正误差系统地产生一个负误差，则意味着这些变量之间存在关系。这通常发生在时间序列中，其中某些模式与时间相关。我们也不会涉及到这一点。但是，如果您检测到某些东西，请尝试添加一个变量，该变量可以解释您得到的效果。这是相关错误的最常见解决方案。

您认为猫王会对这个冗长的解释怎么说？“少聊一点，请多采取一些行动”？大概...顺便问一下，您知道猫王的最后一次成功是什么吗？

我感觉就像霍格沃茨的学生发现了一个新的酷法术

通过圆锥（图形的一侧较小的色散，相对的一侧较大的色散）或菱形（分布中心的大量点）等形状显示了偏离均等色散的情况。

* 下面的一些评论
感谢您的有趣笔记本！但是，在我看来，我们需要注意不要过多地强调可以从这样的EDA获得的见解。主要问题是，通常它最多只能告诉我们有关二维关系的信息，这可能会引起误解。
例如，我将非常谨慎地根据散点图来确定关系是线性的还是二次/指数的，因为一旦我们控制了其他变量，效果可能会大不相同（例如，辛普森悖论）。我并不是说我们甚至不应该查看这些散点图。它们可能有助于获得有关尝试哪些东西的想法，但是我的观点是，我不会决定如何基于这些散点图建模。相反，最好检查残差图以检测任何非线性关系。实际上，我们知道实际的数据生成过程几乎可以肯定是非线性的，因此，要归结为我们可以如何很好地对此模型进行建模，而又不会太大地增加估计的方差。除少量数据集外，性能最佳的预测模型可能会使用诸如回归样条之类的模型对这种非线性进行建模。在实践中，只能通过尝试不同的模型（尤其是具有不同的正则化程度）来确定哪个模型在保留的数据上具有最低的均方误差，从而决定这个问题。
同样，我将使用更正式的过程进行特征选择，并且非常不愿意丢弃任何数据。即使多个预测变量紧密相关，也很难凭直觉说出要使用的阈值。因此，我将它们全部包括在内，并且如果各个变量的回归系数不稳定，那么我会发现如果只使用一个模型，模型的性能会下降多少（再次使用保留的数据）。更好的解决方案是提取主要成分并将其用作预测变量。
此外，虽然我同意检查因变量的直方图，然后获取对数非常重要，因为它的分布是不对称的，但我想强调的是，数据的多元正态性不是回归分析的假设（可能是解决此问题的最自然方法）。相反，它假设误差项是正态分布的（即，给定X时y）。同样，我们需要检查残差图以进行检查。类似地，同质性是指给定X的y的条件分布，因此，如果要在二维上绘制该误差项，则需要再次查看误差项。
我知道EDA经常检查这些事情，但是我不相信我们应该寻找的洞察力无法以更客观的方式告诉我们。我并不是说在进入建模之前我们根本不应该查看数据，但是我通常要寻找的（除了因变量是否不对称分布之外）是数据中的意外模式–例如，截断阈值–否则很难检测到，甚至可能指示数据有问题。（我希望随着时间的推移在此列表中添加更多点…）
