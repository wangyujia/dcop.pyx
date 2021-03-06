# -*- coding: GB2312 -*-


# 
# [机器学习涉及的概念]
#   有三种主要类型的机器学习：'监督学习'、'非监督学习'和'强化学习'
#       监督学习涉及一组标记数据。监督学习的两种主要类型是：'分类'和'回归'
#           '分类'：机器被训练成将一个组划分为特定的类 (比如垃圾邮件过滤器)
#           '回归'：机器使用先前的(标记的)数据来预测未来 (比如天气预测)
#       无监督学习中，数据是无标签的。无监督学习分为：'聚类'和'降维'
#           '聚类'：用于根据属性和行为对象进行分组，和'分类'不同，这些组不是你提供的 (比如营销方案)
#           '降维'：通过找到共同点来减少数据集的变量。大多数大数据可视化使用降维来识别趋势和规则
#       强化学习使用机器的个人历史和经验来做出决定
#           '强化学习'：经典应用是玩游戏。与监督和非监督学习不同，强化学习不涉及提供“正确的”答案或输出。相反，
#               它只关注性能。这反映了人类是如何根据积极和消极的结果学习的。很快就学会了不要重复这一动作。
#   细化分支：'深度学习'和'神经网络'
# 
# [机器学习涉及的数学：'线性代数'、'微积分'、'概率和统计']
#   线性代数概念：
#       1. 矩阵运算
#       2. 特征值 / 特征向量
#       3. 向量空间和范数
#   微积分概念：
#       1. 偏导数
#       2. 向量 - 值函数
#       3. 方向梯度
#   统计和统计概念：
#       1. Bayes定理
#       2. 组合学；
#       3. 抽样方法
# 
# [机器学习开展步骤]
#   1. 从各种来源收集数据
#   2. 清洗数据具有同质性
#   3. 模型建立 - 选择正确的ML算法
#   4. 从模型的结果中获得见解
#   5. 数据可视化 - 转换结果为可视的图形
# 
# [机器学习涉及的算法]
#   1. 朴素贝叶斯分类器
#       定义：朴素贝叶斯分类器是一系列基于同一个原则的算法，即某一特定特征值独立于任何其它特征值。
#       朴素贝叶斯让我们可以根据我们所知道的相关事件的条件预测事件发生的概率。该名称源于贝叶斯定理，数学公式如下：
#           P(A|B) = P(B|A)P(A) / P(B)
#       其中有事件 A 和事件 B，且 P(B) 不等于 0。可以把它拆解为三部分：
#           a. P(A|B) 是一个条件概率。即在事件 B 发生的条件下事件 A 发生的概率。
#           b. P(B|A) 也是一个条件概率。即在事件 A 发生的条件下事件 B 发生的概率。
#           c. P(A) 和 P(B) 是事件 A 和事件 B 分别发生的概率，其中两者相互独立。
#       所需数学知识：如果你想了解朴素贝叶斯分类器算法的基本原理以及贝叶斯定理的所有用法，一门概率论课程就足够了。
#   2. 线性回归
#       定义：线性回归是最基本的回归类型。它帮助我们理解两个连续变量间的关系。简单的线性回归就是获取一组数据点并绘制可用于预测未来的趋势线。
#       线性回归是参数化机器学习的一个例子。在参数化机器学习中，训练过程使机器学习算法变成一个数学函数，能拟合在训练集中发现的模式。
#       然后可以使用该数学函数来预测未来的结果。在机器学习中，数学函数被称为模型。在线性回归的情况下，模型可以表示为：
#           y = a0 + a1x1 + a2x2 + ... + anxn
#       其中 a1, a2, …, an 表示数据集的特定参数值，x1, x2, …, xn 表示我们选择在最终的模型中使用的特征列，y 表示目标列。
#       线性回归的目标是找到能描述特征列和目标列之间关系的最佳参数值。换句话说，就是找到最能最佳拟合数据的直线，以便根据线的趋势来预测未来结果。
#       为了找到线性回归模型的最佳参数，我们要最小化模型的残差平方和。残差通常也被称为误差，用来描述预测值和真实值之间的差异。
#       残差平方和的公式可以表示为：
#           RSS = (y1 - y'1)^2 + (y2 - y'2)^2 + (yn - y'n)^2
#       其中 y' 是目标列的预测值，y 是真实值。
#       所需数学知识：如果你只想简单了解一下线性回归，学习一门基础统计学的课程就可以了。
#       如果你想对概念有深入的理解，你可能就需要知道如何推导出残差平方和的公式，这在大多数高级统计学课程中都有介绍。
#   3. 逻辑回归
#       定义：Logistic 回归重点关注在因变量取二值（即只有两个值，0 和 1 表示输出结果）的情况下估算发生事件的概率。
#       与线性回归一样，Logistic 回归是参数化机器学习的一个例子。因此，这些机器学习算法的训练结果是得到一个能够最好地近似训练集中模式的数学函数。
#       区别在于，线性回归模型输出的是实数，而 Logistic 回归模型输出的是概率值。
#       正如线性回归算法产生线性函数模型一样，Logistic 回归算法生成 Logistic 函数模型。它也被称作 Sigmoid 函数，会将所有输入值映射为 0 和 1 
#       之间的概率结果。Sigmoid 函数可以表示如下：
#           y = 1 / (1 + e^(-x))
#       那么为什么 Sigmoid 函数总是返回 0 到 1 之间的值呢？请记住，代数中任意数的负数次方等于这个数正数次方的倒数。
#       所需数学知识：我们在这里已经讨论过指数和概率，你需要对代数和概率有充分的理解，以便理解 Logistic 算法的工作原理。
#       如果你想深入了解概念，我建议你学习概率论以及离散数学或实数分析。
#   4. K-Means 聚类
#       定义：K Means 聚类算法是一种无监督机器学习，用于对无标签数据（即没有定义的类别或分组）进行归类。该算法的工作原理是发掘出数据中的聚类簇，
#       其中聚类簇的数量由 k 表示。然后进行迭代，根据特征将每个数据点分配给 k 个簇中的一个。K 均值聚类依赖贯穿于整个算法中的距离概念将数据点「分配」
#       到不同的簇中。距离的概念是指两个给定项之间的空间大小。在数学中，描述集合中任意两个元素之间距离的函数称为距离函数或度量。
#       其中有两种常用类型：欧氏距离和曼哈顿距离。欧氏距离的标准定义如下：
#           d((x1,y1),(x2,y2)) = Math.sqrt((x2-x1)^2 + (y2-y1)^2)
#       其中 (x1,y1) 和 (x2,y2) 是笛卡尔平面上的坐标点。虽然欧氏距离应用面很广，但在某些情况下也不起作用。假设你在一个大城市散步；
#       如果有一个巨大的建筑阻挡你的路线，这时你说「我与目的地相距 6.5 个单位」是没有意义的。为了解决这个问题，我们可以使用曼哈顿距离。
#       曼哈顿距离公式如下：
#           d((x1,y1),(x2,y2)) = Math.abs(x1-x2) + Math.abs(y1-y2)
#       其中 (x1,y1) 和 (x2,y2) 是笛卡尔平面上的坐标点。
#       所需数学知识：实际上你只需要知道加减法，并理解代数的基础知识，就可以掌握距离公式。但是为了深入了解每种度量所包含的基本几何类型，
#       我建议学习一下包含欧氏几何和非欧氏几何的几何学。为了深入理解度量和度量空间的含义，我会阅读数学分析并选修实数分析的课程。
#   5. 决策树
#       定义：决策树是类似流程图的树结构，它使用分支方法来说明决策的每个可能结果。树中的每个节点代表对特定变量的测试，每个分支都是该测试的结果。
#       决策树依赖于信息论的理论来确定它们是如何构建的。在信息论中，人们对某个事件的了解越多，他们能从中获取的新信息就越少。信息论的关键指标之一
#       被称为熵。熵是对给定变量的不确定性量进行量化的度量。熵可以被表示为：
#           Entroppy = sigma(1, n, function(xi){ return P(xi) * log(b, P(xi)); })
#       在上式中，P(xi) 是随机事件 xi 发生的概率。对数的底数 b 可以是任何大于 0 的实数；通常底数的值为 2、e（2.71）和 10。
#       像「S」的花式符号是求和符号，即可以连续地将求和符号之外的函数相加，相加的次数取决于求和的下限和上限。在计算熵之后，我们可以通过利用信息增
#       益开始构造决策树，从而判断哪种分裂方法能最大程度地减少熵。信息增益的公式如下：
#           IG(T,A) = Entroppy(T) - sigma(v in A, Math.abs(Tv)/Math.abs(T)) * Entroppy(Tv)
#       信息增益可以衡量信息量，即获得多少「比特」信息。在决策树的情况下，我们可以计算数据集中每列的信息增益，
#       以便找到哪列将为我们提供最大的信息增益，然后在该列上进行分裂。
#       所需数学知识：想初步理解决策树只需基本的代数和概率知识。如果你想要对概率和对数进行深入的概念性理解，我推荐你学习概率论和代数课程。
#   6. K近邻法 (k-NN)
#   7. 支持向量机 (SVM)
# 


print('你好，世界！')

