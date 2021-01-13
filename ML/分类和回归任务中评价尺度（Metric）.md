# 分类和回归任务中评价尺度

## 分类任务

### 准确率

### 召回率

### $F_1$分数

### $F_{\beta}$分数

具体可参考另一篇文章[单标签和多标签分类模型的性能度量指标总结](https://zhuanlan.zhihu.com/p/331776896)

## 回归任务

假设测试数据集有 $m$ 个样本， 样本的目标值$y=<y_1, y_2, \cdots y_m>$， 样本的预测值为$\hat{y} = <\hat{y}_1, <\hat{y}_2 \cdots <\hat{y}_m>$ 。 容易得到，观察数据的平均值（目标值的平均）：
$$
\overline{y} = \sum_{i=1}^m{y_i}
$$

###  

### MAE(Mean Absolute Error)

$$
MAE = \frac{1}{m}\sum_{i=1}^m\left|y_i - \hat{y}_i\right|
$$



### MSE(Mean Squared Error)

$$
MSE = \frac{1}{m}\sum_{i=1}^m\left(y_i - \hat{y}_i\right)^2
$$



### RMSE(Mean Squared Error)

$$
RMSE =\sqrt{MSE} = \sqrt{\frac{1}{m}\sum_{i=1}^m\left(y_i - \hat{y}_i\right)_2}
$$



### R-squared

 R-squared 也称为决定系数（coefficient of determination）
$$
R^2 = 1 - \frac{SS_{res}}{SS_{total}}
$$
其中 $SS_{res}$ 称为残差平方和（residual Sum of Square):
$$
SS_{res} = \sum_{i=1}^m \left(y_i-\hat{y}_i \right)^2 = \sum_{i=1}^m {e_i^2}
$$
$SS_{reg}$ 代表回归值与真实值之间的平方差异（回归差异）

$SS_{total}$ 称为总平方和（total Sum of Square）
$$
SS_{total} = \sum_{i=1}^m \left(y_i-\overline{y}_i \right)^2
$$
$SS_{total}$与数据的方差(var)成比例, 代表测试数据集真实值的方差（内部差异）

如果定义回归平方和$SS_{reg}$(Regression/explained Sum of Squared)
$$
SS_{reg} = \sum_{i=1}^m \left(\hat{y}_i-\overline{y}_i \right)^2
$$
容易得到
$$
SS_{total} = SS_{res} + SS_{reg}
$$


### 自适应 R-squared

自适应 R-squared (Adjusted R squared), 自适应于数据集中自变量（特征）的个数， 它的值总数小于等于$R^2$, 公式如下
$$
R_{adj}^2 = 1 - \frac{(1-R^2)({m-1)}}{m-k-1}
$$
其中 $m$ 表示观察数据数量的大小, $k$表示数据集中自变量的数目

## 参考资料

* <https://en.wikipedia.org/wiki/Coefficient_of_determination>