# YOLO v3 网络解读

## 摘要

## 动机

## 贡献

## 网络结构

### 损失函数

$$
\begin{align}
    loss =& \lambda_{coord}\sum_{i=0}^{S\times S}\sum_{j=0}^{B} I_{ij}^{obj}(2-w_i^j\times h_i^j)(x_i^j -\hat{x}_i^j)(y_i^j -\hat{y}_i^j) \\
&+\lambda_{coord}\sum_{i=0}^{S\times S}\sum_{j=0}^{B} I_{ij}^{obj}(2-w_i^j\times h_i^j)(w_i^j -\hat{w}_i^j)(h_i^j -\hat{h}_i^j) \\
&- \sum_{i=0}^{S\times S}\sum_{j=0}^{B} I_{ij}^{obj}[\hat{C_i}log(C_i^j) + (1-\hat{C}_i^j)log(1-C_i^j)] \\
&- \lambda_{nobj}\sum_{i=0}^{S\times S}\sum_{j=0}^{B} I_{ij}^{nobj}[\hat{C}_i^j log(C_i^j) + (1-\hat{C}_i^j)log(1-C_i^j)] \\
&- \sum_{i=0}^{S\times S}\sum_{j=0}^{B} I_{ij}^{obj}\sum_{c \in classes}[\hat{p}_i^j(c)log(p_i^j(c)) + (1-\hat{p}_i^j(c))log(1-p_i^j(c))] \\
\end{align}
$$

其中边框回归损失为
$$
\begin{align}
loss\_bbox =& \lambda_{coord}\sum_{i=0}^{S\times S}\sum_{j=0}^{B} I_{ij}^{obj}(2-w_i\times h_i)(x_i -\hat{x_{i}})(y_i -\hat{y_{i}}) \\
&+\lambda_{coord}\sum_{i=0}^{S\times S}\sum_{j=0}^{B} I_{ij}^{obj}(2-w_i\times h_i)(w_i -\hat{w_{i}})(h_i -\hat{h_{i}}) \\
\end{align}
$$
目标置信度损失
$$
\begin{align}
loss\_conf =&- \sum_{i=0}^{S\times S}\sum_{j=0}^{B} I_{ij}^{obj}[\hat{C_i}log(C_i) + (1-\hat{C_i})log(1-C_i)] \\
&-\lambda_{nobj}\sum_{i=0}^{S\times S}\sum_{j=0}^{B} I_{ij}^{nobj}[\hat{C_i}log(C_i) + (1-\hat{C_i})log(1-C_i)] \\

\end{align}
$$
分类损失为
$$
\begin{align}
loss\_cls = -\sum_{i=0}^{S\times S}\sum_{j=0}^{B} I_{ij}^{obj}\sum_{c \in classes}[\hat{p_i}(c)log(p_i(c)) + (1-\hat{p_i}(c))log(1-p_i(c))] \\
\end{align}
$$
总的损失函数为
$$
loss = loss\_bbox + loss\_conf + loss\_cls
$$


式中 $S$表示网格框的大小， 对于输入尺寸为$426\times 416$的图片，$S\times S$分别对应$13\times13$, $26\times26$和$52\times 52$三种尺度。

​		$B$表示每个特征点的bbox先验框尺度的数量大小，YOLOv3中一般取值为9。

## 训练细节



## 性能指标

## 参考资料

* <https://pjreddie.com/media/files/papers/YOLOv3.pdf>
* <https://towardsdatascience.com/yolo-v3-object-detection-53fb7d3bfe6b>

* <https://medium.com/@jonathan_hui/what-do-we-learn-from-single-shot-object-detectors-ssd-yolo-fpn-focal-loss-3888677c5f4d>

* <https://zhuanlan.zhihu.com/p/35325884>

