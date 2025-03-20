# Transformer 论文解读



## 关键点

### 自注意力

$$
\operatorname{Attention}(Q, K, V)=\operatorname{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V
$$

其中$d_k$为键向量的维度，用于缩放，防止点积随$d_k$增大而增大，将softmax 推向梯度极小的区域，从2⃣️避免梯度消失

* 计算每个元素之间的相关性，用于捕获全局依赖关系。

### 多头注意力

$$\begin{aligned} \operatorname{MultiHead}(Q, K, V) & =\operatorname{Concat}\left(\operatorname{head}_1, \ldots, \operatorname{head}_{\mathrm{h}}\right)W^O  \\ \text { where head }_i&=\left(Q W_i^Q, K W_i^K, V W_i^V\right)\end{aligned}  $$

其中投影参数矩阵$W_i^Q \in \mathbb{R}^{d_{\text {model }} \times d_k}$， $W_i^K \in \mathbb{R}^{d_{\text {model }} \times d_k}$，$W_i^V \in \mathbb{R}^{d_{\text {model }} \times d_v}$, $W_i^O \in \mathbb{R}^{hd_v \times d_{\text {model }} }$；论文中实际部署中模型输出维度为$d_{\text{model}}=512$，并行个数为$h=8$的并行注意力机制，对应的$d_k=d_v=d_{\text{model}}/h=64$。

归因于降低了每个头的维度，多头注意力机制与使用全部维度的单一头的计算损耗相当。

* 将注意力机制分为多个头，通过不同子空间学习不同维度的特征和之间的关系，增强模型捕捉复杂依赖关系的能力。

  

### 位置编码

$$\begin{aligned}P E_{(p o s, 2 i)}=\sin \left(p o s / 10000^{2 i / d_{\text {model }}}\right)  \\ P E_{(p o s, 2 i+1)}=\cos \left(p o s / 10000^{2 i / d_{\text {model }}}\right) \end{aligned}$$

* 为了解决序列顺序丢失的问题，引入了位置编码（Position Embedding, PE)，使用正弦和余弦函数进行位置编码。

### 前馈神经网络

$$\operatorname{FFN}(x)=\max \left(0, x W_1+b_1\right) W_2+b_2$$

* 前馈神经网络（Feed Forward Network, FFN），使用两个线性层和一个ReLU激活，独立处理每个位置的表示，增强非线性表达能力。

### 编码器-解码器架构

## 参考资料

* https://zhuanlan.zhihu.com/p/365386753
* https://zhuanlan.zhihu.com/p/454482273

* https://jalammar.github.io/illustrated-transformer/