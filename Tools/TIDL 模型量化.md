# 基于 TIDL 模型量化

## 模型量化

量化的目的是将浮点型（float32）的参数映射到整型（int8/int32）的参数空间，一般为$[-128, 127]$。

参考 [QAT](https://arxiv.org/pdf/1712.05877.pdf) 中的公式，首先引入**反量化（dequntization）**公式，可以表示为
$$
r = S(q-Z)
$$
其中 $r$ 表示FP32 的浮点数类型的真实值(**r**eal value)；$q$ 为INT8 的量化后的IN8/INT16 的量化值(**q**uantization value)；$S$表示缩放因子（**s**cale-factor）；$Z$ 表示零点(**z**ero-point)。

对应的可以得出**量化（quantization）**公式如下
$$
q = \lfloor \frac{r}{S} + Z \rceil
$$
参数含义，同上。

容易看出，**量化的目的本质是确定$S$ 和 $Z$ 这两个参数，这两个参数直接决定了量化的精度** 当确定了$S$ 和$Z$我们就可以执行最后的量化操作，完成量化。

假设浮点类型参数的取值空间为$[r_{min}, r_{max}]$，量化后参数的取值空间为$[q_{min}, q_{max}]$，缩放因子$S$ 可以表示为
$$
S = \frac{r_{max} - r_{min}}{s_{max} - s_{min}}
$$
进一步容易得到，零点的公式
$$
Z = -\frac{r_{min}}{S} + q_{min}
$$


## 量化方法

### PTQ量化

* 量化流程

  ```mermaid
  graph TD
  
  A(预训练) -->B(校准)
  B --> |执行量化| C(模型量化)
  C --> D(模型导出)
  ```

  

  1. 预训练：预训练首先以FP32 精度训练模型，得到预训练模型。

  2. 校准(calibration)

     校准具体可以分为两步

     1. 使用小部分数据对FP32模型进行校准，统计网络各层的权重和激活的数据分布（最大最小值）
     2. 使用数据分布特性，计算各层的缩放因子$S$和零点参数$Z$。

  3. 模型量化：使用校准得到的量化参数对FP32模型的各层进行量化。

  4. 模型导出：将模型导出为 onnx 

PQT 量化对于达模型来说很有效，但是对于小模型会导致准确度的显著下降。

### QAT 量化

[QAT](https://arxiv.org/pdf/1712.05877.pdf)  量化算法由Google 提出，通过在训练过程**模拟前向过程中量化的效果**，提高了量化精度。具体来说，QAT 是在模型训练时加入**伪量化（fake quantization）**节点，模拟量化引起的误差。

* 伪量化节点

  假设浮点型参数的取值范范围为$[a, b]$，通过量化级别的熟练和范围截断，量化过程可以参数化为如下公式：
  $$
  clamb(r:a,b) := min(max(r,a),b)
  $$
  结合公式$(3)$ 可以得到缩放因子
  $$
  s(a,b,n):= \frac{b-a}{n-1}
  $$
  

  式中 $n$ 表示量化级别的数量，如8比特量化时$n=2^8=256$
  $$
  q(r:a,b,n) = \lfloor \frac{clamb(r:a,b)}{s(a,b,n)} \rceil s(a,b,n) + a
  $$
  **伪量化节点的**会存储缩放因子和零点的值。**在微调过程中，各层伪量化节点中的缩放因子$S=s(a,b,n)$ 和零点 $Z=z(a,b,n)$会得到估计和更新**。对于卷积层伪量化节点来说

* QAT量化过程

  ![qt-fake-qt](../graph/image-20230302162119468.png)

  上图分别表示推理和训练时的计算图。

  在训练/微调过程中，所有的变量和节点都使用32位浮点进行运算。在需要量化的参数后面插入**伪量化节点**（上图中分别对应 wt quant 和 act quant），来模型量化的效果。同时使用常规的优化算法进行训练。

  在推理过程中，推理框架根据不同类型计算类型可能会对应不同的计算精度。其中卷积和激活算子采用INT8精度；偏置加法只涉及INT32精度。

  *伪量化节点可以看作是通过模型模型量化过程中的取整（round）操作引起的误差，将这种误差看作一种训练噪声。通过fine-tune，让模型去适应这种噪声。从而在模型量化位IN8时，减少由于量化操作造成的精度损失*

* 量化流程

  ```mermaid
  graph TD
  
  A(预训练) -->B(插入伪量化节点)
  B -->  C[微调 QAT 模型]
  C --> |存储量化参数|D(模型量化)
  D --> E(模型导出)
  ```

  1. 预训练：预训练首先以FP32 精度训练模型，得到预训练模型。
  2. 插入伪量化节点：在预训练模型中插入伪量化节点，得到QAT 模型
  3. 微调 QAT 模型：估计和更新各层伪量化节点中的缩放因子$S$ 和零点$Z$等量化参数
  4. 模型量化：使用得到的量化参数，对FP32模型执行量化操作得到量化模型
  5. 模型导出：将模型导出为 onnx 

## TIDL 量化过程

TIDL 库支持两种量化方法后训练量化（Post Training Quantization, PTQ）和量化感知训练(Quantization Aware, QAT ) 量化两种方式。

![qt-compare](../graph/image-20230302110535914.png)



TIDL 提供了 一个封装模块`QuantTrainModule` ，可以去自动化QAT所有的任务。用户只需要使用`QuantTrainModule`封装模型并进行训练。

### QuantTrainModule 处理流程

![qt-tidl-quanti-train-module](../graph/image-20230302185952939.png)

1. 用PACT2 替换模型中的所有RELU、RELU6 层：具体实现中，使用 pytorch 中 nn.Modules 的前向 hook 机制去回调额外的激活函数。这样能够不干扰现有预训练权重的情况下添加这些额外的激活。
2. 权重范围裁剪：如果权重的范围太高，就裁剪权重的值到合适的范围
3. 卷积层与批归一化层合并：在量化训练的前向过程中，将融合卷积层与其相邻的BN 层动态地合并，可以提高缩放因子$S$ 和零点预估准确性，提升量化感知训练的精度。
4. 在量化训练的过程中同时量化权重和激活。

### QAT 训练过程

1. 每次迭代，使用原始的权重和偏差执行前向浮点运算。**这个过程中，PACT2 层会使用直方图和移动平均收集输入的范围，使用统计范围进行裁剪来提高量化精度（与最大-最小范围裁剪相比）**
2. 执行卷积和BN 的合并，并量化合并后得到的权重。这些量化和反量化的权重被用与前向推理。PACT2 收集的范围用于激活量化和量化输出。
3. 使用**STE**反向传播更新参数，去降低量化损失。
4. 使用小的学习率，训练少量 epoch，获得合理的量化精度。

### 关于QAT 部署的推荐和限制规则

* **同一模块不应在模块内重复使用**，以保证特征图范围估计的准确性。这里比如在ResNet 的 BasicBlock 和 BottleneckBlock 中的同一个ReLU在不同层重复使用多次。在执行感知量化前需要对对应模块进行重写，为每一个卷积层分配不同的ReLU激活层。*这里与pytorch 官方QAT量化训练的要求一致*
* **使用`Modules` (继承自nn.Module的类)替换 `funtionals` 或 `operations`**。比如使用 `torch.nn.ReLU` 替换 `torch.nn.functional.relu()`，使用`torch.nn.AdaptiveAvgPool2d()` 替换 `torch.nn.functional.adaptive_avg_pool2d()`， 使用 `torch.nn.Flatten()`替换` torch.nn.functional.flatten()`等。这样做的目的是为了方便量化感知训练过程中的范围收集、卷积层与BN 层的合并等操作。*这里与pytorch 官方QAT量化训练的要求一致*
* **TIDL 量化库自定义的模块**：`xnn.layers.AddBlock`执行元素级加法，`xnn.layers.CatBlock`执行张量拼接。这里是**与第二条规则对应的**，使用自定义的模块去解决，pytorch 官方库中只有`funtionals`，而没有对应`Modules`类的问题。当然，这些**自定义模块也遵循第一条规则**。
* **关于模型导出为onnx**：`Interpolation`/`Upsample`/`Resize`这些操作在pytorch 导出onnx 的过程中有一定的棘手性，必须使用正确的选项才能获得干净的onnx图。TIDL 库提供了这些运算符的`funtionals`形式`xnn.layer.resize_with`和`Modules`形式`xnn.layers.ResizeWith`导出干净的onnx 图。
* **关于训练**：在QAT训练期间的几个 epoch 之后，**冻结BN 和 量化范围对获得更高的量化准确性有益**，实用的函数 `xnn.utils.freeze_bn(model) `和` xnn.layers.freeze_quant_range(model) `可以被用在这里。 
* 其他：如果一个函数不改变特征图的范围，那么是否以`Modules`形式使用它并不重要。比如`torch.nn.functional.interpolate`

## 参考资料

* https://zhuanlan.zhihu.com/p/548174416
* <https://github.com/TexasInstruments/edgeai-torchvision/blob/master/docs/pixel2pixel/Quantization.md>
* https://software-dl.ti.com/jacinto7/esd/processor-sdk-rtos-jacinto7/08_04_00_06/exports/docs/tidl_j721e_08_04_00_16/ti_dl/docs/user_guide_html/md_tidl_fsg_quantization.html>