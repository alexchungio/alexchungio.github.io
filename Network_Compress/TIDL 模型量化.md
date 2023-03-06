# 基于 TIDL 模型量化

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

* <https://github.com/TexasInstruments/edgeai-torchvision/blob/master/docs/pixel2pixel/Quantization.md>
* https://software-dl.ti.com/jacinto7/esd/processor-sdk-rtos-jacinto7/08_04_00_06/exports/docs/tidl_j721e_08_04_00_16/ti_dl/docs/user_guide_html/md_tidl_fsg_quantization.html>