# FCN 网络解读

### downsample

### upsample

* Up-Sampling

  bilinear interpolation

* Transpose Conv 

  transpose conv = kernel clockwise 180 + conv with padding

* Up-Pooling

  pooling 的反向操作。需要index,可随机生成，或者记录下采样时的index.

$$
W_{out} = \left\lfloor\frac{W_{in}  + 2 \times \text{padding}[1] - \text{dilation}[1]
                        \times (\text{kernel\_size}[1] - 1) - 1}{\text{stride}[1]} + 1\right\rfloor
$$



