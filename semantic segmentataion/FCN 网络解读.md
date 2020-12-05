# FCN 网络解读

### downsample

### upsample

* Up-Sampling

  bilinear interpolation

* Transpose Conv 

  transpose conv = kernel clockwise 180 + conv with padding

* Up-Pooling

  pooling 的反向操作。需要index,可随机生成，或者记录下采样时的index.

