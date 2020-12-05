#  基于tensorflow中L2的正则化实现

##  前置条件

### 什么是正则化(regularization)

​		如果用一句话解释：正则化就是通过增加权重惩罚(penalty)项到损失函数，让网络倾向于学习小一点的权重，从而达到抑制过拟合，增加模型泛化能力的效果。常见的正则化方法有$L1$正则化，$L2$正则化和Dropout正则化等。关于正则化的原理和作用请参考**深度学习常用正则化方法**

​		本文以$L2$正则化进行实现，为了完整性，这个给出$L2$正则化的公式
$$
L = L_0 + \frac{\lambda}{2}\sum_{i=1}^{n}{(w^2)}
$$
​		式中$L_0$是原始代价损失函数

​                $\frac{\lambda}{2}\sum_{i=1}^{n}{(w^2)}$是$L2$正则化损失函数， 其中$\lambda$是权重因子，$w$为权重

### tensorflow依赖函数

​	在tensorflow 中，计算图(graph)通过集合(collection)来管理包括张量(tensor)、变量(variable)、资源

  * tf.add_to_collection

    将资源添加到特定的集合中

  * tf.get_collection

    从特定集合中取出对应的资源 

* 示例

  ```python
  import tensorflow as tf
  
  # step 1 contruct variable
  v_0 = tf.Variable(tf.constant([1.0, 2.0, 3.0]), name="v_0")
  v_1 = tf.get_variable(shape=(), name="v_1")
  
  # step 2 add variable to collection
  tf.add_to_collection(name="variable", value=v_0)
  tf.add_to_collection(name="variable", value=v_1)
  
  init_op = tf.group(tf.global_variables_initializer(),
  tf.local_variables_initializer())
  with tf.Session() as sess:
      sess.run(init_op)
      # step 3 get variable from collection
      for var in tf.get_collection(key="variable"):
          print('{0}: {1}'.format(var.op.name, var.eval()))
  ```

* 结果

  ```
  v_0: [ 1.  2.  3.]
  v_1: -1.4189265966415405
  ```

### requirement enviroment

* **software**: tensorflow==1.14.0
* **hardware**: GTX 2060

## 正则化到底做了什么

### 理论计算

​		这里假设处理条件：权重$W$ 为 $ [1.0, 2.0, 3.0]$, 正则化因子$\lambda$为$0.00004$

​		根据公式$L2$，正则化损失函数如下

​		$$weight\_loss=\frac{1}{2}*0.00004*(1.0^2+2.0^2+3.0^2)=0.00028$$

### 代码验证

​		这里使用了三种方式计算$L2$正则化，前两种为tensorflow 接口，其中第一种为底层接口，第二种为更高级的接口，不过在tensorflow 2.0中已经抛弃了；第三种为根据公式自定义实现接口。代码如下

```python
weight_decay = 0.00004  # 正则化权重因子
weight = tf.Variable(initial_value=tf.constant(value=[1.0, 2.0, 3.0])) # 权重

# use tensorflow interface
# method 1
weight_loss_1 = tf.nn.l2_loss(weight) * weight_decay
# method 2
weight_loss_2 = tf.contrib.layers.l2_regularizer(scale=weight_decay)(weight)

# cunstom 
# method 3
custom_weight_loss = tf.reduce_sum(tf.multiply(weight, weight))
custom_weight_loss = 1 / 2 * weight_decay * custom_weight_loss

init_op = tf.group(tf.global_variables_initializer(), 
                   tf.local_variables_initializer())
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
sess.run(init_op)
```

### 结果

```
method 1 result: 0.00028
method 2 result: 0.00028
method 3 result: 0.00028
```

可以看出运行结果于理论推导一致，接下来就是如何在网络中加入$L2$正则化，并完成训练。

## 在网络中引入L2正则化

​      上述内容已经介绍了$L2$的基本概念和使用，接下来将介绍如何在神经网络的构建中引入$L2$正则化。$L2$正则化在神经网络中的使用主要包括三个步骤：

* 计算权重的$L2$损失并添加到集合(collection)中 

* 分别取出集合中所有权重的$L2$损失值并相加
* $L2$正则化损失函数与原始代价损失函数相加得到总的损失函数

### 第一步：三种方式收集权重损失函数

* 使用f.nn.l2_loss()接口 与自定义collection 接口

  ```python
  def get_weights_1(shape, weight_decay=0.0, dtype=tf.float32, trainable=True):
      """
      add weight regularization to loss collection
      Args:
          shape: 
          weight_decay: 
          dtype: 
          trainable: 
  
      Returns:
      """
      weight = tf.Variable(initial_value=tf.truncated_normal(shape=shape, stddev=0.01),                          name='Weights', dtype=dtype, trainable=trainable)
      if weight_decay > 0:
          weight_loss = tf.nn.l2_loss(weight) * weight_decay
          # weight_loss = tf.nn.l2_loss(weight, name="weight_loss")
          # tf.add_to_collection(tf.GraphKeys.LOSSES, value=weight_loss)
          tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, value=weight_loss)
      else:
          pass
      return weight
  ```

   分为两步：

  1. 计算正则化损失

     ```python
     weight_loss = tf.nn.l2_loss(weight) * weight_decay
     ```

  2. 将正则化损失添加到特定集合中(这里直接添加到tensorflow内置集合，也可以添加到自定义集合)

     ```python
     tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, value=weight_loss)
     ```

* 使用 tf.contrib.layers.l2_regularizer 与自定义collection 接口

  ```python
  def get_weights_2(shape, weight_decay=0.0, dtype=tf.float32, trainable=True):
      """
  
      Args:
          shape:
          weight_decay:
          dtype:
          trainable:
      Returns:
  
      """
      weight = tf.Variable(initial_value=tf.truncated_normal(shape=shape, stddev=0.01),                          name='Weights', dtype=dtype, trainable=trainable)
      if weight_decay > 0:
          weight_loss = tf.contrib.layers.l2_regularizer(weight_decay)(weight)
          # weight_loss = tf.nn.l2_loss(weight, name="weight_loss")
          # tf.add_to_collection("weight_loss", value=weight_loss)
          tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, value=weight_loss)
      else:
          pass
      return weight
  ```

  分为两步：

  1. 计算正则化损失

     ```python
     weight_loss = tf.contrib.layers.l2_regularizer(weight_decay)(weight)
     ```

  2. 将正则化损失添加到特定集合中(这里直接添加到tensorflow内置集合，也可以添加到自定义集合)

     ```python
     tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, value=weight_loss)
     ```

* 使用tf.contrib.layers.l2_regularizer 与 tf.get_variable接口

  ```python
  def get_weights_3(shape, weight_decay=0.0, dtype=tf.float32, trainable=True):
      """
      add weight to tf.get_variable
      Args:
          shape:
          weight_decay:
          dtype:
          trainable:
      Returns:
  
      """
      # create regularizer
      if weight_decay > 0:
          regularizer= tf.contrib.layers.l2_regularizer(weight_decay)
      else:
          regularizer = None
          weight = tf.get_variable(name='Weights', shape=shape, dtype=dtype,                                              regularizer=regularizer, trainable=trainable)
      return weight
  ```

  分为两步：

  1. 生成正则化器

     ```python
     regularizer= tf.contrib.layers.l2_regularizer(weight_decay)
     ```

  2. 将正则化器参数传入tf.get_variable，tf.get_variable 会内置计算正则化损失函数，并添加到tf.GraphKeys.REGULARIZATION_LOSSES 集合中

     ```python
     weight = tf.get_variable(name='Weights', shape=shape, dtype=dtype,                                              regularizer=regularizer, trainable=trainable)
     ```

### 第二步：从集合中获取权重损失

​		有两种方法可以获取集合中的权重损失函数：

1. 通过tf.get_collection()接口，支持所有的集合遍历，包括内置集合和自定义集合

   ```python
   weight_loss_op = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
   weight_loss_op = tf.add_n(weight_loss_op)
   ```

2. 通过tf.losses.get_regularization_losses()接口，只支持正则化损失收集到REGULARIZATION_LOSSES内置集合的情况

   ```python
   weight_loss_op = tf.losses.get_regularization_losses()
   weight_loss_op = tf.add_n(weight_loss_op)
   ```

两种方法都执行两步：

1. 从特定集合中获取收集的全部权重损失
2. 使用tf.add_n()接口，遍历并相加所有权重损失项，并返回权重损失之和

### 第三步：获取总的损失函数

```python
 with tf.variable_scope("loss"):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, 			                         labels=input_label_placeholder,name='entropy')
        loss_op = tf.reduce_mean(input_tensor=cross_entropy, name='loss')
        weight_loss_op = tf.losses.get_regularization_losses()
        weight_loss_op = tf.add_n(weight_loss_op)
        total_loss_op = loss_op + weight_loss_op
```

## 完整代码示例

​	 下面构建一个完整的包含三层全连接层的网络模型，完成一次迭代训练

### 完整代码

```python
# @ File       : tf_regularization.py
# @ Description: realize regularization base tensorflow
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com

import tensorflow as tf

#+++++++++++++++++++++++++construct 
def fully_connected(input_op, scope, num_outputs, weight_decay=0.00004, is_activation=True, fineturn=True):
    """
     full connect layer
    Args:
        input_op: 
        scope: 
        num_outputs: 
        weight_decay: 
        is_activation: 
        fineturn: 

    Returns:

    """
    # get feature num
    shape = input_op.get_shape().as_list()
    if len(shape) == 4:
        size = shape[-1] * shape[-2] * shape[-3]
    else:
        size = shape[1]
    with tf.compat.v1.variable_scope(scope):
        flat_data = tf.reshape(tensor=input_op, shape=[-1, size], name='Flatten')

        weights =get_weights_1(shape=[size, num_outputs], weight_decay=weight_decay, trainable=fineturn)
        # weights = get_weights_2(shape=[size, num_outputs], weight_decay=weight_decay, trainable=fineturn)
        # weights = get_weights_3(shape=[size, num_outputs], weight_decay=weight_decay, trainable=fineturn)
        biases =get_bias(shape=[num_outputs], trainable=fineturn)

        if is_activation:
             return tf.nn.relu_layer(x=flat_data, weights=weights, biases=biases)
        else:
            return tf.nn.bias_add(value=tf.matmul(flat_data, weights), bias=biases)


def get_bias(shape, trainable=True):
    """
    get bias
    Args:
        shape: 
        trainable: 

    Returns:

    """
    bias = tf.get_variable(shape=shape, name='Bias', dtype=tf.float32, trainable=trainable)

    return bias

def get_weights_1(shape, weight_decay=0.0, dtype=tf.float32, trainable=True):
    """
    add weight regularization to loss collection
    Args:
        shape:
        weight_decay:
        dtype:
        trainable:

    Returns:

    """
    weight = tf.Variable(initial_value=tf.truncated_normal(shape=shape, stddev=0.01), name='Weights', dtype=dtype,
                         trainable=trainable)
    if weight_decay > 0:
        weight_loss = tf.nn.l2_loss(weight) * weight_decay
        # weight_loss = tf.nn.l2_loss(weight, name="weight_loss")
        # tf.add_to_collection(tf.GraphKeys.LOSSES, value=weight_loss)
        tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, value=weight_loss)
    else:
        pass

    return weight


def get_weights_2(shape, weight_decay=0.0, dtype=tf.float32, trainable=True):
    """

    Args:
        shape:
        weight_decay:
        dtype:
        trainable:

    Returns:

    """
    weight = tf.Variable(initial_value=tf.truncated_normal(shape=shape, stddev=0.01), name='Weights', dtype=dtype,
                         trainable=trainable)
    if weight_decay > 0:
        weight_loss = tf.contrib.layers.l2_regularizer(weight_decay)(weight)
        # weight_loss = tf.nn.l2_loss(weight, name="weight_loss")
        # tf.add_to_collection("weight_loss", value=weight_loss)
        tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, value=weight_loss)
    else:
        pass

    return weight


def get_weights_3(shape, weight_decay=0.0, dtype=tf.float32, trainable=True):
    """
    add weight to tf.get_variable
    Args:
        shape:
        weight_decay:
        dtype:
        trainable:
    Returns:

    """
    # create regularizer
    if weight_decay > 0:
        regularizer= tf.contrib.layers.l2_regularizer(weight_decay)
    else:
        regularizer = None
    weight = tf.get_variable(name='Weights', shape=shape, dtype=dtype, regularizer=regularizer,
                              trainable=trainable)
    return weight

#+++++++++++++++++++++++++++++++++consruct network+++++++++++++++++++++++++++++++++++
def model_nets(input_batch, num_classes=None, weight_decay=0.00004, scope="test_nets"):
    """
    full connect network
    Args:
        input_batch: 
        num_classes: 
        weight_decay: 
        scope: 

    Returns:

    """
    with tf.variable_scope(scope):
        net = fully_connected(input_batch, num_outputs=128, weight_decay=weight_decay, scope='fc1')
        net = fully_connected(net, num_outputs=32, weight_decay=weight_decay, scope='fc2')
        net = fully_connected(net, num_outputs=num_classes, is_activation=False, weight_decay=weight_decay, 
                              scope='logits')
        prob = tf.nn.softmax(net, name='prob')
    return prob


#++++++++++++++++++++++++++++++++execute trarin+++++++++++++++++++++++++++++++++
def main():
    
    # parameter config 
    BATCH_SIZE = 10
    DATA_LENGTH = 1024
    NUM_CLASSES = 5
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 0.00004
	
    # inference part
    global_step = tf.train.get_or_create_global_step()
    input_data_placeholder = tf.placeholder(dtype=tf.float32, shape=[None, DATA_LENGTH], name="input_data")
    input_label_placeholder = tf.placeholder(dtype=tf.float32, shape=[None, NUM_CLASSES], name="input_label")
    # inference part
    logits = model_nets(input_batch=input_data_placeholder, num_classes=NUM_CLASSES, weight_decay=WEIGHT_DECAY)

    # calculate loss part
    with tf.variable_scope("loss"):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=input_label_placeholder,
                                                                name='entropy')
        loss_op = tf.reduce_mean(input_tensor=cross_entropy, name='loss')
        weight_loss_op = tf.losses.get_regularization_losses()
        weight_loss_op = tf.add_n(weight_loss_op)
        total_loss_op = loss_op + weight_loss_op

    # generate data and label
    tf.random.set_random_seed(0)
    data_batch = tf.Variable(tf.random_uniform(shape=(BATCH_SIZE, DATA_LENGTH), minval=0, maxval=1, dtype=tf.float32))
    label_batch = tf.Variable(tf.random_uniform(shape=(BATCH_SIZE,), minval=1, maxval=NUM_CLASSES, dtype=tf.int32))
    label_batch = tf.one_hot(label_batch, depth=NUM_CLASSES) # convert label to onehot
	
    
    # initial variable and graph
    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sess.run(init_op)
        input_data, input_label = sess.run([data_batch, label_batch])
		
        print('regularization loss op:')
        for var in tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES):
            print(var.op.name, var.shape)

        # training part
        train_op = tf.train.GradientDescentOptimizer(learning_rate=LEARNING_RATE).minimize(loss=total_loss_op,
                                                                                           global_step=global_step)

        feed_dict = {input_data_placeholder:input_data,
                     input_label_placeholder:input_label}

        _, total_loss, loss, weight_loss = sess.run([train_op, total_loss_op, loss_op, weight_loss_op],
                                                             feed_dict=feed_dict)
        print('loss:{0} weight_loss:{1} total_loss:{2}'.format(loss, weight_loss, total_loss))

 
if __name__ == "__main__":
    main()
```

### 执行结果

```
regularization loss op:
test_nets/fc1/mul ()
test_nets/fc2/mul ()
test_nets/logits/mul ()

loss:1.649293303489685 weight_loss:0.0002091786591336131 total_loss:1.6495025157928467
```



## 参考资料

* [参考资料一](https://www.ritchieng.com/machine-learning/deep-learning/tensorflow/regularization/ "ritchieng")
* [参考资料二](https://stackoverflow.com/questions/37107223/how-to-add-regularizations-in-tensorflow "stackoverflow")
* [参考资料三](https://towardsdatascience.com/regularization-techniques-and-their-implementation-in-tensorflow-keras-c06e7551e709 "towardsdatascience")

