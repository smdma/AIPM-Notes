

## TensorFlow的特点

官网：<https://www.tensorflow.org/>

- 高度灵活（Deep Flexibility）
  - 它不仅可以用来做神经网络算法研究，也可以用来做普通的机器学习算法，甚至是只要把计算表示成数据流图，都可以用TensorFlow。
- 语言多样（Language Options）
  - TensorFlow使用C++实现的，然后用Python封装。谷歌号召社区通过SWIG开发更多的语言接口来支持TensorFlow。
- 设备支持
  - TensorFlow可以运行在各种硬件上，同时根据计算的需要，合理将运算分配到相应的设备，比如卷积就分配到GPU上，也允许在 CPU 和 GPU 上的计算分布，甚至支持使用 gRPC 进行水平扩展。
- Tensorboard可视化
  - TensorBoard是TensorFlow的一组Web应用，用来监控TensorFlow运行过程，或可视化Computation Graph。TensorBoard目前支持5种可视化：标量（scalars）、图片（images）、音频（audio）、直方图（histograms）和计算图（Computation Graph）。TensorBoard的Events Dashboard可以用来持续地监控运行时的关键指标，比如loss、学习速率（learning rate）或是验证集上的准确率（accuracy）

## TensorFlow的安装

### CPU版本

安装较慢，指定镜像源，请在带有numpy等库的虚拟环境中安装

- ubuntu安装

```
pip install tensorflow==1.8 -i https://mirrors.aliyun.com/pypi/simple
```

- MacOS安装

```
pip install tensorflow==1.8 -i https://mirrors.aliyun.com/pypi/simple
```

### GPU版本

参考官网：

- [在 Ubuntu 上安装 TensorFlow](https://www.tensorflow.org/install/install_linux)
- [在 macOS 上安装 TensorFlow](https://www.tensorflow.org/install/install_mac)

> **CPU与GPU的对比**
>
> CPU：核芯的数量更少；
>
> - 但是每一个核芯的速度更快，性能更强；
>
> - 更适用于处理连续性（sequential）任务。
>
> GPU：核芯的数量更多；
>
> - 但是每一个核芯的处理速度较慢；
>
> - 更适用于并行（parallel）任务。



#### tensorflow中的文件读取流程?

**四个步骤:** 

1. 创建文件名队列
2. 创建和数据类型匹配的读取器并读取样本
3. 对样本数据解码
4. 创建批处理队列



## softmax、交叉熵损失API

- tf.nn.softmax_cross_entropy_with_logits(labels=None, logits=None,name=None)
  - 计算logits和labels之间的交叉损失熵
  - labels:标签值（真实值）
  - logits：样本加权之后的值
  - return:返回损失值列表
- tf.reduce_mean(input_tensor)
  - 计算张量的尺寸的元素平均值



### 卷积网络API

- tf.nn.conv2d(input, filter, strides=, padding=, name=None)

  - 计算给定4-D input和filter张量的2维卷积
  - input：给定的输入张量，具有[batch,heigth,width,channel]，类型为float32,64
  - filter：指定过滤器的权重数量，[filter_height, filter_width, in_channels, out_channels]
  - strides：strides = [1, stride, stride, 1],步长
  - padding：“SAME”, “VALID”，具体解释见下面。

- Tensorflow的零填充方式有两种方式，SAME和VALID

  - SAME：越过边缘取样，取样的面积和输入图像的像素宽度一致。公式 :

     

    ceil(\frac{H}{S})ceil(SH)

    - H 为输入的图片的高或者宽，S为步长。
    - 无论过滤器的大小是多少，零填充的数量由API自动计算。

  - VALID：不越过边缘取样，取样的面积小于输入人的图像的像素宽度。不填充。

  > 在Tensorflow当中，卷积API设置”SAME”之后，如果步长为1，输出高宽与输入大小一样（重要）



### 激活函数API

- tf.nn.relu(features, name=None)
  - features:卷积后加上偏置的结果
  - return:结果

### 池化层API

- tf.nn.max_pool(value, ksize=, strides=, padding=,name=None)
  - 输入上执行最大池数
  - value：4-D Tensor形状[batch, height, width, channels]
  - channel：并不是原始图片的通道数，而是多少filter观察
  - ksize：池化窗口大小，[1, ksize, ksize, 1]
  - strides：步长大小，[1,strides,strides,1]
  - padding：“SAME”， “VALID”，使用的填充算法的类型，默认使用“SAME”





