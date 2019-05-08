



### 深度学习框架对比

| 框架名     | 主语言 | 从语言             | 灵活性 | 上手难易 | 开发者           |
| ---------- | ------ | ------------------ | ------ | -------- | ---------------- |
| Tensorflow | C++    | cuda/python        | 好     | 难       | Google           |
| Caffe      | C++    | cuda/python/Matlab | 一般   | 中等     | 贾杨清           |
| PyTorch    | python | C/C++              | 好     | 中等     | FaceBook         |
| MXNet      | c++    | cuda/R/julia       | 好     | 中等     | 李沐和陈天奇等   |
| Torch      | lua    | C/cuda             | 好     | 中等     | Facebook         |
| Theano     | python | C++/cuda           | 好     | 易       | 蒙特利尔理工学院 |

- 总结：
  - 最常用的框架当数**TensorFlow**和**Pytorch**, 而 Caffe 和 Caffe2 次之。
  - PyTorch 和 Torch 更适用于学术研究（research）；TensorFlow，Caffe，Caffe2 更适用于工业界的生产环境部署（industrial production）
  - Caffe 适用于处理静态图像（static graph）；Torch 和 PyTorch 更适用于动态图像（dynamic graph）；TensorFlow 在两种情况下都很实用。
  - Tensorflow 和 Caffe2 可在移动端使用。





