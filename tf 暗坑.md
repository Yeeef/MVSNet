# tf 暗坑

- 用 moments 的时候一定要把 cast 到 float32
- batch_size 只能通过 tf.shape(tensor)[0] 获取，且 tf.shape() 不支持拆包操作，获取其他的可以通过 _.get_shape().as_list()
- uint8 的 图片要比 float32 更快
- var.op.name 可以获得一个变量的名称

## tricks

- tf.reshape 过程中如果某一个维度为 -1，则将自动填充

## cuda related

- CUDA driver version is insufficient for CUDA runtime version
  - change tf version

## estimator related

- cannot import name estimator
  - purge all tf versions
  - reinstall compatible tensorflow
  
## on the importance of scope

渐渐明白 name_scope 存在的意义了，我们希望 tensorboard 里的图片更好看，同时也希望 get_variable 的名字能够更加简短, 我之前
是通过 `tf.identity` 实现的，不知道是否会造成多余的复制？

## tf.identity

- [stackoverflow](https://stackoverflow.com/questions/34877523/in-tensorflow-what-is-tf-identity-used-for)

## tf.layers.batch_normalization

一定要注意 axis 的设置

## tf.nn.embedding_lookup

这个函数小复杂，最主要的参数是 `params` 和 `ids`, 顾名思义，要通过 ids 里的 id 去一个 embedding 矩阵找对应的向量。就是 ids 的维度并不限于 (b,), 所以最终输出的维度是 list(ids.shape) + list(params.shape[1:]), embedding 被加在了最后一个维度

## namescope and variable scope 

difference 就不说了，name_scope 有用的一点就在于构造更好看的 graph, 但是变量命名没有加前缀，貌似也不是这样，这个地方行为确实奇怪。。。fc 在外边，drop out 又在里面。。。

## tf.layers.dropout

tensorpack 会自动帮我在 inferenceRunner 中变成一个 tf.identity 非常神奇

