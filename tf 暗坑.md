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