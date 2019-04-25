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
  - resinstall compatibel tensorflow
