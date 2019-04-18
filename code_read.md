# code reading

## notations

- `tf.nn.layers.conv2d` reuse parameter: Boolean, whether to reuse the weights of a previous layer by the same name., [doc](https://www.tensorflow.org/api_docs/python/tf/layers/conv2d)

### batch normalization should be paid significant attention

`tf.layers.batch_normalization` [tf 1.13 doc](https://www.tensorflow.org/api_docs/python/tf/layers/batch_normalization), 其中提到一点重要信息，_when you use low-level API, be careful to manipulate update ops_

    when training, the moving_mean and moving_variance need to be updated. By default the update ops are placed in tf.GraphKeys.UPDATE_OPS, so they need to be executed alongside the train_op. Also, be sure to add any batch_normalization ops before getting the update_ops collection. Otherwise, update_ops will be empty, and training/inference will not work properly

也就是说，`moving_mean`, `moving_variance` 的更新需要我们特别注意，官方给的例子是这样的

```python
x_norm = tf.layers.batch_normalization(x, training=training)

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
train_op = optimizer.minimize(loss)
train_op = tf.group([train_op, update_ops])
```

比较重要的参数：

- `epsilon` is 1e-3, in __tf 1.13!!__ (tf 1.12 中默认为 1e-5), __在 MVSNet 源码中，default epsilon is 1e-5__
- `center`:  If True, add offset of `beta` to normalized tensor. If False, `beta` is ignored.
- `scale`: If True, multiply by `gamma`. If False, `gamma` is not used. When the next layer is linear (also e.g. nn.relu), this can be disabled since the scaling can be done by the next layer.
- `training`: 默认为 `False`, Whether to return the output in training mode (normalized with statistics of the current batch) or in inference mode (normalized with moving statistics). 也就是 `moving_mean`, `moving_variance` 是否用当前的 batch 的信息，在 inference 中，往往要把 `training` 设为 False, 利用已经训练好的 `moving_mean`, `moving_variance`. 在 train 中，如果我们要利用 pretrained weight, 则也应该把这个参数设为 False, 如果我们要自己重头训练，则要设置为 True, 且如之前所说的把 `update_ops` 加入 `train_op` 中
- `trainable`: 默认为 `True`

总共有两个参数 `training`, `trainable` 来控制, 推测的行为如下如果 `trainable` 为 True, gamma, beta, moving_mean, moving_variance 应该都会训练，

- `training` == `trainable` == True, 用于常规训练，但需要注意对 `update_op` 的操作，才能保证 moving_mean, moving_variance 的正常更新
- `training` == `trainable` == False, 用于正常的 inference
- `training` == True, `trainable` == False, 这应该算是一个暗坑，在 inference 过程中将不会利用训练好的 moving_mean, moving_variance, 而是直接计算当前 feed forward 的 batch 的相关值
- `training` == False, `trainable` == True, 一般用于 load pretrained model, 利用之前模型计算好的 moving_mean, moving_variance, 但要特别注意之前模型的 `epsilon` 是否等于默认值，这个值是不会在 pretrained weight 中保存的

在 tensorpack 中则有另一套类似，但更加清晰的机制，详见 [tensorpack docs](https://tensorpack.readthedocs.io/modules/models.html#tensorpack.models.BatchRenorm), 需要注意的是，如果 tensorpack 不显式指定 training 参数，则 `training` = `ctx.is_training`, [学长的报告](https://note.youdao.com/group/#/93778363/(full:md/434812763)

至于在 MVSNet 中，应该是默认 training == trainable

### dataset / augmentation

- `gen_dtu_resized_path` 生成 dtu 数据集的路径，便于之后读取
- `channels_last`
- camera files format:

```
extrinsic
E00 E01 E02 E03
E10 E11 E12 E13
E20 E21 E22 E23
E30 E31 E32 E33

intrinsic
K00 K01 K02
K10 K11 K12
K20 K21 K22

DEPTH_MIN DEPTH_INTERVAL (DEPTH_NUM DEPTH_MAX) 
```
- [ ] depth 的 min, max 是否全局相等？
- [ ] 如何解决 depth, camera params 的 scale 问题？

#### data preprocess

- center_image，细粒度，每一张图片都按照自己的 mean, var 做 normalized

```python
def center_image(img):
    """ normalize image input """
    img = img.astype(np.float32)
    var = np.var(img, axis=(0,1), keepdims=True)
    mean = np.mean(img, axis=(0,1), keepdims=True)
    return (img - mean) / (np.sqrt(var) + 0.00000001)
```


### model architecture

- 网络 base class 在 cnn_wrapper/network.py 下定义，cnn_wrapper/mvsnet.py 定义了网络前端 feature 网路以及中间的 cost volume 网络，以及最后的 depth map refinement, 网络的结合在 mvsnet/model.py 中的 `inference` 函数中定义, 其中也包含了 homography warping 等操作。

#### front end: deep feature extraction

- share weights between N branches
- output N 32-channel feature maps downsized by 4

#### core part: cost volume construction

- build a _3D cost volume_ from the extracted feature maps and input cameras.
- feature volume size: W/4 * H/4 * D * F (F is the number of channels, ie. 32)
- 3D version U-Net will output a _Probability Volume_, for cost volume regularazition
  - last conv layer outputs a 1-channel volume (Probability Volume)
- retrive depth map estimation from probability volume via _soft argmin_ (expectation, approxiamates argmax to some degree), output depth map is W/4 * H/4 
- depth map refinement

## our network architecture

Why are you interested in this programme?
Is there any particular project (or research topic) done at HKU CS that is of great interest to you?
What would you like to achieve from this experience?

### Why are you interested in this programme?


