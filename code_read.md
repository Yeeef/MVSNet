# code reading

## notations

- `tf.nn.layers.conv2d` reuse parameter: Boolean, whether to reuse the weights of a previous layer by the same name., [doc](https://www.tensorflow.org/api_docs/python/tf/layers/conv2d)

### batch normalization should be paid significant attention

`tf.layers.batch_normalization` [tf 1.13 doc](https://www.tensorflow.org/api_docs/python/tf/layers/batch_normalization), 其中提到一点重要信息，_when you use low-level API, be careful to manipulate update ops_

    when training, the moving_mean and moving_variance need to be updated. By default the update ops are placed in tf.GraphKeys.UPDATE_OPS, so they need to be executed alongside the train_op. Also, be sure to add any batch_normalization ops before getting the update_ops collection. Otherwise, update_ops will be empty, and training/inference will not work properly

也就是说，`moving_mean`, `moving_variance` 的更新需要我们特别注意，官方给的例子是这样的

```python
 x_norm = tf.layers.batch_normalization(x, training=training)

  # ...

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    train_op = optimizer.minimize(loss)
```

比较重要的参数：

- `epsilon` is 1e-3, in __tf 1.13!!__ (tf 1.12 中默认为 1e-3), __在 MVSNet 源码中，default epsilon is 1e-5__
- `center`:  If True, add offset of `beta` to normalized tensor. If False, `beta` is ignored.
- `scale`: If True, multiply by `gamma`. If False, `gamma` is not used. When the next layer is linear (also e.g. nn.relu), this can be disabled since the scaling can be done by the next layer.
- `training`: 默认为 `False`, Whether to return the output in training mode (normalized with statistics of the current batch) or in inference mode (normalized with moving statistics). 也就是 `moving_mean`, `moving_variance` 是否用当前的 batch 的信息，在 inference 中，往往要把 `training` 设为 False, 利用已经训练好的 `moving_mean`, `moving_variance`. 在 train 中，如果我们要利用 pretrained weight, 则也应该把这个参数设为 False, 如果我们要自己重头训练，则要设置为 True, 且如之前所说的把 `update_ops` 加入 `train_op` 中
- `trainable`: 默认为 `True`

总共有两个参数 `training`, `trainable` 来控制, 推测的行为如下如果 `trainable` 为 True, gamma, beta, moving_mean, moving_variance 应该都会训练，

- `training` == `trainable` == True, 用于常规训练，但需要注意对 `update_op` 的操作，才能保证 moving_mean, moving_variance 的正常更新
- `training` == `trainable` == False, 用于正常的 inference
- `training` == True, `trainable` == False, 这应该算是一个暗坑，在 inference 过程中将不会利用训练好的 moving_mean, moving_variance, 而是直接计算当前 feed forward 的 batch 的相关值
- `training` == False, `trainable` == True, 一般用于 load pretrained model, 利用之前模型计算好的 moving_mean, moving_variance, 但要特别注意之前模型的 `epsilon` 是否等于默认值，这个值是不会在 pretrained weight 中保存的

特别注意，tensorflow 并没有自动设置 `training` 的机制，所以一定要在 train, val 的时候正确设置 `training`

在 tensorpack 中则有另一套类似，但更加清晰的机制，详见 [tensorpack docs](https://tensorpack.readthedocs.io/modules/models.html#tensorpack.models.BatchRenorm), 需要注意的是，如果 tensorpack 不显式指定 training 参数，则 `training` = `ctx.is_training`, [学长的报告](https://note.youdao.com/group/#/93778363/(full:md/434812763)

至于在 MVSNet 中，应该是默认 training == self.is_training, 通过 Network 类封装了 is_training 参数，从而实现上述需求

## dataset / augmentation

### dtu training set overview

- val set: 3, 5, 17, 21, 28, 35, 37, 38, 40, 43, 56, 59, 66, 67, 82, 86, 106, 117
- eval set: 1, 4, 9, 10, 11, 12, 13, 15, 23, 24, 29, 32, 33, 34, 48, 49,62, 75, 77, 110, 114, 118
- train set: other 79 scans

every scan contains 49 images and associated cams
Considering each scan contains 49 images with 7 different lighting conditions,by setting each image as the reference,DTUdataset provides _27097_ trainingsamples in total.

79 * 7 = 563 组图片，每组49张，每一张都可以作为 ref view, 所以训练集共有 563 * 49 = 27097, 其实还可以扩展的更多，但是他有 view selection 的过程，也就是规定了一个顺序呢

### mydataflow writing

现在这个图片数量很大，不便于一次读入内存，所以 dataflow 要写的更加注意效率一点，我觉得一点点的 yield 必然是很慢的

### notations


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

### data preprocess

- center_image，细粒度，每一张图片都按照自己的 mean, var 做 normalized

```python
def center_image(img):
    """ normalize image input """
    img = img.astype(np.float32)
    var = np.var(img, axis=(0,1), keepdims=True)
    mean = np.mean(img, axis=(0,1), keepdims=True)
    return (img - mean) / (np.sqrt(var) + 0.00000001)
```

- load_cam, cam 是一个 2\*4\*4 维度的矩阵，cam[1] 是 extrinsic 矩阵，cam[2] 是intrinsic 矩阵，需要注意的是，extrinsic 矩阵确实有 4×4 但是 intrinsic 实际上只有 3*3 , cam 有一些位置是浪费的，这样设置只是为了自己处理方便一些
- cam[1, 3] 的四个元素比较有意思，需要格外注意一下
  - 0th: _DEPTH_MIN_
  - 1th: _DEPTH_INTERVAL_
  - 2th: _DEPTH_NUM_
  - 3th: _DEPTH_MAX_
  
- github 上的注意事项： Note that the depth range and depth resolution are determined by the minimum depth DEPTH_MIN, the interval between two depth samples DEPTH_INTERVAL, and also the depth sample number DEPTH_NUM (or max_d in the training/testing scripts if DEPTH_NUM is not provided). We also left the interval_scale for controlling the depth resolution. The maximum depth is then computed as:

DEPTH_MAX = DEPTH_MIN + (interval_scale * DEPTH_INTERVAL) * (max_d - 1)

- 也就是说，如果不提供 DEPTH_MAX 与 DEPTH_NUM，DEPTH_MAX 将由上述公式进行计算
- 如果提供了 DEPTH_NUM, 则将公式中的 max_d 替换为 DEPTH_NUM
- 如果提供了 DEPTH_MAX 则不利用上述公式


```python
def load_cam(file, interval_scale=1):
    """ read camera txt file """
    cam = np.zeros((2, 4, 4))
    words = file.read().split()
    # read extrinsic
    for i in range(0, 4):
        for j in range(0, 4):
            extrinsic_index = 4 * i + j + 1
            cam[0][i][j] = words[extrinsic_index]

    # read intrinsic
    for i in range(0, 3):
        for j in range(0, 3):
            intrinsic_index = 3 * i + j + 18
            cam[1][i][j] = words[intrinsic_index]
            
    if len(words) == 29:
        cam[1][3][0] = words[27]
        cam[1][3][1] = float(words[28]) * interval_scale
        cam[1][3][2] = FLAGS.max_d
        cam[1][3][3] = cam[1][3][0] + cam[1][3][1] * cam[1][3][2]
    elif len(words) == 30:
        cam[1][3][0] = words[27]
        cam[1][3][1] = float(words[28]) * interval_scale
        cam[1][3][2] = words[29]
        cam[1][3][3] = cam[1][3][0] + cam[1][3][1] * cam[1][3][2]
    elif len(words) == 31:
        cam[1][3][0] = words[27]
        cam[1][3][1] = float(words[28]) * interval_scale
        cam[1][3][2] = words[29]
        cam[1][3][3] = words[30]
    else:
        cam[1][3][0] = 0
        cam[1][3][1] = 0
        cam[1][3][2] = 0
        cam[1][3][3] = 0

    return cam

```

- load_pfm __depth_image__ 的读取

### 如何构造符合他输入的训练集

- [x] cam.txt 比较好构造
- [ ] pair.txt
  - 他的数据集局限性其实也是有的，别看有这么多图片，实际的 cam 位置只有 40 多种 

#### pair.txt

- first line: 总共有多少个训练集？


## model architecture

- 网络 base class 在 cnn_wrapper/network.py 下定义，cnn_wrapper/mvsnet.py 定义了网络前端 feature 网路以及中间的 cost volume 网络，以及最后的 depth map refinement, 网络的结合在 mvsnet/model.py 中的 `inference` 函数中定义, 其中也包含了 homography warping 等操作。

### front end: deep feature extraction

- share weights between N branches
- output N 32-channel feature maps downsized by 4

#### details

- conv 的 kernel_initializer? bias_initializer? need bias? layer_regularizer?
- kernel_initializer is the default initializer of `tf.layers.conv2d`, default is `glorot_uniform_initializer`
- bias_initialzer is the default initializer: `tf.zeros_initializer()`
- kernel regularizer: tf.contrib.layers.l2_regularizer(1.0)
- if biased, bias regularizer: tf.contrib.layers.l2_regularizer(1.0)
- bn_epsilon is 1e-5
- 所有卷积层，包括最后一个没有 bn 的，都没有 Biased
- Bn 的 epsilon 为 1e-5
  - momentum 默认为 **0.99**, tensorpack 默认为 **0.9**
  - is_training = True
  - 我们需要重头训练，所以不需要把 is_training 设置为 False
  - gamma, beta 并不设有 regularizer
  - fused 参数为 True

### core part: cost volume construction

- build a _3D cost volume_ from the extracted feature maps and input cameras.
- feature volume size: W/4 * H/4 * D * F (F is the number of channels, ie. 32)
- 3D version U-Net will output a _Probability Volume_, for cost volume regularazition
  - last conv layer outputs a 1-channel volume (Probability Volume)
- retrive depth map estimation from probability volume via _soft argmin_ (expectation, approxiamates argmax to some degree), output depth map is W/4 * H/4 
- depth map refinement
- **发现 bug 已经向作者提交 issue**

### cost volume regulariztion

- 均为 conv3D, biased=False
- kernel_initializer=`glorot_uniform_initializer`

### refinement

- 注意最后一层 conv 允许 bias

## our network architecture

Why are you interested in this programme?
Is there any particular project (or research topic) done at HKU CS that is of great interest to you?
What would you like to achieve from this experience?

## input data

### dtu traing set overview

### my dataflow

## problems

- 各个 gpu 之间共享参数需要显式指定 reuse=True 吗？
  - 不需要利用 tensorpack 的 `_ParamServerTrainer` 即可
- channel_first 带来的一些副作用，要好好看一遍，尤其是 conv3d 出来的大小
  - 全部改成 channels_last
- [x] 多 gpu 下的 summary 怎么搞: see add_image_summary()
- 这是害怕 exp 之后溢出？
```python
probability_volume = tf.nn.softmax(
            tf.scalar_mul(-1, filtered_cost_volume), axis=1, name='prob_volume')
```
- refine_depth 过程中需要 resize image, 是否需要要把图片 h+1, w+1?
- 仔细检查是否有其他地方使用了 resize
  - 应该只有一个地方，且已改为 `TFUpsampleBilinear`
- batch norm 是否有 regularizers
  - 没有
- 他的 regularize 的 \lambda 为1，不知道是不是太大了，本着尽量复现的原则，暂时还是在各层中设置 regularizer，最后再写成 wd_cost 的形式
- 源代码没有考虑 rgb, bgr 的问题，cv2.imread 默认是 bgr
- dataflow 中计算 depth_start, depth_end 都少了一个 depth_interval
- depth 并没有 scale 到 [0, 1] 之间
- update ops 和 local varible 的区别？
- 为什么 deeplab v3+ 的图看上去很干净？ inference tower 出现了奇怪的黄色边，要看看怎么回事，很可能是 bn 的 moving_mean 和 moving_var 在 inference 的时候反过来更新 training 了
  - 强行设置为 False 看看表现
  - 需要在 bn warp 那里利用 ctx.is_training 进行修改
- 为什么 conv3d 是 argscope 显示的 log? 看一下官方例子是什么样子的
  - 官方例子中，也是这样
- MVSNet 貌似压根就没做 weight decay orz

## tensorpack layer wrapping

会改变几个名字，`kernel->W`, `bias->b`, `mean/EMA`, `variance/EMA`

```python
def rename_tflayer_get_variable():
    """
    Rename all :func:`tf.get_variable` with rules that transforms tflayer style to tensorpack style.

    Returns:
        A context where the variables are renamed.

    Example:

    .. code-block:: python

        with rename_tflayer_get_variable():
            x = tf.layer.conv2d(input, 3, 3, name='conv0')
            # variables will be named 'conv0/W', 'conv0/b'
    """
    mapping = {
        'kernel': 'W',
        'bias': 'b',
        'moving_mean': 'mean/EMA',
        'moving_variance': 'variance/EMA',
    }
    return rename_get_variable(mapping)
```

发现 tensorpack 的 batchNorm 不支持 5d-tensor, 首先去看一下源码，再者一定要小心这些不同名字之间的歧义，[related issue](https://github.com/tensorpack/tensorpack/issues/291) batch_norm 确实没有和 `tf.layers.batch_normalization` 保持一致

tensorpack 不会再增加更多而 warpper 层，因为他基本已经将所有层都底层实现都转成了 `tf.layers.*`, 那么问题就来了，`argscope` 这些语法糖怎么用？ [这么用](https://github.com/tensorpack/tensorpack/pull/1017), 但是需要注意的是 `enable_argscope_for_module(tf.layers)` 并不能帮我们把 `kernel` 改为 `W`, 把 `bias` 改为 `b`, `moving_mean'`: `mean/EMA`, `moving_variance`: `variance/EMA`, 这个还需要我们自己去 wrap

并且，tf.layers.batch_normalization 并没有 `data_format` 而只有 `axis`, 也需要我们去 wrap

并且，moving_mean, moving_variance 还需要比较特殊的更新？我晕了，但是 mvsnet 的源码中并没有这么做，也许不用？这个需要打印出来测试一下
tf.layers._ 的 API 会自动把该加的 update_op 加入 `tf.GraphKeys.UPDATE_OPS`，`with control_dependencies` 其实就是保证其中的 op 需要先被执行

- [一篇对于 batch norm 的讲解](https://towardsdatascience.com/batch-normalization-theory-and-how-to-use-it-with-tensorflow-1892ca0173ad) 其中提到，如果不 `control_dependencies` 会出问题

本着复现的原则，先不进行上述操作，在运行中自己也试试能不能

如果直接用 tf.layers.bn 能保证正确的 reuse 吗？ tf.layers._ 的 reuse 含义是什么？

多个 gpu 的 Graph 到底是什么关系？ 

### 现在暂时的解决方案

- 通过 `rename_tflayer_get_variable` 来进行 W,b,moving_mean, moving_variance 的命名
- 通过 `enable_argscope_for_module(tf.layers)` 来实现 argscope 的使用，同时将 `axis` 设置为1
- update ops 的问题先放一放
- reuse 的问题先放一放

### tensorpack BatchNorm 封装

tensorpack 中对 batch_normalization 进行了一些比较好的封装，我们可以借鉴

- `tf.GraphKeys.MODEL_VARIABLES` 的含义 [tf graph keys](https://www.tensorflow.org/api_docs/python/tf/GraphKeys)