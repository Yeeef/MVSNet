# MVSNet record

__4.24__

- 终于跑通了 MVSNet, 其中遇到了很多很多问题，闲下来会写一个集锦
- 在小数据集中测试，发现速度很慢, 感觉 warping layer 的效率非常低下，再测试 Batch_size 的极限，搞清瓶颈在哪
- refine_depth is weired

__4.25__

测试源代码 train 过程，并且在自己代码上跑完全相同的测试尝试复现，但我忘记在自己代码中注释 refine 过程，所以现在不能完全确定是哪里出了问题，又发现他用的 feature extraction 有所改进
输出的 depth 搞成五颜六色
接下来准备再看看，且为源代码添加 image summary
还有一点一直不确定，就是他用的 Kernel_initializer 下次可以在 tensorboard 的 histogram 里看一下

__5.5__

与原先的结果还是有细微的差异，所以必须要好好复现一下，首先检查代码，再去逐层的输入输出检查
有没有可能是输入出了问题? 有没有可能是 feature_extraction_net 那里的参数并没有共享? channels_first?
为了彻底检查，不如先把单gpu的行为搞清楚
测试一组我们的图片
应该是数据出问题了, 去除 prefetch 试试, 数据确实因为 prefetch 出了问题，把 prefetch 去除后还是不能完全复现，数据的问题应该不大
检查一下 gn
现在思路是这样，总之整个模块分离的很清晰了，我先把两边的 feature extraction 和 3d regularization 都搞得很浅
再一个就是赶紧测试一组, 他的 crop image 绝对有 bug, 如果 max_d, max_h 比较小的话就会出问题
他的图片到底有多大呢？

__5.7__

终于找到了在我们的数据上合适的参数，接下来写好一个脚本，基础功能就是根据咱们的数据集转换为他的输入，现在的 depth_min, depth_interval 必须人工调整，争取写一个自适应算法
接下来搞清楚早上讨论的问题，之后继续着手复现

__5.9__

细粒度复现开始，思路是这样，总之整个模块分离的很清晰了，我先把两边的 feature extraction 和 3d regularization 都搞得浅一点

meshlab 出现太小的原因大概是飞的该厉害了？对

他的 refine 部分到底是怎么做的，那点一直没有特别懂

__5.11__

Bugs may locate at `cost volume regularization part`

Bug located and found, but now I have to construct my model in `channels_last` fashion.

The bug is in `build_cost_volume`

总结这次 debug 过程，一是数据，二是细节，网络结构不会轻易搭错的

__5.13__

跑完一版之后，去 evaluate val_data, 达到真正意义的复现

__7.7__

今天的第一个任务在于搞定公网 ip 的问题，争取跑一次 R-MVSNet

### todo

- [x] 添加 image summary
- [x] 测试原始代码多gpu是否正常, 源代码可以多 gpu 但不确定这些机制是不是对
  - [x] 在 graph 上看，应该是 conv 被两个 tower 共享了, 而且从他的代码来看，压根就没有 variable scope 的概念，是通过 `tf.layers._` api 中的 reuse 来进行 reuse 的，不同 gpu 通过 name_scope 来进行区分
  - [x] 如果我把 reuse 强行变成 false 呢？确实无法通过，会提示 name 已经用过了，不能再构建一次
- [x] 如果原始代码多 gpu 正常，那么肯定是 tp 的某个机制出了问题
  - [x] 能否把 stageinput 从代码里去掉？ 这样一种比较 dirty 的方法不是长久之计，有时间提个 issue 去
- [x] 确定 kernel initializerfsdf
  - [x] 确实是 glorot_uniform_distribution
- [x] replicate 一下他的 unet
  - [x] 对比 wyx, zyk, mvsnet 的 groupnorm 实现方法，_有所不同_, channel_wise? reshape order
    - [ ] gn 为什么不需要把 EMA 加入 UPDATE_KEYS?
- [x] 源代码的 EMA 有没有正常更新？
    - tf.nn.layer.* 的 api 是一个高级api, 会自动加 update_keys, 应该不需要我担心，就是 tensorpack 中要记住加入 `RunUpdateOps`
- [ ] 检查一遍代码 todos
- [x] 我的那种 variable sharing policy 对吗？
    - 现在我更加细致的判断了 ctx.is_main_tower, 并且在 param-log 能看到 weight 确实只有一份
- [ ] 检查代码的未使用变量

## speed enhancement

### gpu 1 batch_size 1

- dataflow alone is not slow (200it/s)
  - 如果把 center image 操作加入 dataflow 会减慢 30x speed
  - 利用 tf.nn.moments 计算 mean, var，gpu util is still high (around 90)
- what about fake dataflow, still slow, the graph is the bottleneck
- what about gpu util? Set step_per_epoch to 20 or use fake dataflow to see the result as soon as possible (gpu utils is around 90%)
- So finally, the only part to optimize is the model itself. And without any doubt, the bottleneck is _warping layer_
- [] Next, I will try to let `allow_soft_place=False` and see the result
- what if remove `StagingInput?`
  - 没有太大影响
- [] try `GraphProfiler`
- not regard warping as a layer anymore
  - is graph right? yes
  - speed is same

已经达到了不改变 `warping_layer` 的极限

## scale to multiple gpus

### gpu 0,1,2,3 batch_size 1

- is graph alright?
  - [ ] how to make a cool graph like Deepv3+？
- use fake dataflow(size = 20)
- `StagingInput` seems to take forever long(with fake dataflow, so it can not be the problem of dataflow)
  - not make warping_layer a layer, not helpful
  - use other multi gpu trainer, mayby not helpful, I didn't wait long
  - remove the warping layer, helpful
- [x] 总之是 warping layer 除了问题，一点一点解开结构去看哪个地方阻止了 multiGPU
  - 在这里我修改了 tensorpack 的源码，强制不进行 staging input, 最后可以成功在多 gpu 上跑了
- `Size of these collections were changed in tower1: (tf.GraphKeys.UPDATE_OPS: 70->92)`？
  - 应该就是多 gpu 的时候，每个都加入了 EMA 的 UPDATE_KEYS
  - [ ] 可能有问题，我把 refinenet 去了之后每个 tower 都只有 22 个，为什么之前不是这样？这22个应该是 3d volume regularization 带来的，也就说之前 uninet 的 BN 的 updateop 加的可能有问题？

## try original code

非常慢，6个 epoch 要跑24个小时

### can multiple gpu?

源代码可以多 gpu, 之后我去修改了 tensorpack 的源码，也成功支持了多 gpu

### how to get prob map?

直接用他们的函数就行了, I don't need to bother myself about this

## reproduce

### data 

data 之前确实有问题，主要在于 prefetch 带来的，现在数据应该问题不大

### feature extraction?

I will simply my feature extraction network as far as possible, like, 2 layers maybe just for necessary stride

Maybe I could just try to use the `uninet` first, I don't believe I wrote wrong way.

#### related to batch normalization

Every `tf.layers.batch_normalization` will add 2 update ops(if `training=True`) and 2 trainable variable(if `trainable=True`) in the graph

### 3d cost volume regularization?

- bug found, last layer should not be activated by BNReLU.

找到 bug 后已经成功复现