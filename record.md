# record

__4.24__

- 终于跑通了 MVSNet, 其中遇到了很多很多问题，闲下来会写一个集锦
- 在小数据集中测试，发现速度很慢, 感觉 warping layer 的效率非常低下，再测试 Batch_size 的极限，搞清瓶颈在哪
- refine_depth is weired

__4.25__

测试源代码 train 过程，并且在自己代码上跑完全相同的测试尝试复现，但我忘记在自己代码中注释 refine 过程，所以现在不能完全确定是哪里出了问题，又发现他用的 feature extraction 有所改进
输出的 depth 搞成五颜六色
接下来准备再看看，且为源代码添加 image summary
还有一点一直不确定，就是他用的 Kernel_initializer 下次可以在 tensorboard 的 histogram 里看一下

### todo

- [x] 添加 image summary
- [x] 测试原始代码多gpu是否正常, 源代码可以多 gpu 但不确定这些机制是不是对
  - [x] 在 graph 上看，应该是 conv 被两个 tower 共享了, 而且从他的代码来看，压根就没有 variable scope 的概念，是通过 `tf.layers._` api 中的 reuse 来进行 reuse 的，不同 gpu 通过 name_scope 来进行区分
  - [x] 如果我把 reuse 强行变成 false 呢？确实无法通过，会提示 name 已经用过了，不能再构建一次
- [x] 如果原始代码多 gpu 正常，那么肯定是 tp 的某个机制出了问题
  - [x] 能否把 stageinput 从代码里去掉？ 这样一种比较 dirty 的方法不是长久之计，有时间提个 issue 去
- [x] 确定 kernel initializer
  - [x] 确实是 glorot_uniform_distribution
- [x] replicate 一下他的 unet
  - [x] 对比 wyx, zyk, mvsnet 的 groupnorm 实现方法，_有所不同_, channel_wise? reshape order
    - [ ] gn 为什么不需要把 EMA 加入 UPDATE_KEYS?
- [ ] 源代码的 EMA 有没有正常更新？
- [ ] 检查一遍代码 todos
- [ ] 我的那种 variable sharing policy 对吗？
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

## gpu 0,1,2,3 batch_size 1

- is graph alright?
  - [] how to make a cool graph like Deepv3+？
- use fake dataflow(size = 20)
- `StagingInput` seems to take forever long(with fake dataflow, so it can not be the problem of dataflow)
  - not make warping_layer a layer, not helpful
  - use other multi gpu trainer, mayby not helpful, I didn't wait long
  - remove the warping layer, helpful
- [] 总之是 warping layer 除了问题，一点一点解开结构去看哪个地方阻止了 multiGPU
- `Size of these collections were changed in tower1: (tf.GraphKeys.UPDATE_OPS: 70->92)`？
  - 应该就是多 gpu 的时候，每个都加入了 EMA 的 UPDATE_KEYS
  - 可能有问题，我把 refinenet 去了之后每个 tower 都只有 22 个，为什么之前不是这样？这22个应该是 3d volume regularization 带来的，也就说之前 uninet 的 BN 的 updateop 加的可能有问题？

## try original code

非常慢，6个 epoch 要跑24个小时

### can multiple gpu?

### how to get prob map?

## reproduce?

### 1

我认为的唯一不同在于