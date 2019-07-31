# R-MVSNet 论文阅读 & 代码修改

## overview and intuition

### 3d cnn -> gru

最大的改动就在于把 3D CNN 的 cost volume regularization 换成了 gru 的 sequential processing. 出发点在于 3d cnn 计算量消耗太大，论文也 coarsely 地把 3d cnn 和 gru formulize 到了一个框架下解释, cost volume 本质上可以看作 several(view number) feature maps 在 depth 方向 stack 的结果，3d cnn 最终输出的 cost volume 的每一个 voxel 的感知野是整个 cost volume（在 cascade conv 过程中感知野越来越大），而 gru 输出的 cost volume 的每一个 voxel 的感知野是之前所有 depth 所对应的feature maps. 

这里其实我觉得有一点怪，因为越小的 depth 利用的信息越少，gru 的通常应用场景下，每一个 time step 确实是有明确的先后意义的（类马氏链过程），但是 depth 作为 time step 的时候，我直觉上觉得没有太强的先后意义。如果用一个 双向 gru 我觉得会更好一点，让小的 depth 也能接触到后面的信息。

### inverse depth / cross entropy loss

第二大的改动在于 loss 的改动，在 MVSNet 中给定 d_min, d_max, 用来 warp 的 depth 是均匀分布在 d_min, d_max 中的，这样的好处在于可以通过 soft argmin 求期望的办法来获得 sub-pixel 的 depth 精度，连续性也更好，坏处在于 d_min, d_max 不能设置太大，并且一般的 mvs 算法都是在 d_min 采样多, d_max 采样少.

所以在 r-mvsnet 中他们采用了 inverse depth 的采样方法，在 d_min 采样多, d_max 采样少，但这样 soft argmin 的方式就不那么稳妥了（个人理解：对真正的期望积分形式不再是一个好的估计）所以把问题从之前的 regression problem -> classification problem. loss 也就相应的变为 cross-entropy loss. 这里论文貌似把 P, Q 写反了。

看作 classification 的另一个问题就在于 sub-pixel 的精度不存在了，并且 inverse depth 的采样方式也导致最终的 depth map 不是那么光滑，需要进行一个后处理，后处理并没有 end-to-end

## 代码阅读

- [ ] 为什么第一个传入的是 -cost?
- [ ] groupnorm 函数中使用 layer_norm 和 instance_norm 的时候并没有 transpose 回来
- [ ] 源码的 cross entropy loss 写反了
- [ ] 为何要有 infer_men 与 infer 的区别 / infer_rnn 与 infer_winner 的区别？
- [ ] 实际上在训练过程中并没有 inverse depth