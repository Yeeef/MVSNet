from tensorpack import *
from nn_utils import *
from loss_utils import *
from tensorpack.tfutils.summary import (add_moving_summary, add_param_summary, add_tensor_summary)
from summary_utils import add_image_summary


def get_depth_meta(cams):
    """

    :param cams: shape: batch, view_num
    :return: depth_start, depth_interval
    """
    ref_cam = cams[:, 0]
    batch_size = tf.shape(cams)[0]
    depth_start = tf.map_fn(lambda cam: cam.depth_min, ref_cam)
    assert depth_start.get_shape().as_list() == [batch_size]
    depth_interval = tf.map_fn(lambda cam: cam.depth_interval, ref_cam)
    assert depth_interval.get_shape().as_list() == [batch_size]

    return depth_start, depth_interval


class MVSNet(ModelDesc):

    height = 513
    width = 769
    view_num = 3
    data_format = 'NCHW'
    lambda_ = 1.

    weight_decay = 1.

    """
    To apply on normalization parameters, use '.*/W|.*/gamma|.*/beta'
    """
    regularization_pattern = '.*/W |.*/b$'

    debug_param_summary = False

    def __init__(self, interval_scale, depth_num):
        super(MVSNet, self).__init__()
        self.interval_scale = interval_scale
        self.depth_num = depth_num

    def inputs(self):
        return [
            tf.placeholder(tf.float32, [None, self.view_num, self.height, self.width, 3], 'imgs'),
            tf.placeholder(tf.float32, [None, self.view_num], 'cams'),
            # tf.placeholder(tf.float32, [None, self.height, self.width, 1], 'seg_map'),
            tf.placeholder(tf.float32, [None, self.height, self.width, 1], 'gt_depth'),
        ]

    def _preprocess(self, imgs, gt_depth):
        # transpose image
        # center image
        return tf.transpose(imgs, [0, 1, 4, 2, 3]), tf.transpose(gt_depth, [0, 3, 1, 2])

    def build_graph(self, imgs, cams, gt_depth):
        # preprocess
        imgs, gt_depth = self._preprocess(imgs, gt_depth)

        # define a general arg scope first, like data_format
        with argscope([tf.layers.conv3d, tf.layers.conv3d_transpose, Conv2D, MaxPooling, AvgPooling, BatchNorm],
                      data_format=self.data_format, padding='same'):
            # feature extraction
            # shape: b, view_num, c, h/4, w/4
            feature_maps = feature_extraction_net(imgs)

            # get depth_start and depth_interval batch-wise
            depth_start, depth_interval = get_depth_meta(cams)
            depth_end = depth_start + (tf.cast(self.depth_num, tf.float32) - 1) * depth_interval

            # warping layer
            # shape of cost_volume: b, c, depth_num, h/4, w/4
            cost_volume = warping_layer(feature_maps, cams, depth_start, depth_interval, self.depth_num)

            # cost volume regularization
            # shape of probability_volume: b, 1, d, h/4, w/4
            prob_volume = cost_volume_regularization(cost_volume)

            # shape of coarse_depth: b, 1, h/4, w/4
            coarse_depth = soft_argmin(prob_volume, depth_start, depth_end, self.depth_num)

            # depth_refinement
            ref_img = imgs[:, 0]
            # shape of refine_depth: b, 1, h/4, w/4
            refine_depth = depth_refinement(coarse_depth, ref_img, depth_start, depth_end)

            loss_coarse, *_ = mvsnet_regression_loss(gt_depth, coarse_depth, depth_interval)
            loss_refine, less_one_acc, less_three_acc = mvsnet_regression_loss(gt_depth, refine_depth, depth_interval)

            loss = tf.add(loss_refine, loss_coarse * self.lambda_, name='loss')
            less_one_acc = tf.identity(less_one_acc, name='less_one_acc')
            less_three_acc = tf.identity(less_three_acc, name='less_three_acc')

            with tf.device('/cpu:0'):
                add_moving_summary(loss, less_one_acc, less_three_acc)
            
            if self.debug_param_summary:
                with tf.device('/cpu:0'):
                    add_param_summary(
                        ['.*/W', ['histogram', 'rms']],
                        ['.*/gamma', ['histogram', 'mean']],
                        ['.*/beta', ['histogram', 'mean']]
                    )
                    all_vars = [var for var in tf.trainable_variables() if "gamma" in var.name or 'beta' in var.name]
                    grad_vars = tf.gradients(loss, all_vars)
                    for var, grad in zip(all_vars, grad_vars):
                        add_tensor_summary(grad, ['histogram', 'rms'], name=var.name + '-grad')




        return loss

    def optimizer(self):
        pass
