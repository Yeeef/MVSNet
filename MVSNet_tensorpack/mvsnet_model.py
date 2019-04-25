from tensorpack import *
from tensorpack.utils import logger
from nn_utils import *
from loss_utils import *
from tensorpack.tfutils.summary import (add_moving_summary, add_param_summary, add_tensor_summary)
from summary_utils import add_image_summary
import tensorflow as tf
from DataManager import Cam

""" monkey-patch """
enable_argscope_for_module(tf.layers)


def get_depth_meta(cams, depth_num):
    """

    :param cams: shape: batch, view_num
    :return: depth_start, depth_interval
    """
    with tf.variable_scope('depth_meta'):
        ref_cam = cams[:, 0]
        logger.warn('cams shape: {}'.format(cams.get_shape().as_list()))
        logger.warn('ref_cam shape: {}'.format(ref_cam.get_shape().as_list()))
        logger.warn('ref_cam type: {}'.format(type(ref_cam)))

        batch_size = tf.shape(cams)[0]
        depth_start = tf.reshape(
            tf.slice(ref_cam, [0, 1, 3, 0], [batch_size, 1, 1, 1]), [batch_size], name='depth_start')
        depth_interval = tf.reshape(
            tf.slice(ref_cam, [0, 1, 3, 1], [batch_size, 1, 1, 1]), [batch_size], name='depth_interval')
        depth_end = tf.add(depth_start, (tf.cast(depth_num, tf.float32) - 1) * depth_interval, name='depth_end')

        # depth_start = tf.map_fn(lambda cam: Cam.get_depth_meta(cam, 'depth_min'), ref_cam)
        # assert depth_start.get_shape().as_list() == [batch_size]
        # depth_interval = tf.map_fn(lambda cam: Cam.get_depth_meta(cam, 'depth_interval'), ref_cam)
        # assert depth_interval.get_shape().as_list() == [batch_size]

    return depth_start, depth_interval, depth_end


def center_image(imgs):
    """

    :param imgs: shape: b, view_num, h, w, c
    :return:
    """
    assert len(imgs.get_shape().as_list()) == 5
    moments = tf.nn.moments(tf.cast(imgs, tf.float32), axes=(2, 3), keep_dims=True)
    return (imgs - moments[0]) / (moments[1] + 1e-7)


class MVSNet(ModelDesc):

    height = 512
    width = 640
    view_num = 3
    data_format = 'channels_first'
    lambda_ = 1.

    weight_decay = 1.

    """
    To apply on normalization parameters, use '.*/W|.*/gamma|.*/beta'
    """
    regularization_pattern = '.*/W |.*/b$'

    debug_param_summary = False

    base_lr = 1e-3

    """Step interval to decay learning rate."""
    decay_steps = 10000

    """Learning rate decay rate"""
    decay_rate = 0.9

    def __init__(self, depth_num, bn_training, bn_trainable, batch_size, branch_function, is_refine):
        super(MVSNet, self).__init__()
        # self.is_training = is_training
        self.bn_training = bn_training
        self.bn_trainable = bn_trainable
        self.depth_num = depth_num
        self.batch_size = batch_size
        self.branch_function = branch_function
        self.is_refine = is_refine

    def inputs(self):
        return [
            tf.placeholder(tf.float32, [None, self.view_num, self.height, self.width, 3], 'imgs'),
            tf.placeholder(tf.float32, [None, self.view_num, 2, 4, 4], 'cams'),
            # tf.placeholder(tf.float32, [None, self.height, self.width, 1], 'seg_map'),
            tf.placeholder(tf.float32, [None, self.height // 4, self.width // 4, 1], 'gt_depth'),
        ]

    def _preprocess(self, imgs, gt_depth):
        # transpose image
        # center image done at dataflow
        with tf.variable_scope('preprocess'):
            imgs = center_image(imgs)
            imgs = tf.transpose(imgs, [0, 1, 4, 2, 3], name='transpose_imgs')
            gt_depth = tf.transpose(gt_depth, [0, 3, 1, 2], name='transpose_gt_depth')
            ref_img = imgs[:, 0]
            ref_img = tf.identity(ref_img, name='ref_img')
            return imgs, gt_depth, ref_img

    def build_graph(self, imgs, cams, gt_depth):
        # preprocess
        imgs, gt_depth, ref_img = self._preprocess(imgs, gt_depth)
        # define a general arg scope first, like data_format
        # ctx = get_current_tower_context()
        # if self.bn_trainable is None:
        #     self.bn_trainable = ctx.is_training
        # if self.bn_training is None:
        #     self.bn_training = ctx.is_training
        with argscope([tf.layers.conv3d, tf.layers.conv3d_transpose,
                       Conv2D, Conv2DTranspose, MaxPooling, AvgPooling, BatchNorm],
                      data_format=self.data_format):
            # feature extraction
            # shape: b, view_num, c, h/4, w/4
            feature_maps = feature_extraction_net(imgs, self.branch_function)

            # get depth_start and depth_interval batch-wise
            depth_start, depth_interval, depth_end = get_depth_meta(cams, depth_num=self.depth_num)

            # warping layer
            # shape of cost_volume: b, c, depth_num, h/4, w/4
            cost_volume = warping_layer('warping', feature_maps, cams, depth_start
                                        , depth_interval, self.depth_num)
            # cost_volume = tf.get_variable('fake_cost_volume', (1, 32, 192, 128, 160))

            # cost volume regularization
            # shape of probability_volume: b, 1, d, h/4, w/4
            prob_volume = cost_volume_regularization(cost_volume, self.bn_training, self.bn_trainable)

            # shape of coarse_depth: b, 1, h/4, w/4
            coarse_depth = soft_argmin('soft_argmin', prob_volume, depth_start, depth_end, self.depth_num, self.batch_size)

            # shape of refine_depth: b, 1, h/4, w/4
            if self.is_refine:
                refine_depth = depth_refinement(coarse_depth, ref_img, depth_start, depth_end)
                loss_coarse, *_ = mvsnet_regression_loss(gt_depth, coarse_depth, depth_interval, 'coarse_loss')
                loss_refine, less_one_acc, less_three_acc = mvsnet_regression_loss(gt_depth, refine_depth,
                                                                                   depth_interval, 'refine_loss')
            else:
                refine_depth = coarse_depth
                # loss_coarse, *_ = mvsnet_regression_loss(gt_depth, coarse_depth, depth_interval, 'coarse_loss')
                loss_refine, less_one_acc, less_three_acc = mvsnet_regression_loss(gt_depth, refine_depth,
                                                                                   depth_interval, 'refine_loss')
                loss_coarse = tf.identity(loss_refine, name='coarse_loss')

            loss = tf.add(loss_refine / 2, loss_coarse * self.lambda_ / 2, name='loss')
            less_one_acc = tf.identity(less_one_acc, name='less_one_acc')
            less_three_acc = tf.identity(less_three_acc, name='less_three_acc')

            with tf.variable_scope('summaries'):
                with tf.device('/cpu:0'):
                    add_moving_summary(loss, loss_coarse, loss_refine, less_one_acc, less_three_acc)
                # add_image_summary(tf.clip_by_value(tf.transpose(coarse_depth, [0, 2, 3, 1]), 0, 255)
                #                   , name='coarse_depth')
                add_image_summary(tf.transpose(coarse_depth, [0, 2, 3, 1])
                                  , name='coarse_depth')
                add_image_summary(tf.transpose(refine_depth, [0, 2, 3, 1])
                                  , name='refine_depth')
                add_image_summary(tf.transpose(ref_img, [0, 2, 3, 1]), name='rgb')
                add_image_summary(tf.transpose(gt_depth, [0, 2, 3, 1]), name='gt_depth')

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
        lr = tf.train.exponential_decay(
            self.base_lr,
            global_step=get_global_step_var(),
            decay_steps=self.decay_steps,
            decay_rate=self.decay_rate,
            name='learning-rate'
        )
        opt = tf.train.RMSPropOptimizer(learning_rate=lr)
        tf.summary.scalar('lr', lr)
        return opt
