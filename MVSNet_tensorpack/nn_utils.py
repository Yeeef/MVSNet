import tensorflow as tf
from tensorflow import Tensor
from tensorpack import *
from DataManager import Cam
from homography_utils import *

__all__ = ['feature_extraction_net', 'warping_layer', 'cost_volume_regularization']


def feature_extraction_branch(img):
    """
    Already in the scope of {[tf.layers.conv3d, tf.layers.conv3d_transpose, Conv2D, MaxPooling, AvgPooling, BatchNorm], data_format=self.data_format, padding='same'}
    :param img: shape: b * view_num * c * h * w
    :return: l
    """
    with argscope([Conv2D], use_bias=False, kernel_initializer=tf.glorot_uniform_initializer(),
                  kernel_regularizer=tf.contrib.layers.l2_regularizer(1.0), paddings='same'), \
         argscope([BatchNorm], epsilon=1e-5):
        with tf.variable_scope('feature_extraction_branch', reuse=tf.AUTO_REUSE):
            base_filter = 8
            l = Conv2D('conv0_0', img, base_filter, 3, 1, activation=BNReLU)
            l = Conv2D('conv0_1', l, base_filter, 3, 1, activation=BNReLU)

            l = Conv2D('conv1_0', l, base_filter * 2, 5, 2, activation=BNReLU)
            l = Conv2D('conv1_1', l, base_filter * 2, 3, 1, activation=BNReLU)
            l = Conv2D('conv1_2', l, base_filter * 2, 3, 1, activation=BNReLU)

            l = Conv2D('conv2_0', l, base_filter * 4, 5, 2, activation=BNReLU)
            l = Conv2D('conv2_1', l, base_filter * 4, 3, 1, activation=BNReLU)
            feature_map = Conv2D('conv2_1', l, base_filter * 4, 3, 1, activation=None)

    return feature_map


def feature_extraction_net(imgs):
    """
    feature extraction net
    Take care of variable_scope's reuse param!
    :param imgs: shape: b, view_num, c, h, w
    :return: feature_maps: shape: view_num, batch, c, h, w
    """

    feature_maps = []
    _, view_num, c, h, w = imgs.get_shape().as_list()
    with tf.variable_scope('feature_extraction_net', reuse=tf.AUTO_REUSE):
        for i in range(view_num):
            feature_map = feature_extraction_branch(imgs[:, i])
            feature_maps.append(feature_map)
    feature_maps = tf.transpose(feature_maps, [1, 0, 2, 3, 4])
    return feature_maps  # shape: batch, view_num, c, h, w


@layer_register(log_shape=True, use_scope=True)
def warping_layer(feature_maps, cams, depth_num):
    """
    :param feature_maps: feature maps output from feature_extraction_net, shape: b, view_num, c, h, w
    :param cams: Cams, shape: b, view_num
    :param depth_num: num of discrete depth value
    :return: cost volume
    """

    _, view_num, c, h, w = feature_maps.get_shape().as_list()
    _, cam_num = cams.get_shape().as_list()
    assert view_num == cam_num, 'view num: {} conflicts with cam num: {}'.format(view_num, cam_num)
    # get depth_start and depth_interval batch-wise
    depth_start, depth_interval = get_depth_meta(cams)
    depth_end = depth_start + (tf.cast(depth_num, tf.float32) - 1) * depth_interval

    # ref image and ref cam
    ref_cam = cams[:, 0]
    ref_feature_map = feature_maps[:, 0]

    # get homographies of all views
    view_homographies = []
    for view in range(1, view_num):
        view_cam = cams[:, view]
        homographies = get_homographies(ref_cam, view_cam, depth_num, depth_start, depth_interval)
        view_homographies.append(homographies)

    # shape of feature_map: b, c, h, w
    # shape of cost_volume: b, depth_num, c, h, w
    cost_volume = build_cost_volume(view_homographies, feature_maps, depth_num)

    return cost_volume


def cost_volume_regularization(cost_volume):
    with argscope([tf.layers.conv3d], use_bias=False, kernel_initializer=tf.glorot_uniform_initializer(),
                  kernel_regularizer=tf.contrib.layers.l2_regularizer(1.0), paddings='same'), \
         argscope([tf.layers.conv3d_transpose], use_bias=False, kernel_initializer=tf.glorot_uniform_initializer(),
                  kernel_regularizer=tf.contrib.layers.l2_regularizer(1.0), paddings='same'), \
         argscope([BatchNorm], epsilon=1e-5):
        base_filter = 8
        with tf.variable_scope('cost_volume_regularization'):
            l1_0 = tf.layers.conv3d(cost_volume, base_filter*2, 3, 2, activation=BNReLU, name='3dconv1_0')
            skip1_0 = tf.layers.conv3d(l1_0, base_filter*2, 3, 1, name='3dconv1_1')

            l2_0 = tf.layers.conv3d(l1_0, base_filter*4, 3, 2, activation=BNReLU, name='3dconv2_0')
            skip2_0 = tf.layers.conv3d(l2_0, base_filter*4, 3, 1, name='3dconv2_1')

            l3_0 = tf.layers.conv3d(l2_0, base_filter*8, 3, 2, activation=BNReLU, name='3dconv3_0')

            l0_1 = tf.layers.conv3d(cost_volume, base_filter, 3, 1, activation=BNReLU, name='3dconv0_1')

            l3_1 = tf.layers.conv3d(l3_0, base_filter*8, 3, 1, activation=BNReLU, name='3dconv3_1')
            l4_0 = tf.layers.conv3d_transpose(l3_1, base_filter*4, 3, 2, activation=BNReLU, name='3dconv4_0')

            l4_1 = tf.add(l4_0, skip2_0, name='3dconv4_1')
            l5_0 = tf.layers.conv3d_transpose(l4_1, base_filter*2, 3, 2, activation=BNReLU, name='3dconv5_0')

            l5_1 = tf.add(l5_0, skip1_0, name='3dconv5_1')
            l6_0 = tf.layers.conv3d_transpose(l5_1, base_filter, 3, 2, name='3dconv6_0')

            l6_1 = tf.add(l6_0, l0_1, name='3dconv6_1')
            l6_2 = tf.layers.conv3d(l6_1, 1, 3, 1, activation=None, name='3dconv6_2')

    return l6_2


def depth_refinement(coarse_depth_map):
    pass



