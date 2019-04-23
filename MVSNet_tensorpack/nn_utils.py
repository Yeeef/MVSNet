from tensorpack import *
from homography_utils import *
from upsample_utils import TFBilinearUpSample
import tensorflow as tf


__all__ = ['feature_extraction_net', 'warping_layer', 'cost_volume_regularization', 'soft_argmin', 'depth_refinement',
           ]


def feature_extraction_branch(img):
    """
    Already in the scope of {[tf.layers.conv3d, tf.layers.conv3d_transpose, Conv2D, MaxPooling, AvgPooling, BatchNorm], data_format=self.data_format, padding='same'}
    :param img: shape: b * view_num * c * h * w
    :return: l
    """
    with argscope([Conv2D], use_bias=False, kernel_initializer=tf.glorot_uniform_initializer(),
                  kernel_regularizer=tf.contrib.layers.l2_regularizer(1.0), padding='same'), \
         argscope([BatchNorm], epsilon=1e-5, momentum=0.99):
        with tf.variable_scope('feature_extraction_branch', reuse=tf.AUTO_REUSE):
            base_filter = 8
            l = Conv2D('conv0_0', img, base_filter, 3, strides=1, activation=BNReLU)
            l = Conv2D('conv0_1', l, base_filter, 3, strides=1, activation=BNReLU)

            l = Conv2D('conv1_0', l, base_filter * 2, 5, strides=2, activation=BNReLU)
            l = Conv2D('conv1_1', l, base_filter * 2, 3, strides=1, activation=BNReLU)
            l = Conv2D('conv1_2', l, base_filter * 2, 3, strides=1, activation=BNReLU)

            l = Conv2D('conv2_0', l, base_filter * 4, 5, strides=2, activation=BNReLU)
            l = Conv2D('conv2_1', l, base_filter * 4, 3, strides=1, activation=BNReLU)
            feature_map = Conv2D('conv2_1', l, base_filter * 4, 3, strides=1, activation=None)

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
def warping_layer(feature_maps, cams, depth_start, depth_interval, depth_num):
    """
    :param feature_maps: feature maps output from feature_extraction_net, shape: b, view_num, c, h, w
    :param cams: Cams, shape: b, view_num
    :param depth_start: TODO
    :param depth_interval: TODO
    :param depth_num: num of discrete depth value
    :return: cost volume
    """

    _, view_num, c, h, w = feature_maps.get_shape().as_list()
    _, cam_num, *_ = cams.get_shape().as_list()
    assert view_num == cam_num, 'view num: {} conflicts with cam num: {}'.format(view_num, cam_num)

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
    # shape of cost_volume: b, c, depth_num, h, w
    cost_volume = build_cost_volume(view_homographies, feature_maps, depth_num)

    return cost_volume


def cost_volume_regularization(cost_volume):
    with argscope([tf.layers.conv3d], use_bias=False, kernel_initializer=tf.glorot_uniform_initializer(),
                  kernel_regularizer=tf.contrib.layers.l2_regularizer(1.0), padding='same'), \
         argscope([tf.layers.conv3d_transpose], use_bias=False, kernel_initializer=tf.glorot_uniform_initializer(),
                  kernel_regularizer=tf.contrib.layers.l2_regularizer(1.0), padding='same'), \
         argscope([tf.layers.batch_normalization], epsilon=1e-5, momentum=0.99):
        base_filter = 8
        with tf.variable_scope('cost_volume_regularization'):
            with rename_tflayer_get_variable():
                l1_0 = tf.layers.conv3d(cost_volume, base_filter * 2, 3, strides=2, activation=BNReLU, name='3dconv1_0')
                skip1_0 = tf.layers.conv3d(l1_0, base_filter * 2, 3, strides=1, activation=BNReLU, name='3dconv1_1')
                
                l2_0 = tf.layers.conv3d(l1_0, base_filter * 4, 3, strides=2, activation=BNReLU, name='3dconv2_0')
                skip2_0 = tf.layers.conv3d(l2_0, base_filter * 4, 3, strides=1, activation=BNReLU, name='3dconv2_1')
                
                l3_0 = tf.layers.conv3d(l2_0, base_filter * 8, 3, strides=2, activation=BNReLU, name='3dconv3_0')
                
                l0_1 = tf.layers.conv3d(cost_volume, base_filter, 3, strides=1, activation=BNReLU, name='3dconv0_1')
                
                l3_1 = tf.layers.conv3d(l3_0, base_filter * 8, 3, strides=1, activation=BNReLU, name='3dconv3_1')
                l4_0 = tf.layers.conv3d_transpose(l3_1, base_filter * 4, 3, strides=2, activation=BNReLU, name='3dconv4_0')
                
                l4_1 = tf.add(l4_0, skip2_0, name='3dconv4_1')
                l5_0 = tf.layers.conv3d_transpose(l4_1, base_filter * 2, 3, strides=2, activation=BNReLU, name='3dconv5_0')
                
                l5_1 = tf.add(l5_0, skip1_0, name='3dconv5_1')
                l6_0 = tf.layers.conv3d_transpose(l5_1, base_filter, 3, strides=2, name='3dconv6_0')
                
                l6_1 = tf.add(l6_0, l0_1, name='3dconv6_1')
                
                # shape of l6_2: b, 1, d, h, w
                l6_2 = tf.layers.conv3d(l6_1, 1, 3, strides=1, activation=None, name='3dconv6_2')
                
                # shape: b, d, h, w
                regularized_cost_volume = tf.squeeze(l6_2, axis=1, name='regularized_cost_volume')
                prob_volume = tf.nn.softmax(
                    tf.scalar_mul(-1, regularized_cost_volume), axis=1, name='prob_volume'
                )

    return prob_volume


def soft_argmin(prob_volume, depth_start, depth_end, depth_num):
    with tf.variable_scope('soft_argmin'):
        volume_shape = tf.shape(prob_volume)
        batch_size = volume_shape[0]
        soft_2d = []
        for i in range(batch_size):
            # shape of 1d: (depth_num, )
            soft_1d = tf.linspace(depth_start[i], depth_end[i], tf.cast(depth_num, tf.int32))
            # shape of 2d: (batch_size, depth_num)
            soft_2d.append(soft_1d)
        # shape of 2d: (b, d, 1, 1)
        soft_2d = tf.reshape(tf.stack(soft_2d, axis=0), [volume_shape[0], volume_shape[1], 1, 1])
        soft_4d = tf.tile(soft_2d, [1, 1, volume_shape[2], volume_shape[3]])
        # shape: (b, 1, h, w)
        estimated_depth_map = tf.reduce_sum(soft_4d * prob_volume, axis=1, keep_dims=True, name='coarse_depth')
        # # shape: (b, 1, h, w)
        # estimated_depth_map = tf.expand_dims(estimated_depth_map, axis=1)

    return estimated_depth_map


def get_prob_map(prob_volume, coarse_depth, depth_start, depth_interval):
    pass


def depth_refinement(coarse_depth, img, depth_start, depth_end):
    """
    get refine depth from coarse depth
    :param coarse_depth: b, 1, h, w
    :param img: b, 3, h, w
    :param depth_start: b,
    :param depth_end: b,
    :return: refine_depth b, 1, h, w
    """
    with tf.variable_scope('depth_refinement'):
        batch_size, _, h, w = tf.shape(coarse_depth)
        # shape: b, h, w, 1
        depth_start_mat = tf.tile(tf.reshape(
            depth_start, [batch_size, 1, 1, 1]), [1, 1, h, w])
        # shape: b, h, w, 1
        depth_end_mat = tf.tile(tf.reshape(
            depth_end, [batch_size, 1, 1, 1]), [1, 1, h, w])
        # shape: b, 1, h, w
        depth_scale_mat = depth_end_mat - depth_start_mat

        normalized_coarse_depth = tf.div(coarse_depth - depth_start_mat, depth_scale_mat)

        # resize normalized img to the same size of coarse depth
        resize_img = TFBilinearUpSample(img, [h, w], 'NCHW')

        refine_depth = depth_refinement_net(normalized_coarse_depth, resize_img)

    return refine_depth


def depth_refinement_net(coarse_depth, img):
    """

    :param coarse_depth: shape: b, 1, h, w
    :param img: b, 3, h, w
    :return:
    """
    with argscope([Conv2D], use_bias=False, kernel_initializer=tf.glorot_uniform_initializer(),
                  kernel_regularizer=tf.contrib.layers.l2_regularizer(1.0), padding='same'), \
         argscope([BatchNorm], epsilon=1e-5, momentum=0.99):
        with tf.variable_scope('depth_refine_net'):
            # shape: b, 4, h, w
            concat_input = tf.concat((img, coarse_depth), axis=1, name='concat_input')
            l0 = Conv2D('conv0', concat_input, 32, 3, strides=1, activation=BNReLU)
            l1 = Conv2D('conv1', l0, 32, 3, strides=1, activation=BNReLU)
            l2 = Conv2D('conv2', l1, 32, 3, strides=1, activation=BNReLU)
            l3 = Conv2D('conv3', l2, 1, 3, strides=1, activation=None, use_bias=True)

            # shape: b, 1, h, w
            refine_depth = tf.add(coarse_depth, l3, name='refine_depth')

    return refine_depth


@layer_register(use_scope=None)
def BatchNorm3D(
        inputs, axis=None, training=None, momentum=0.99, epsilon=1e-5,
        center=True, scale=True,
        beta_initializer=tf.zeros_initializer(),
        gamma_initializer=tf.ones_initializer(),
        trainable=True,
        name=None,
        reuse=None,
        renorm=False,
        renorm_clipping=None,
        renorm_momentum=0.99,
        fused=None,
        virtual_batch_size=None,
        adjustment=None

):
    pass

@layer_register(use_scope=None)
def BNReLU3D(x, name=None):
    """
    A shorthand of BatchNormalization + ReLU.
    """
    with rename_tflayer_get_variable():
        x = tf.layers.batch_normalization('bn', x)
        x = tf.nn.relu(x, name=name)
    return x