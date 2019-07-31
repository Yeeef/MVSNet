from tensorpack import *
from homography_utils import *
from upsample_utils import TFBilinearUpSample
import tensorflow as tf
from tensorpack.utils import logger
from tensorpack.tfutils.collection import *


# __all__ = ['feature_extraction_net', 'warping_layer', 'cost_volume_regularization', 'soft_argmin', 'depth_refinement',
#            ]


def uni_feature_extraction_branch(img):
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


def unet_feature_extraction_branch(img):
    """
    2D U-Net with gn
    Already in the scope of {[tf.layers.conv3d, tf.layers.conv3d_transpose, Conv2D, MaxPooling, AvgPooling, BatchNorm], data_format=self.data_format, padding='same'}
    :param img: shape: b * view_num * c * h * w
    :return: l
    """
    with argscope([Conv2D, Conv2DTranspose], use_bias=False, kernel_initializer=tf.glorot_uniform_initializer(),
                  kernel_regularizer=tf.contrib.layers.l2_regularizer(1.0), padding='same'):
         # argscope([mvsnet_gn], data_format='channels_first'):
        with tf.variable_scope('feature_extraction_branch', reuse=tf.AUTO_REUSE):
            base_filter = 8
            l1_0 = Conv2D('2dconv1_0', img, base_filter*2, 3, strides=2, activation=mvsnet_gn_relu)
            l2_0 = Conv2D('2dconv2_0', l1_0, base_filter*4, 3, strides=2, activation=mvsnet_gn_relu)
            l3_0 = Conv2D('2dconv3_0', l2_0, base_filter*8, 3, strides=2, activation=mvsnet_gn_relu)
            l4_0 = Conv2D('2dconv4_0', l3_0, base_filter*16, 3, strides=2, activation=mvsnet_gn_relu)

            l0_1 = Conv2D('2dconv0_1', img, base_filter, 3, strides=1, activation=mvsnet_gn_relu)
            l0_2 = Conv2D('2dconv0_2', l0_1, base_filter, 3, strides=1, activation=mvsnet_gn_relu)

            l1_1 = Conv2D('2dconv1_1', l1_0, base_filter*2, 3, strides=1, activation=mvsnet_gn_relu)
            l1_2 = Conv2D('2dconv1_2', l1_1, base_filter*2, 3, strides=1, activation=mvsnet_gn_relu)

            l2_1 = Conv2D('2dconv2_1', l2_0, base_filter*4, 3, strides=1, activation=mvsnet_gn_relu)
            l2_2 = Conv2D('2dconv2_2', l2_1, base_filter*4, 3, strides=1, activation=mvsnet_gn_relu)

            l3_1 = Conv2D('2dconv3_1', l3_0, base_filter*8, 3, strides=1, activation=mvsnet_gn_relu)
            l3_2 = Conv2D('2dconv3_2', l3_1, base_filter*8, 3, strides=1, activation=mvsnet_gn_relu)

            l4_1 = Conv2D('2dconv4_1', l4_0, base_filter*16, 3, strides=1, activation=mvsnet_gn_relu)
            l4_2 = Conv2D('2dconv4_2', l4_1, base_filter*16, 3, strides=1, activation=mvsnet_gn_relu)
            # 1 / 8
            l5_0 = Conv2DTranspose('2dconv5_0', l4_2, base_filter*8, 3, strides=2, activation=mvsnet_gn_relu)

            # 1 / 4
            concat5_0 = tf.concat((l5_0, l3_2), axis=3, name='2dconcat5_0')
            l5_1 = Conv2D('2dconv5_1', concat5_0, base_filter*8, 3, strides=1, activation=mvsnet_gn_relu)
            l5_2 = Conv2D('2dconv5_2', l5_1, base_filter*8, 3, strides=1, activation=mvsnet_gn_relu)
            # 1 / 2
            l6_0 = Conv2DTranspose('2dconv6_0', l5_2, base_filter*4, 3, strides=2, activation=mvsnet_gn_relu)

            concat6_0 = tf.concat((l6_0, l2_2), axis=3, name='2dconcat6_0')
            l6_1 = Conv2D('2dconv6_1', concat6_0, base_filter*4, 3, strides=1, activation=mvsnet_gn_relu)
            l6_2 = Conv2D('2dconv6_2', l6_1, base_filter*4, 3, strides=1, activation=mvsnet_gn_relu)
            l7_0 = Conv2DTranspose('2dconv7_0', l6_2, base_filter*2, 3, strides=2, activation=mvsnet_gn_relu)

            concat7_0 = tf.concat((l7_0, l1_2), axis=3, name='2dconcat7_0')
            l7_1 = Conv2D('2dconv7_1', concat7_0, base_filter*2, 3, strides=1, activation=mvsnet_gn_relu)
            l7_2 = Conv2D('2dconv7_2', l7_1, base_filter*2, 3, strides=1, activation=mvsnet_gn_relu)
            l8_0 = Conv2DTranspose('2dconv8_0', l7_2, base_filter, 3, strides=2, activation=mvsnet_gn_relu)

            concat8_0 = tf.concat((l8_0, l0_2), axis=3, name='2dconcat8_0')
            l8_1 = Conv2D('2dconv8_1', concat8_0, base_filter, 3, strides=1, activation=mvsnet_gn_relu)
            l8_2 = Conv2D('2dconv8_2', l8_1, base_filter, 3, strides=1, activation=mvsnet_gn_relu)

            l9_0 = Conv2D('2dconv9_0', l8_2, base_filter*2, 5, strides=2, activation=mvsnet_gn_relu)
            l9_1 = Conv2D('2dconv9_1', l9_0, base_filter*2, 3, strides=1, activation=mvsnet_gn_relu)
            l9_2 = Conv2D('2dconv9_2', l9_1, base_filter*2, 3, strides=1, activation=mvsnet_gn_relu)

            l10_0 = Conv2D('2dconv10_0', l9_2, base_filter*4, 5, strides=2, activation=mvsnet_gn_relu)
            l10_1 = Conv2D('2dconv10_1', l10_0, base_filter*4, 3, strides=1, activation=mvsnet_gn_relu)

            feature_map = Conv2D('2dconv10_2', l10_1, base_filter*4, 3, strides=1, activation=None)

            return feature_map


def feature_extraction_net(imgs, branch_function):
    """
    feature extraction net
    Take care of variable_scope's reuse param!
    :param imgs: shape: b, view_num, c, h, w
    :return: feature_maps: shape: view_num, batch, c, h, w
    """

    feature_maps = []
    _, view_num, c, h, w = imgs.get_shape().as_list()
    ctx = get_current_tower_context()
    if ctx.is_main_training_tower:
        reuse_flag = False
    else:
        reuse_flag = True
    with tf.variable_scope('feature_extraction_net', reuse=reuse_flag):
        # ref view
        feature_map = branch_function(imgs[:, 0])
        feature_maps.append(feature_map)
    with tf.variable_scope('feature_extraction_net', reuse=True):
        for i in range(1, view_num):
            feature_map = branch_function(imgs[:, i])
            feature_maps.append(feature_map)
            # transpose is aiming at swap the position of batch and view_num
    feature_maps = tf.transpose(feature_maps, [1, 0, 2, 3, 4], name='feature_maps')
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
    with tf.variable_scope('warping_layer'):
        _, view_num, c, h, w = feature_maps.get_shape().as_list()
        _, view_num, h, w, c = feature_maps.get_shape().as_list()

        _, cam_num, *_ = cams.get_shape().as_list()
        assert view_num == cam_num, 'view num: {} conflicts with cam num: {}'.format(view_num, cam_num)
        
        # ref image and ref cam
        ref_cam = cams[:, 0]
        ref_feature_map = feature_maps[:, 0]
        
        # get homographies of all views
        view_homographies = []
        for view in range(1, view_num):
            # view_cam = cams[:, view]
            view_cam = tf.squeeze(tf.slice(cams, [0, view, 0, 0, 0], [-1, 1, 2, 4, 4]), axis=1)
            homographies = get_homographies(ref_cam, view_cam, depth_num=depth_num, depth_start=depth_start,
                                                    depth_interval=depth_interval)
            view_homographies.append(homographies)
        
        # shape of feature_map: b, h, w, c
        # shape of cost_volume: b, depth_num, h, w, c
        cost_volume = build_cost_volume(view_homographies, feature_maps, depth_num)

    return cost_volume


def simple_cost_volume_regularization(cost_volume, training, trainable):
    with argscope([tf.layers.conv3d], use_bias=False, kernel_initializer=tf.glorot_uniform_initializer(),
                  kernel_regularizer=tf.contrib.layers.l2_regularizer(1.0), padding='same'), \
         argscope([tf.layers.conv3d_transpose], use_bias=False, kernel_initializer=tf.glorot_uniform_initializer(),
                  kernel_regularizer=tf.contrib.layers.l2_regularizer(1.0), padding='same'), \
         argscope([tf.layers.batch_normalization], epsilon=1e-5, momentum=0.99):
        base_filter = 8
        with tf.variable_scope('cost_volume_regularization'):
            with rename_tflayer_get_variable():
                l1_0 = conv3d_bn_relu(cost_volume, base_filter * 2, 3, strides=2, training=training,
                                      trainable=trainable,
                                      name='3dconv1_0')
                l2_0 = conv3d_bn_relu(l1_0, base_filter * 4, 3, 2, training, trainable, '3dconv2_0')
                l5_0 = deconv3d_bn_relu(l2_0, base_filter * 2, 3, 2, training, trainable, name='3dconv5_0')
                l6_0 = deconv3d_bn_relu(l5_0, base_filter, 3, 2, training, trainable, name='3dconv6_0')
                l6_2 = tf.layers.conv3d(l6_0, 1, 3, strides=1, activation=None, name='3dconv6_2')

                regularized_cost_volume = tf.squeeze(l6_2, axis=4, name='regularized_cost_volume')

    return regularized_cost_volume


def cost_volume_regularization(cost_volume, training, trainable):

    with argscope([tf.layers.conv3d], use_bias=False, kernel_initializer=tf.glorot_uniform_initializer(),
                  kernel_regularizer=tf.contrib.layers.l2_regularizer(1.0), padding='same'), \
         argscope([tf.layers.conv3d_transpose], use_bias=False, kernel_initializer=tf.glorot_uniform_initializer(),
                  kernel_regularizer=tf.contrib.layers.l2_regularizer(1.0), padding='same'), \
         argscope([tf.layers.batch_normalization], epsilon=1e-5, momentum=0.99):
        base_filter = 8
        with tf.variable_scope('cost_volume_regularization'):
            with rename_tflayer_get_variable():
                l1_0 = conv3d_bn_relu(cost_volume, base_filter*2, 3, strides=2, training=training, trainable=trainable,
                                            name='3dconv1_0')

                # l1_0 = tf.layers.conv3d(cost_volume, base_filter * 2, 3, strides=2, activation=None, name='3dconv1_0')
                # l1_0 = tf.layers.batch_normalization(l1_0, training=training, trainable=trainable, reuse=None, name='3dconv1_0_bn')

                # skip1_0 = tf.layers.conv3d(l1_0, base_filter * 2, 3, strides=1, activation=None, name='3dconv1_1')
                skip1_0 = conv3d_bn_relu(l1_0, base_filter*2, 3, 1, training, trainable, '3dconv1_1')

                l2_0 = conv3d_bn_relu(l1_0, base_filter*4, 3, 2, training, trainable, '3dconv2_0')
                # l2_0 = tf.layers.conv3d(l1_0, base_filter * 4, 3, strides=2, activation=BNReLU, name='3dconv2_0')
                # skip2_0 = tf.layers.conv3d(l2_0, base_filter * 4, 3, strides=1, activation=None, name='3dconv2_1')
                skip2_0 = conv3d_bn_relu(l2_0, base_filter*4, 3, 1, training, trainable, '3dconv2_1')

                # l3_0 = tf.layers.conv3d(l2_0, base_filter * 8, 3, strides=2, activation=BNReLU, name='3dconv3_0')
                l3_0 = conv3d_bn_relu(l2_0, base_filter*8, 3, 2, training, trainable, '3dconv3_0')
                
                # l0_1 = tf.layers.conv3d(cost_volume, base_filter, 3, strides=1, activation=BNReLU, name='3dconv0_1')
                l0_1 = conv3d_bn_relu(cost_volume, base_filter, 3, 1, training, trainable, name='3dconv0_1')
                
                # l3_1 = tf.layers.conv3d(l3_0, base_filter * 8, 3, strides=1, activation=BNReLU, name='3dconv3_1')
                l3_1 = conv3d_bn_relu(l3_0, base_filter*8, 3, 1, training, trainable, name='3dconv3_1')
                # l4_0 = tf.layers.conv3d_transpose(l3_1, base_filter * 4, 3, strides=2, activation=BNReLU, name='3dconv4_0')
                l4_0 = deconv3d_bn_relu(l3_1, base_filter*4, 3, 2, training, trainable, name='3dconv4_0')

                
                l4_1 = tf.add(l4_0, skip2_0, name='3dconv4_1')
                # l5_0 = tf.layers.conv3d_transpose(l4_1, base_filter * 2, 3, strides=2, activation=BNReLU, name='3dconv5_0')
                l5_0 = deconv3d_bn_relu(l4_1, base_filter*2, 3, 2, training, trainable, name='3dconv5_0')
                
                l5_1 = tf.add(l5_0, skip1_0, name='3dconv5_1')
                # l6_0 = tf.layers.conv3d_transpose(l5_1, base_filter, 3, strides=2, name='3dconv6_0')
                l6_0 = deconv3d_bn_relu(l5_1, base_filter, 3, 2, training, trainable, name='3dconv6_0')
                
                l6_1 = tf.add(l6_0, l0_1, name='3dconv6_1')
                
                # shape of l6_2: b, 1, d, h, w
                l6_2 = tf.layers.conv3d(l6_1, 1, 3, strides=1, activation=None, name='3dconv6_2')
                # l6_2 = conv3d_bn_relu(l6_1, 1, 3, 1, training, trainable, name='3dconv6_2')
                # l6_2 = tf.layers.conv3d(l6_1, 1, 3, 1, training, trainable, name='3dconv6_2')

                # shape: b, d, h, w
                regularized_cost_volume = tf.squeeze(l6_2, axis=4, name='regularized_cost_volume')

    return regularized_cost_volume


def gru_regularization(cost_volume, training, trainable):
    with argscope([tf.layers.conv2d], use_bias=False, kernel_initializer=tf.glorot_uniform_initializer(),
                  kernel_regularizer=tf.contrib.layers.l2_regularizer(1.0), padding='same'), \
         argscope([tf.layers.batch_normalization], epsilon=1e-5, momentum=0.99):
        with tf.variable_scope("gru_regularization"):
            with rename_tflayer_get_variable():
                gru1_filters = 16
                gru2_filters = 4
                gru3_filters = 2
                # b, d, h, w, c
                cost_volume_shape = tf.shape(cost_volume)
                batch = cost_volume_shape[0]
                _, d, h, w, c = cost_volume.get_shape().as_list()
                gru_input_shape = [h, w]
                state1 = tf.zeros([batch, h, w, gru1_filters])
                state2 = tf.zeros([batch, h, w, gru2_filters])
                state3 = tf.zeros([batch, h, w, gru3_filters])
                conv_gru1 = ConvGRUCell(shape=gru_input_shape, kernel=[3, 3], filters=gru1_filters)
                conv_gru2 = ConvGRUCell(shape=gru_input_shape, kernel=[3, 3], filters=gru2_filters)
                conv_gru3 = ConvGRUCell(shape=gru_input_shape, kernel=[3, 3], filters=gru3_filters)
                
                seperate_cost_volumes = tf.split(cost_volume, d, axis=1)
                depth_costs = []
                for single_cost_volume in seperate_cost_volumes:
                    # b, h, w, c
                    # gru
                    single_cost_volume = tf.squeeze(single_cost_volume, axis=1)
                    reg_cost1, state1 = conv_gru1(-single_cost_volume, state1, scope='conv_gru1')
                    reg_cost2, state2 = conv_gru2(reg_cost1, state2, scope='conv_gru2')
                    reg_cost3, state3 = conv_gru3(reg_cost2, state3, scope='conv_gru3')
                    # reg_cost: b, h, w, 1
                    reg_cost = tf.layers.conv2d(
                        reg_cost3, 1, 3, padding='same', reuse=tf.AUTO_REUSE, name='prob_conv', use_bias=True)
                    depth_costs.append(reg_cost)
                # prob_volume: b, d, h, w, 1
                prob_volume = tf.stack(depth_costs, axis=1)
                prob_volume = tf.nn.softmax(prob_volume, axis=1, name='prob_volume')
                return prob_volume


# @layer_register(use_scope=True)
def conv3d_bn_relu(inputs, filters, kernel_size, strides, training, trainable, name):
    ctx = get_current_tower_context()
    training = ctx.is_training
    trainable = ctx.is_training
    l = tf.layers.conv3d(inputs, filters, kernel_size, strides=strides, activation=None, reuse=None, name=name)
    l = tf.layers.batch_normalization(l, training=training, trainable=trainable, reuse=None, name=name + '_bn')
    l = tf.nn.relu(l, name=name + '_relu')
    return l


# @layer_register(use_scope=True)
def deconv3d_bn_relu(inputs, filters, kernel_size, strides, training, trainable, name):
    ctx = get_current_tower_context()
    training = ctx.is_training
    trainable = ctx.is_training
    l = tf.layers.conv3d_transpose(inputs, filters, kernel_size, strides=strides, activation=None, reuse=None, name=name)
    l = tf.layers.batch_normalization(l, training=training, trainable=trainable, reuse=None, name=name + '_bn')
    l = tf.nn.relu(l, name=name + '_relu')
    return l


@layer_register(log_shape=True)
def soft_argmin(regularized_cost_volume, depth_start, depth_end, depth_num, depth_interval, batch_size):
    with tf.variable_scope('soft_argmin'):
        # batch_size =
        # b, d, h, w
        probability_volume = tf.nn.softmax(
            tf.scalar_mul(-1, regularized_cost_volume), axis=1, name='prob_volume')
        volume_shape = tf.shape(probability_volume)
        # batch_size = volume_shape[0]
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
        estimated_depth_map = tf.reduce_sum(soft_4d * probability_volume, axis=1, name='coarse_depth')
        # # shape: (b, 1, h, w)
        # estimated_depth_map = tf.expand_dims(estimated_depth_map, axis=1)
        # shape of prob_map: b, h, w, 1
        estimated_depth_map = tf.expand_dims(estimated_depth_map, axis=3)
        prob_map = get_propability_map(probability_volume, estimated_depth_map, depth_start, depth_interval)
    return estimated_depth_map, prob_map


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
        *_, h, w = coarse_depth.get_shape().as_list()
        batch_size = tf.shape(coarse_depth)[0]
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
        resize_img = TFBilinearUpSample('downsample_img', img, [h, w], 'NCHW')

        refine_depth = depth_refinement_net(normalized_coarse_depth, resize_img)
        refine_depth = tf.multiply(refine_depth, depth_scale_mat) + depth_start_mat

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
            l3 = Conv2D('conv3', l2, 1, 3, strides=1, activation=None)

            # shape: b, 1, h, w
            refine_depth = tf.add(coarse_depth, l3, name='refine_depth')

    return refine_depth


@layer_register(log_shape=True)
def mvsnet_gn(x, group=32, group_channel=8, epsilon=1e-5,
              channel_wise=True, data_format='channels_last',
              beta_initializer=tf.constant_initializer(),
              gamma_initializer=tf.constant_initializer(1.)):
    assert len(x.get_shape().as_list()) == 4, len(x.get_shape().as_list())
    assert data_format in ['channels_first', 'channels_last'], data_format

    if data_format == 'channels_first':
        _, c, h, w = x.get_shape().as_list()

        logger.info('fuck you fuck you! %s' % data_format)
    else:
        _, h, w, c = x.get_shape().as_list()
        x = tf.transpose(x, [0, 3, 1, 2])
        # assert c < 100, c
    if channel_wise:
        g = tf.cast(tf.maximum(1, c // group_channel), tf.int32)
    else:
        g = tf.cast(tf.minimum(group, c), tf.int32)

    # normalization
    # tf.Print()
    x = tf.reshape(x, (-1, g, c // g, h, w))
    new_shape = [1, c, 1, 1]
    mean, var = tf.nn.moments(x, [2, 3, 4], keep_dims=True)
    beta = tf.get_variable('beta', [c], dtype=tf.float32, initializer=beta_initializer)
    beta = tf.reshape(beta, new_shape)
    gamma = tf.get_variable('gamma', [c], dtype=tf.float32, initializer=gamma_initializer)
    gamma = tf.reshape(gamma, new_shape)
    x = (x - mean) / tf.sqrt(var + epsilon)
    x = tf.reshape(x, [-1, c, h, w]) * gamma + beta

    if data_format == 'channels_last':
        x = tf.transpose(x, [0, 2, 3, 1])
    return x


def mvsnet_gn_relu(x, name=None):
    """
    GroupNorm + ReLU.
    """
    x = mvsnet_gn('gn', x)
    x = tf.nn.relu(x, name='relu')
    return x


@layer_register
def GroupNorm(x, group, gamma_initializer=tf.constant_initializer(1.)):
    """
    https://arxiv.org/abs/1803.08494
    More code that reproduces the paper can be found at https://github.com/ppwwyyxx/GroupNorm-reproduce/.
    """
    shape = x.get_shape().as_list()
    ndims = len(shape)
    assert ndims == 4, shape
    chan = shape[1]
    assert chan % group == 0, chan
    group_size = chan // group

    orig_shape = tf.shape(x)
    h, w = orig_shape[2], orig_shape[3]

    x = tf.reshape(x, tf.stack([-1, group, group_size, h, w]))

    mean, var = tf.nn.moments(x, [2, 3, 4], keep_dims=True)

    new_shape = [1, group, group_size, 1, 1]

    beta = tf.get_variable('beta', [chan], initializer=tf.constant_initializer())
    beta = tf.reshape(beta, new_shape)

    gamma = tf.get_variable('gamma', [chan], initializer=gamma_initializer)
    gamma = tf.reshape(gamma, new_shape)

    out = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-5, name='output')
    return tf.reshape(out, orig_shape, name='output')


class ConvGRUCell(tf.contrib.rnn.RNNCell):
    """A GRU cell with convolutions instead of multiplications."""

    def __init__(self,
                 shape,
                 filters,
                 kernel,
                 initializer=None,
                 activation=tf.tanh,
                 normalize=True,
                 data_format='channels_last'):
        self._filters = filters
        self._kernel = kernel
        self._initializer = initializer
        self._activation = activation
        # self._size = tf.TensorShape(shape + [self._filters])
        # self._size = tf.TensorShape([shape[0], shape[1], self._filters])

        self._normalize = normalize
        # TODO: feature_axis = 3?
        # TODO: because the first axis is batch, actually we can just rewrite it as -1
        if data_format == 'channels_last':
            self._feature_axis = -1
        else:
            self._feature_axis = 1


    # @property
    # def state_size(self):
    #     return self._size
    #
    # @property
    # def output_size(self):
    #     return self._size

    def __call__(self, x, h, scope=None):
        # shape of x,h: b,h,w,c
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            with tf.variable_scope('Gates'):
                # concatenation channel-wise
                inputs = tf.concat([x, h], axis=self._feature_axis)

                # convolution
                conv = tf.layers.conv2d(
                    inputs, 2 * self._filters, self._kernel, padding='same', name='conv')
                reset_gate, update_gate = tf.split(conv, 2, axis=self._feature_axis)

                with tf.variable_scope("reset_gate"):

                    reset_gate = mvsnet_gn('gn', reset_gate, group_channel=16)
                    reset_gate = tf.sigmoid(reset_gate)
                with tf.variable_scope("update_gate"):

                    update_gate = mvsnet_gn('gn', update_gate, group_channel=16)
                    update_gate = tf.sigmoid(update_gate)

            with tf.variable_scope('Output'):
                # concatenation
                inputs = tf.concat([x, reset_gate * h], axis=self._feature_axis)

                # convolution
                conv = tf.layers.conv2d(
                    inputs, self._filters, self._kernel, padding='same', name='output_conv')

                # group normalization
                conv = mvsnet_gn('output_gn', conv,  group_channel=16)

                # activation
                y = self._activation(conv)

                # soft update
                output = update_gate * h + (1 - update_gate) * y

            return output, output
