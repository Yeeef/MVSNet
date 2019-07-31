from tensorpack import *
import tensorflow as tf


def mvsnet_regression_loss(gt_depth, pred_depth, depth_interval, scope_name):

    with tf.variable_scope(scope_name):
        masked_mae = non_zero_mean_absolute_diff(gt_depth, pred_depth, depth_interval)

        less_one_accuracy = less_one_percentage(gt_depth, pred_depth, depth_interval)

        less_three_accuracy = less_three_percentage(gt_depth, pred_depth, depth_interval)

    return masked_mae, less_one_accuracy, less_three_accuracy


def mvsnet_classification_loss(prob_volume, gt_depth_image, depth_num, depth_start, depth_interval):
    """ compute loss and accuracy """

    # get depth mask
    mask_true = tf.cast(tf.not_equal(gt_depth_image, 0.0), dtype='float32')
    valid_pixel_num = tf.reduce_sum(mask_true, axis=[1, 2, 3]) + 1e-7
    # gt depth map -> gt index map
    shape = tf.shape(gt_depth_image)
    start_mat = tf.tile(tf.reshape(depth_start, [shape[0], 1, 1, 1]), [1, shape[1], shape[2], 1])

    interval_mat = tf.tile(tf.reshape(depth_interval, [shape[0], 1, 1, 1]), [1, shape[1], shape[2], 1])
    gt_index_image = tf.div(gt_depth_image - start_mat, interval_mat)
    gt_index_image = tf.multiply(mask_true, gt_index_image)
    gt_index_image = tf.cast(tf.round(gt_index_image), dtype='int32')
    # gt index map -> gt one hot volume (B x H x W x 1)
    gt_index_volume = tf.one_hot(gt_index_image, depth_num, axis=1)
    # cross entropy image (B x H x W x 1)
    cross_entropy_image = -tf.reduce_sum(gt_index_volume * tf.log(prob_volume), axis=1)
    # masked cross entropy loss
    masked_cross_entropy_image = tf.multiply(mask_true, cross_entropy_image)
    masked_cross_entropy = tf.reduce_sum(masked_cross_entropy_image, axis=[1, 2, 3])
    masked_cross_entropy = tf.reduce_sum(masked_cross_entropy / valid_pixel_num)

    # winner-take-all depth map
    wta_index_map = tf.cast(tf.argmax(prob_volume, axis=1), dtype='float32')
    wta_depth_map = wta_index_map * interval_mat + start_mat

    # non zero mean absulote loss
    masked_mae = non_zero_mean_absolute_diff(gt_depth_image, wta_depth_map, tf.abs(depth_interval))
    # less one accuracy
    less_one_accuracy = less_one_percentage(gt_depth_image, wta_depth_map, tf.abs(depth_interval))
    # less three accuracy
    less_three_accuracy = less_three_percentage(gt_depth_image, wta_depth_map, tf.abs(depth_interval))

    return masked_cross_entropy, masked_mae, less_one_accuracy, less_three_accuracy, wta_depth_map


def non_zero_mean_absolute_diff(gt_depth, pred_depth, depth_interval):
    """
    non zero mean absolute error(MAE)
    :param gt_depth:
    :param pred_depth:
    :param depth_interval:
    :return:
    """
    with tf.variable_scope('MAE'):
        # *_, h, w = tf.shape(pred_depth)
        batch_size = tf.shape(pred_depth)[0]
        depth_interval = tf.reshape(depth_interval, [batch_size])
        # we have masked out wrong depth before as 0.
        mask_true = tf.cast(tf.not_equal(gt_depth, 0.0), dtype='float32')
        # shape of denom: (b,)
        denom = tf.reduce_sum(mask_true, axis=[1, 2, 3]) + 1e-7
        # shape: b, 1, h, w
        masked_abs_error = tf.abs(mask_true * (gt_depth - pred_depth))  # 4D
        # shape: (b,)
        masked_mae = tf.reduce_sum(masked_abs_error, axis=[1, 2, 3])  # 1D
        masked_mae = tf.reduce_sum((masked_mae / depth_interval) / denom)  # 1

    return masked_mae


def less_one_percentage(gt_depth, pred_depth, depth_interval):
    """
    less one accuracy for one batch
    :param gt_depth:
    :param pred_depth:
    :param depth_interval:
    :return:
    """
    with tf.variable_scope('less_one_error'):
        *_, h, w = pred_depth.get_shape().as_list()
        batch_size = tf.shape(pred_depth)[0]
        mask_true = tf.cast(tf.not_equal(gt_depth, 0.0), dtype='float32')
        # shape: () denom is a scalar
        denom = tf.reduce_sum(mask_true) + 1e-7
        interval_image = tf.tile(tf.reshape(depth_interval, [batch_size, 1, 1, 1]), [1, 1, h, w])
        abs_diff_image = tf.abs(gt_depth - pred_depth) / interval_image
        less_one_image = mask_true * tf.cast(tf.less_equal(abs_diff_image, 1.0), dtype='float32')
    return tf.reduce_sum(less_one_image) / denom


def less_three_percentage(gt_depth, pred_depth, depth_interval):
    """
    less three accuracy for one batch
    :param gt_depth:
    :param pred_depth:
    :param depth_interval:
    :return:
    """
    with tf.variable_scope('less_three_percentage'):
        *_, h, w = pred_depth.get_shape().as_list()
        batch_size = tf.shape(pred_depth)[0]
        mask_true = tf.cast(tf.not_equal(gt_depth, 0.0), dtype='float32')
        denom = tf.reduce_sum(mask_true) + 1e-7
        interval_image = tf.tile(tf.reshape(depth_interval, [batch_size, 1, 1, 1]), [1, 1, h, w])
        abs_diff_image = tf.abs(gt_depth - pred_depth) / interval_image
        less_three_image = mask_true * tf.cast(tf.less_equal(abs_diff_image, 3.0), dtype='float32')
    return tf.reduce_sum(less_three_image) / denom

