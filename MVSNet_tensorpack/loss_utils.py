from tensorpack import *


def mvsnet_regression_loss(gt_depth, pred_depth, depth_interval):

    with tf.variable_scope('mvsnet_loss'):
        masked_mae = non_zero_mean_absolute_diff(gt_depth, pred_depth, depth_interval)

        less_one_accuracy = less_one_percentage(gt_depth, pred_depth, depth_interval)

        less_three_accuracy = less_three_percentage(gt_depth, pred_depth, depth_interval)

    return masked_mae, less_one_accuracy, less_three_accuracy


def non_zero_mean_absolute_diff(gt_depth, pred_depth, depth_interval):
    """
    non zero mean absolute error(MAE)
    :param gt_depth:
    :param pred_depth:
    :param depth_interval:
    :return:
    """
    with tf.variable_scope('MAE'):
        batch_size, _, h, w = tf.shape(pred_depth)
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
        batch_size, _, h, w = tf.shape(pred_depth)
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
        batch_size, _, h, w = tf.shape(pred_depth)
        mask_true = tf.cast(tf.not_equal(gt_depth, 0.0), dtype='float32')
        denom = tf.reduce_sum(mask_true) + 1e-7
        interval_image = tf.tile(tf.reshape(depth_interval, [batch_size, 1, 1, 1]), [1, 1, h, w])
        abs_diff_image = tf.abs(gt_depth - pred_depth) / interval_image
        less_three_image = mask_true * tf.cast(tf.less_equal(abs_diff_image, 3.0), dtype='float32')
    return tf.reduce_sum(less_three_image) / denom

