# -*- coding: utf-8 -*-
# File: upsampling_layers.py

import tensorflow as tf
from tensorpack.models import layer_register
import numpy as np


@layer_register(log_shape=True)
def TFBilinearUpSample(x, shape, data_format='NCHW'):
    assert data_format in ['NHWC', 'NCHW']
    assert isinstance(shape, (int, list, tuple, tf.Tensor)), "shape must be an int or a list or a Tensor!"
    if data_format == 'NCHW':
        x = tf.transpose(x, [0, 2, 3, 1])

    if isinstance(shape, int):
        inp_shape = x.shape.as_list()
        out_shape = tf.constant([inp_shape[1] * shape, inp_shape[2] * shape], tf.int32)
    elif isinstance(shape, (list, tuple)):
        assert len(shape) == 2
        out_shape = tf.constant(shape, tf.int32)
    elif isinstance(shape, tf.Tensor):
        # out_shape = tf.shape(shape)[2:]
        out_shape = shape.get_shape().as_list()[2:]
    up = tf.image.resize_bilinear(x, out_shape, align_corners=True)

    if data_format == 'NCHW':
        up = tf.transpose(up, [0, 3, 1, 2])

    return up


@layer_register(log_shape=True)
def TFNearestUpSample(x, shape, data_format='channels_first'):
    assert data_format in ['channels_first', 'channels_last']
    assert isinstance(shape, (int, list, tuple, tf.Tensor)), "shape must be an int or a list or a Tensor!"
    if data_format == 'channels_first':
        x = tf.transpose(x, [0, 2, 3, 1])

    if isinstance(shape, int):
        inp_shape = x.shape.as_list()
        out_shape = tf.constant([inp_shape[1] * shape, inp_shape[2] * shape], tf.int32)
    elif isinstance(shape, (list, tuple)):
        assert len(shape) == 2
        out_shape = tf.constant(shape, tf.int32)
    elif isinstance(shape, tf.Tensor):
        # out_shape = tf.shape(shape)[2:]
        out_shape = shape.get_shape().as_list()[2:]
    up = tf.image.resize_nearest_neighbor(x, out_shape, align_corners=True)

    if data_format == 'channels_first':
        up = tf.transpose(up, [0, 3, 1, 2])

    return up


@layer_register(log_shape=True)
def CaffeBilinearUpSample(x, shape, data_format='NCHW'):
    """
    Deterministic bilinearly-upsample the input images.
    It is implemented by deconvolution with "BilinearFiller" in Caffe.
    It is aimed to mimic caffe behavior.
    Args:
        x (tf.Tensor): a tensor
        shape (int): the upsample factor
    Returns:
        tf.Tensor: a tensor.
    """
    assert data_format in ['NHWC', 'NCHW']
    if data_format == 'NCHW':
        x = tf.transpose(x, [0, 2, 3, 1])

    inp_shape = x.shape.as_list()
    ch = inp_shape[3]
    assert ch is not None

    shape = int(shape)
    filter_shape = 2 * shape

    def bilinear_conv_filler(s):
        """
        s: width, height of the conv filter
        https://github.com/BVLC/caffe/blob/99bd99795dcdf0b1d3086a8d67ab1782a8a08383/include/caffe/filler.hpp#L219-L268
        """
        f = np.ceil(float(s) / 2)
        c = float(2 * f - 1 - f % 2) / (2 * f)
        ret = np.zeros((s, s), dtype='float32')
        for x in range(s):
            for y in range(s):
                ret[x, y] = (1 - abs(x / f - c)) * (1 - abs(y / f - c))
        return ret

    w = bilinear_conv_filler(filter_shape)

    weight_var = tf.constant(w, tf.float32,
                             shape=(filter_shape, filter_shape, 1, 1),
                             name='bilinear_upsample_filter')
    x = tf.pad(x, [[0, 0], [shape - 1, shape - 1], [shape - 1, shape - 1], [0, 0]], mode='SYMMETRIC')
    x_chs = tf.split(x, num_or_size_splits=ch, axis=-1)
    out_shape = tf.shape(x_chs[0]) * tf.constant([1, shape, shape, 1], tf.int32)
    edge = shape * (shape - 1)

    deconv_chs = []
    for x_ch in x_chs:
        deconv = tf.nn.conv2d_transpose(x_ch, weight_var, out_shape,
                                        [1, shape, shape, 1], 'SAME')
        deconv_chs.append(deconv[:, edge:-edge, edge:-edge, :])
    deconv = tf.concat(deconv_chs, axis=-1)

    if inp_shape[1]:
        inp_shape[1] *= shape
    if inp_shape[2]:
        inp_shape[2] *= shape
    deconv.set_shape(inp_shape)

    if data_format == 'NCHW':
        deconv = tf.transpose(deconv, [0, 3, 1, 2])

    return deconv


@layer_register(log_shape=True)
def SeparableTransposedConvolutionAsUpsampling(x, shape, \
                                               channel_multiplier=4, data_format='NHWC', mimic_bilinear=False):
    """
    Separable (depth-wise) transposed convolution layer worked as an upsampling layer.
    Args:
        x (tf.Tensor): input tensor
        shape (int): The upsample factor
        channel_multiplier (int): channel multiplier while applying depth-wise convolution
    Returns:
        tf.Tensor: Upsampled tensor
    """
    assert data_format in ['NHWC', 'NCHW']
    if data_format == 'NCHW':
        x = tf.transpose(x, [0, 2, 3, 1])

    assert isinstance(channel_multiplier, int), "channel_multiplier must be an integer!"
    assert channel_multiplier > 0, 'channel_multiplier must be > 0!'

    inp_shape = x.shape.as_list()
    ch = inp_shape[3]
    assert ch is not None

    shape = int(shape)
    filter_shape = 2 * shape

    def bilinear_conv_filler(s, ch, multi):
        """
        s: width, height of the conv filter
        https://github.com/BVLC/caffe/blob/99bd99795dcdf0b1d3086a8d67ab1782a8a08383/include/caffe/filler.hpp#L219-L268
        """
        f = np.ceil(float(s) / 2)
        c = float(2 * f - 1 - f % 2) / (2 * f)
        ret = np.zeros((s, s), dtype='float32')
        for x in range(s):
            for y in range(s):
                ret[x, y] = (1 - abs(x / f - c)) * (1 - abs(y / f - c))
        ret = np.repeat(ret, ch * multi).reshape((filter_shape, filter_shape, multi, ch))
        return ret

    # depth-wise convolution weight init
    if mimic_bilinear:
        W_init = tf.constant_initializer(
            bilinear_conv_filler(filter_shape, ch, channel_multiplier))
    else:
        W_init = tf.contrib.layers.xavier_initializer()
    W = tf.get_variable('depth/W',
                        [filter_shape, filter_shape, channel_multiplier, ch],
                        dtype=x.dtype, initializer=W_init)
    W_chs = tf.split(W, num_or_size_splits=ch, axis=-1)

    # input tensor initialization
    x = tf.pad(x, [[0, 0], [shape - 1, shape - 1], [shape - 1, shape - 1], [0, 0]], mode='SYMMETRIC')
    x_chs = tf.split(x, num_or_size_splits=ch, axis=-1)
    out_shape = tf.shape(x_chs[0]) * tf.constant([1, shape, shape, channel_multiplier], tf.int32)
    edge = shape * (shape - 1)

    deconv_chs = []
    for x_ch, W_ch in zip(x_chs, W_chs):
        deconv = tf.nn.conv2d_transpose(x_ch, W_ch, out_shape,
                                        [1, shape, shape, 1], 'SAME')
        deconv_chs.append(deconv[:, edge:-edge, edge:-edge, :])
    deconv = tf.concat(deconv_chs, axis=-1)

    # point-wise 1x1 convolution
    pW = tf.get_variable('point/W', [1, 1, ch * channel_multiplier, ch],
                         initializer=tf.contrib.layers.xavier_initializer(),
                         dtype=x.dtype)
    pb = tf.get_variable('point/b', [ch], initializer=tf.zeros_initializer(), dtype=x.dtype)
    deconv = tf.nn.conv2d(deconv, pW, [1, 1, 1, 1], 'SAME')
    deconv = tf.nn.bias_add(deconv, pb, data_format='NHWC')

    if inp_shape[1]:
        inp_shape[1] *= shape
    if inp_shape[2]:
        inp_shape[2] *= shape
    deconv.set_shape(inp_shape)

    if data_format == 'NCHW':
        deconv = tf.transpose(deconv, [0, 3, 1, 2])

    return deconv