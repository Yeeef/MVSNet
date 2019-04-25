# @layer_register(use_scope=None)
# def BatchNorm3D(
#         inputs, axis=None, training=None, momentum=0.99, epsilon=1e-5,
#         center=True, scale=True,
#         beta_initializer=tf.zeros_initializer(),
#         gamma_initializer=tf.ones_initializer(),
#         data_format='channels_last',
#         trainable=True,
#         name=None
#
# ):
#     shape = inputs.get_shape().as_list()
#     ndims = len(shape)
#     assert ndims == 5, 'bn3d only accept 5d tensor, not {}'.format(ndims)
#     if axis is None:
#         axis = 1 if data_format == 'channels_first' else 4
#     assert axis in (1, 4), axis
#
#     # parse training/ctx
#     ctx = get_current_tower_context()
#     if training is None:
#         training = ctx.is_training
#     training = bool(training)
#     freeze_bn_backward = not training and ctx.is_training
#     TF_version = get_tf_version_tuple()
#     if freeze_bn_backward:
#         assert TF_version >= (1, 4), \
#             "Fine tuning a BatchNorm model with fixed statistics needs TF>=1.4!"
#         if ctx.is_main_training_tower:  # only warn in first tower
#             logger.warn("[BatchNorm] Using moving_mean/moving_variance in training.")
#         # Using moving_mean/moving_variance in training, which means we
#         # loaded a pre-trained BN and only fine-tuning the affine part.
#     if not (training and ctx.is_training):
#         coll_bk = backup_collection([tf.GraphKeys.UPDATE_OPS])
#         with rename_tflayer_get_variable():
#             tf_args = dict(
#                 axis=axis,
#                 momentum=momentum, epsilon=epsilon,
#                 center=center, scale=scale,
#                 beta_initializer=beta_initializer,
#                 gamma_initializer=gamma_initializer,
#                 # https://github.com/tensorflow/tensorflow/issues/10857#issuecomment-410185429
#                 fused=(ndims == 4 and axis in [1, 3] and not freeze_bn_backward),
#                 _reuse=tf.get_variable_scope().reuse)
#
# @layer_register(use_scope=None)
# def BNReLU3D(x, name=None):
#     """
#     A shorthand of BatchNormalization + ReLU.
#     """
#     with rename_tflayer_get_variable():
#         x = tf.layers.batch_normalization(x, reuse=None, )
#         x = tf.nn.relu(x, name=name)
#     return x