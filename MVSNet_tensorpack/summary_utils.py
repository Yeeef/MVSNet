from tensorpack import *
import tensorflow as tf

def add_image_summary(x, name=None, collections=None):
    if name is None:
        name = x.op.name
    ctx = get_current_tower_context()
    if ctx is not None and not ctx.is_main_training_tower:
        return
    with tf.device('/cpu:0'):
        tf.summary.image(name + '-image', x, collections=collections)

#
# def depth_tp_rgb(depth_image, rows, cols):
#     """
#
#     :param depth_image: 3d/2d depth image with shape h, w, 1 or h, w
#     :param rows:
#     :param cols:
#     :return:
#     """
#     max_depth = -1.0e-24
#     for i in range(rows * cols):
#         depth =


