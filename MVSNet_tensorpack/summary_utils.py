from tensorpack import *


def add_image_summary(x, name=None, collections=None):
    if name is None:
        name = x.op.name
    ctx = get_current_tower_context()
    if ctx is not None and not ctx.is_main_training_tower:
        return
    with tf.device('/cpu:0'):
        tf.summary.image(name + '-image', x, collections=collections)