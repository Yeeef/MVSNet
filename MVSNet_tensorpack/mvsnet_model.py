import tensorflow as tf
from tensorpack import *
from nn_utils import *


class MVSNet(ModelDesc):

    height = 513
    width = 769
    view_num = 3
    data_format = 'channels_first'

    def __init__(self, interval_scale, depth_num):
        super(MVSNet, self).__init__()
        self.interval_scale = interval_scale
        self.depth_num = depth_num

    def inputs(self):
        return [
            tf.placeholder(tf.float32, [None, self.view_num, self.height, self.width, 3], 'imgs'),
            tf.placeholder(tf.float32, [None, self.view_num], 'cams'),
            # tf.placeholder(tf.float32, [None, self.height, self.width, 1], 'seg_map'),
            tf.placeholder(tf.float32, [None, self.height, self.width, 1], 'depth_map'),
        ]

    def _preprocess(self, imgs, depth_map):
        # transpose image
        # center image
        pass

    def cost_volume(self, l):
        return l

    def depth_refinement(self, l):
        return l

    def build_graph(self, imgs, cams, depth_map):
        # preprocess
        self._preprocess(imgs, depth_map)

        ref_img = imgs[:, 0]
        ref_cam = cams[:, 0]

        # define a general argscope first, like data_format
        with argscope([tf.layers.conv3d, tf.layers.conv3d_transpose, Conv2D, MaxPooling, AvgPooling, BatchNorm],
                      data_format=self.data_format, padding='same'):
            # feature extraction
            # shape: b, view_num, c, h, w
            feature_maps = feature_extraction_net(imgs)

            # warping layer
            cost_volume = warping_layer(feature_maps, cams, self.depth_num)

            # cost volume regularization
            probability_volume = cost_volume_regularization(cost_volume)

            # depth_refinement
            logits = self.depth_refinement(probability_volume)
        # 2 branches: one is ASPP, the other is depth map refinement

        return logits
