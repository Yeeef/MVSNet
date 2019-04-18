import tensorflow as tf
from tensorpack import *



class MVSNet(ModelDesc):

    height = 513
    width = 769

    def __init__(self):
        super(MVSNet, self).__init__()

    def inputs(self):
        return [
            tf.placeholder(tf.float32, [None, 5, self.height, self.width, 3], 'imgs'),
            tf.placeholder(tf.float32, [None, 5, self.height, self.width, 3], 'cams'),
            tf.placeholder(tf.float32, [None, self.height, self.width, 1], 'seg_map'),
            tf.placeholder(tf.float32, [None, self.height, self.width, 1], 'depth_map'),
        ]

    def build_graph(self, imgs, seg_map, depth_map):
        # preprocess

        # feature extraction

        # warping

        # cost volume

        # 2 branches: one is ASPP, the other is depth map refinement
        pass