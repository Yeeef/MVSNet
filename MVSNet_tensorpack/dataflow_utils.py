# -*- coding: utf-8 -*-
# File: data.py
# Author: Yeeef

import os
import numpy as np
from tensorpack import *
from DataManager import (Cam, PFMReader, mask_depth_image)
import cv2

__all__ = ['DTU']


class DTU(RNGDataFlow):
    """
    produces [imgs, cams, depth_image]
    imgs is of shape (view_num, h, w, 3)
    cams is of shape (view_num, ) where the element is of type 'Cam'
    depth_image is of shape (h, w, 1)
    """

    test = False

    def __init__(self, dtu_data_root, view_num, train_or_val, interval_scale, max_d, shuffle=None):

        assert train_or_val in ['train', 'val'], 'train or val but '.format(train_or_val)
        assert isinstance(view_num, int), 'view_num ought to be of type int'
        assert isinstance(interval_scale, (int, float)), 'interval_scale ought to be a number'

        self.max_d = max_d
        self.interval_scale = interval_scale
        self.is_train = (train_or_val == 'train')
        self.train_or_val = train_or_val
        self.dtu_data_root = dtu_data_root
        if shuffle is None:
            shuffle = (train_or_val == 'train')
        self.shuffle = shuffle
        self.view_num = view_num
        self.sample_list = gen_dtu_resized_path(dtu_data_root, view_num, train_or_val)
        self.count = 0

    def __len__(self):
        # // TODO: check if it is true
        if self.is_train:
            return 27097
        else:
            return 882

    def __iter__(self):
        for data in self.sample_list:
            imgs = []
            cams = []
            for view in range(self.view_num):
                # read_image
                # // TODO: center image is left to augmentor or tf Graph
                img = cv2.imread(data[2 * view])
                # load cam and do basic interval_scale
                cam = Cam(data[2 * view + 1], max_d=self.max_d)
                cam.depth_interval = cam.depth_interval * self.interval_scale
                imgs.append(img)
                cams.append(cam.get_mat_form())
            # load depth image of ref view
            depth_image = PFMReader(data[2 * self.view_num]).data
            # depth_image = np.zeros((10, 10))

            # mask invalid depth_image
            ref_cam = cams[0]
            depth_min, depth_interval = Cam.get_depth_meta(ref_cam, 'depth_min', 'depth_interval')
            depth_start = depth_min + depth_interval
            depth_end = depth_min + (self.max_d - 2) * depth_interval
            # depth_image's shape: (h, w, 1)
            depth_image = mask_depth_image(depth_image, depth_start, depth_end)
            # view_num, h, w, 3
            imgs = np.array(imgs)
            # (view_num, )
            cams = np.array(cams)
            if self.test and self.count % 100 == 0:
                print('Forward pass: d_min = %f, d_max = %f.' %
                      (depth_min, depth_min + (self.max_d - 1) * depth_interval))
            assert cams.shape == (self.view_num, 2, 4, 4)
            self.count += 1
            yield [imgs, cams, depth_image]


def gen_dtu_resized_path(dtu_data_folder, view_num, mode='train'):
    """ generate data paths for dtu dataset """

    assert mode in ['train', 'val'], 'undefined mode: {}'.format(mode)
    assert isinstance(view_num, int), 'view_num ought to be of type int'
    sample_list = []

    # parse camera pairs
    cluster_file_path = os.path.join(dtu_data_folder, 'Cameras/pair.txt')

    with open(cluster_file_path, mode='r') as cluster_file:
        cluster_list = cluster_file.read().split()
    # cluster_list = file_io.FileIO(cluster_file_path, mode='r').read().split()

    # 3 sets
    training_set = [2, 6, 7, 8, 14, 16, 18, 19, 20, 22, 30, 31, 36, 39, 41, 42, 44,
                    45, 46, 47, 50, 51, 52, 53, 55, 57, 58, 60, 61, 63, 64, 65, 68, 69, 70, 71, 72,
                    74, 76, 83, 84, 85, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100,
                    101, 102, 103, 104, 105, 107, 108, 109, 111, 112, 113, 115, 116, 119, 120,
                    121, 122, 123, 124, 125, 126, 127, 128]
    validation_set = [3, 5, 17, 21, 28, 35, 37, 38, 40, 43, 56, 59, 66, 67, 82, 86, 106, 117]

    data_set = []
    if mode == 'train':
        data_set = training_set
    elif mode == 'val':
        data_set = validation_set

    # for each dataset
    for i in data_set:

        image_folder = os.path.join(dtu_data_folder, ('Rectified/scan%d_train' % i))
        cam_folder = os.path.join(dtu_data_folder, 'Cameras/train')
        depth_folder = os.path.join(dtu_data_folder, ('Depths/scan%d_train' % i))

        if mode == 'train':
            # for each lighting
            for j in range(0, 7):
                # for each reference image
                for p in range(0, int(cluster_list[0])):
                    paths = []
                    # ref image
                    ref_index = int(cluster_list[22 * p + 1])
                    ref_image_path = os.path.join(
                        image_folder, ('rect_%03d_%d_r5000.png' % ((ref_index + 1), j)))
                    ref_cam_path = os.path.join(cam_folder, ('%08d_cam.txt' % ref_index))
                    paths.append(ref_image_path)
                    paths.append(ref_cam_path)
                    # view images
                    for view in range(view_num - 1):
                        view_index = int(cluster_list[22 * p + 2 * view + 3])
                        view_image_path = os.path.join(
                            image_folder, ('rect_%03d_%d_r5000.png' % ((view_index + 1), j)))
                        view_cam_path = os.path.join(cam_folder, ('%08d_cam.txt' % view_index))
                        paths.append(view_image_path)
                        paths.append(view_cam_path)
                    # depth path
                    depth_image_path = os.path.join(depth_folder, ('depth_map_%04d.pfm' % ref_index))
                    paths.append(depth_image_path)
                    sample_list.append(paths)
        elif mode == 'val':
            j = 3
            # for each reference image
            for p in range(0, int(cluster_list[0])):
                paths = []
                # ref image
                ref_index = int(cluster_list[22 * p + 1])
                ref_image_path = os.path.join(
                    image_folder, ('rect_%03d_%d_r5000.png' % ((ref_index + 1), j)))
                ref_cam_path = os.path.join(cam_folder, ('%08d_cam.txt' % ref_index))
                paths.append(ref_image_path)
                paths.append(ref_cam_path)
                # view images
                for view in range(view_num - 1):
                    view_index = int(cluster_list[22 * p + 2 * view + 3])
                    view_image_path = os.path.join(
                        image_folder, ('rect_%03d_%d_r5000.png' % ((view_index + 1), j)))
                    view_cam_path = os.path.join(cam_folder, ('%08d_cam.txt' % view_index))
                    paths.append(view_image_path)
                    paths.append(view_cam_path)
                # depth path
                depth_image_path = os.path.join(depth_folder, ('depth_map_%04d.pfm' % ref_index))
                paths.append(depth_image_path)
                sample_list.append(paths)

    return sample_list


if __name__ == "__main__":
    # ds = GlassData('../data', 'train')
    # print(len(ds)) # 1200
    # ds_val = GlassData('../data', 'val')
    # print(len(ds_val)) # 24
    import multiprocessing
    DTU_DATA_ROOT = '/home/yeeef/Desktop/mvsnet/training_data/dtu_training'
    """dataflow testing"""
    ds = DTU(DTU_DATA_ROOT, 3, 'train', 1.06, 192)
    parallel = min(40, multiprocessing.cpu_count() // 2)  # assuming hyperthreading
    # if parallel < 16:
    #     logger.warn("DataFlow may become the bottleneck when too few processes are used.")
    print(parallel)
    # ds1 = MultiThreadMapData(
    #       ds, nr_thread=parallel,
    #       map_func=lambda dp: dp,
    #       buffer_size=1000)
    ds = PrefetchData(ds, 4, 16)
    ds = BatchData(ds, 1)
    # ds = PrintData(ds)
    # ds = PrefetchDataZMQ(ds, nr_proc=1)
    
    # 160it/s
    TestDataSpeed(ds).start()
