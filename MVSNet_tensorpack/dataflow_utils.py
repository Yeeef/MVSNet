# -*- coding: utf-8 -*-
# File: data.py
# Author: Yeeef

import os
import numpy as np
from tensorpack import *
from DataManager import (Cam, PFMReader, mask_depth_image)
import cv2
from tensorpack.utils import logger
import math

__all__ = ['DTU']


def center_image(img):
    """ normalize image input """
    img = img.astype(np.float32)
    var = np.var(img, axis=(0, 1), keepdims=True)
    mean = np.mean(img, axis=(0, 1), keepdims=True)
    return (img - mean) / (np.sqrt(var) + 0.00000001)


def gen_test_input_sample_list(data_dir, view_num):
    """ mvs input path list """
    image_folder = os.path.join(data_dir, 'images')
    cam_folder = os.path.join(data_dir, 'cams')
    cluster_list_path = os.path.join(data_dir, 'pair.txt')
    cluster_list = open(cluster_list_path).read().split()

    # for each dataset
    mvs_list = []
    pos = 1
    for i in range(int(cluster_list[0])):
        paths = []
        # ref image
        ref_index = int(cluster_list[pos])
        pos += 1
        ref_image_path = os.path.join(image_folder, ('%08d.jpg' % ref_index))
        ref_cam_path = os.path.join(cam_folder, ('%08d_cam.txt' % ref_index))
        paths.append(ref_image_path)
        paths.append(ref_cam_path)
        # view images
        all_view_num = int(cluster_list[pos])
        pos += 1
        check_view_num = min(view_num - 1, all_view_num)
        for view in range(check_view_num):
            view_index = int(cluster_list[pos + 2 * view])
            view_image_path = os.path.join(image_folder, ('%08d.jpg' % view_index))
            view_cam_path = os.path.join(cam_folder, ('%08d_cam.txt' % view_index))
            paths.append(view_image_path)
            paths.append(view_cam_path)
        pos += 2 * all_view_num
        # depth path
        mvs_list.append(paths)
    return mvs_list


def scale_mvs_input(images, cams, depth_image=None, scale=1.):
    """ resize input to fit into the memory """
    print('-'*100)
    view_num = len(images)
    for view in range(view_num):

        images[view] = scale_image(images[view], x_scale=scale, y_scale=scale)
        cams[view] = scale_camera(cams[view], scale=scale)

    if depth_image is None:
        return images, cams
    else:
        depth_image = scale_image(depth_image, x_scale=scale, y_scale=scale, interpolation='nearest')
        return images, cams, depth_image


def scale_camera(cam, scale=1.):
    """ resize input in order to produce sampled depth map """
    new_cam = np.copy(cam)
    # focal:
    new_cam[1][0][0] = cam[1][0][0] * scale
    new_cam[1][1][1] = cam[1][1][1] * scale
    # principle point:
    new_cam[1][0][2] = cam[1][0][2] * scale
    new_cam[1][1][2] = cam[1][1][2] * scale
    return new_cam


def scale_image(image, x_scale=1., y_scale=1., interpolation='linear'):
    """ resize image using cv2 """
    if interpolation == 'linear':
        return cv2.resize(image, None, fx=x_scale, fy=y_scale, interpolation=cv2.INTER_LINEAR)
    if interpolation == 'nearest':
        return cv2.resize(image, None, fx=x_scale, fy=y_scale, interpolation=cv2.INTER_NEAREST)


def crop_mvs_input(images, cams, max_h, max_w, base_image_size=8, depth_image=None):
    """
    resize images and cameras to fit the network (can be divided by base image size)
    the input must be dividable with 8
    """

    view_num = len(images)
    # crop images and cameras
    for view in range(view_num):
        h, w, _ = images[view].shape
        new_h = h
        new_w = w
        if new_h > max_h:
            new_h = max_h
        else:
            new_h = int(math.ceil(h / base_image_size) * base_image_size)
        if new_w > max_w:
            new_w = max_w
        else:
            new_w = int(math.ceil(w / base_image_size) * base_image_size)
        print('h: {}, w:{}, new_h: {}, new_w: {}'.format(h, w, new_h, new_w))
        start_h = int(math.ceil((h - new_h) / 2))
        start_w = int(math.ceil((w - new_w) / 2))
        finish_h = start_h + new_h
        finish_w = start_w + new_w
        images[view] = images[view][start_h:finish_h, start_w:finish_w]
        cams[view][1][0][2] = cams[view][1][0][2] - start_w
        cams[view][1][1][2] = cams[view][1][1][2] - start_h

    # crop depth image
    if depth_image is not None:
        depth_image = depth_image[start_h:finish_h, start_w:finish_w]
        return images, cams, depth_image
    else:
        return images, cams


def scale_mvs_camera(cams, scale=1.):
    """ resize input in order to produce sampled depth map """
    view_num = len(cams)
    for view in range(view_num):
        cams[view] = scale_camera(cams[view], scale=scale)
    return cams


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
        self.sample_list, self.len_data = gen_dtu_resized_path(dtu_data_root, view_num, train_or_val)
        self.count = 0

    def __len__(self):
        # // fixed : check if it is true
        return self.len_data
        # if self.is_train:
        #     return 27097
        # else:
        #     return 882

    def __iter__(self):
        if self.shuffle is not None:
            self.rng.shuffle(self.sample_list)
        for data in self.sample_list:
            imgs = []
            cams = []
            for view in range(self.view_num):
                # read_image
                # // [fixedTODO]: center image is left to augmentor or tf Graph
                # // I have done it here
                img = cv2.imread(data[2 * view])
                # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
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
            if self.test and self.count % 10 == 0:
                print('Forward pass: d_min = %f, d_max = %f.' %
                      (depth_min, depth_min + (self.max_d - 1) * depth_interval))
            assert cams.shape == (self.view_num, 2, 4, 4)
            self.count += 1
            yield [imgs, cams, depth_image]

    @staticmethod
    def make_test_data(data_dir, view_num, max_h, max_w, max_d, interval_scale):
        """
        the data_dir should be organized like:
        * images
        * cams
        * pair.txt
        :param data_dir:
        :return:
        """
        dir_files = os.listdir(data_dir)
        assert 'images' in dir_files and 'cams' in dir_files and 'pair.txt' in dir_files
        sample_list = gen_test_input_sample_list(data_dir, view_num)
        for data in sample_list:
            imgs = []
            cams = []

            for view in range(view_num):
                # read_image
                # // [fixedTODO]: center image is left to augmentor or tf Graph
                # // I have done it here
                img = cv2.imread(data[2 * view])
                # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                # load cam and do basic interval_scale
                cam = Cam(data[2 * view + 1], max_d=max_d, interval_scale=interval_scale)
                imgs.append(img)
                cams.append(cam.get_mat_form())

            logger.info('range: {} {} {} {}'.format(cams[0][1, 3, 0], cams[0][1, 3, 1], cams[0][1, 3, 2], cams[0][1, 3, 3]))

            general_h_scale = -1.
            general_w_scale = -1.
            # 选取较大的scale的好处是，宁愿 crop 也不要 padding
            for view in range(view_num):
                h, w, _ = imgs[view]
                height_scale = float(max_h) / h
                width_scale = float(max_w) / w
                general_h_scale = height_scale if height_scale > general_h_scale else general_h_scale
                general_w_scale = width_scale if width_scale > general_w_scale else general_w_scale
                assert height_scale < 1 and width_scale < 1, 'max_h, max_w shall be less than h, w'
            resize_scale = general_h_scale if general_h_scale > general_w_scale else general_w_scale
            logger.info('resize scale is %.2f' % resize_scale)

            # first scale
            imgs, cams = scale_mvs_input(imgs, cams, scale=resize_scale)

            # then crop to fit the nn input
            imgs, cams = crop_mvs_input(imgs, cams, max_h, max_w, base_image_size=8)

            # then scale the cam and img, because the final resolution is not full-res
            imgs, cams = scale_mvs_input(imgs, cams, scale=0.25)

            ref_cam = cams[0]
            depth_min, depth_interval, depth_max = Cam.get_depth_meta(ref_cam, 'depth_min', 'depth_interval', 'depth_max')
            # view_num, h, w, 3
            imgs = np.array(imgs)
            # (view_num, )
            cams = np.array(cams)
            logger.info('d_min = %f, interval: %f, d_max = %f.' %
                      (depth_min, depth_interval, depth_max))

            assert cams.shape == (view_num, 2, 4, 4)
            yield imgs, cams


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
    training_set = training_set[:]
    validation_set = [3, 5, 17, 21, 28, 35, 37, 38, 40, 43, 56, 59, 66, 67, 82, 86, 106, 117]
    validation_set = validation_set[:1]
    logger.warn('num of scans in training_set: {}'.format(len(training_set)))
    logger.warn('num of scans in val_set: {}'.format(len(validation_set)))

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
                    # ref image' index
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

    return sample_list, len(data_set) * 343


from matplotlib import pyplot as plt


if __name__ == "__main__":
    # ds = GlassData('../data', 'train')
    # print(len(ds)) # 1200
    # ds_val = GlassData('../data', 'val')
    # print(len(ds_val)) # 24
    import multiprocessing
    DTU_DATA_ROOT = '/home/yeeef/Desktop/dtu_training'
    """dataflow testing"""
    ds = DTU(DTU_DATA_ROOT, 3, 'train', 1.06, 192)
    ds = BatchData(ds, 2, remainder=True)
    count = 0
    print(ds.size())
    # for imgs, cams, depth_image in ds:
    #     # print(point)
    #     if count == 1420:
    #         plt.figure()
    #         print(imgs[0].shape)
    #
    #         plt.subplot(1, 3, 1)
    #         plt.imshow(imgs[0].astype('uint8'))
    #         plt.subplot(1, 3, 2)
    #         plt.imshow(imgs[1].astype('uint8'))
    #         plt.subplot(1, 3, 3)
    #         plt.imshow(imgs[2].astype('uint8'))
    #         plt.show()
    #         break
    #         # for cam in cams:
    #         #     depth_min, depth_interval, depth_num, depth_max = Cam.get_depth_meta(cam, 'depth_min', 'depth_interval',
    #         #                                                                                            'depth_num',
    #         #                                                                                             'depth_max')
    #         #     depth_max = depth_min + (depth_num - 1) * depth_interval
    #         #     print('{} {} {} {}'.format(depth_min, depth_interval, depth_num, depth_max))
    #     count += 1
        # if count >= 410:
        #     break
    # parallel = min(40, multiprocessing.cpu_count() // 2)  # assuming hyperthreading
    # # if parallel < 16:
    # #     logger.warn("DataFlow may become the bottleneck when too few processes are used.")
    # print(parallel)
    # # ds1 = MultiThreadMapData(
    # #       ds, nr_thread=parallel,
    # #       map_func=lambda dp: dp,
    # #       buffer_size=1000)
    # # ds = PrefetchData(ds, 4, 16)
    #
    # ds = PrefetchDataZMQ(ds, nr_proc=16)
    # ds = BatchData(ds, 2)
    # ds = PrintData(ds)

    # 160it/s
    # TestDataSpeed(ds).start()
