import numpy as np
import cv2
import re
from tensorpack.utils import logger
import tensorflow as tf


def mask_depth_image(depth_image, min_depth, max_depth):
    """ mask out-of-range pixel to zero """
    # print ('mask min max', min_depth, max_depth)
    ret, depth_image = cv2.threshold(depth_image, min_depth, 100000, cv2.THRESH_TOZERO)
    ret, depth_image = cv2.threshold(depth_image, max_depth, 100000, cv2.THRESH_TOZERO_INV)
    depth_image = np.expand_dims(depth_image, 2)
    return depth_image


class Cam(object):

    def __init__(self, file_name=None, max_d=None, interval_scale=1):
        # self.file_name = file_name

        if file_name is not None:
            if max_d is not None:
                self.max_d = max_d
            self.interval_scale = interval_scale
            self.extrinsic_mat = None
            self.intrinsic_mat = None
            self.depth_min = None
            self.depth_max = None
            self.depth_num = None
            self.depth_interval = None
            self.K = None
            self._load_cam_from_file(file_name)
            self.R, self.T = self._get_R_and_T()
        else:
            # just for speed test
            self.interval_scale = interval_scale
            self.extrinsic_mat = 1
            self.intrinsic_mat = 1
            self.depth_min = 1
            self.depth_max = 1
            self.depth_num = 1
            self.depth_interval = 1

    # def get_list_form(self):

    @staticmethod
    def get_depth_meta(cam_mat, *queries):
        assert isinstance(cam_mat, (np.ndarray, tf.Tensor)), type(cam_mat)
        responses = []
        for query in queries:
            if query == 'depth_min':
                responses.append(cam_mat[1, 3, 0])
            elif query == 'depth_interval':
                responses.append(cam_mat[1, 3, 1])
            elif query == 'depth_num':
                responses.append(cam_mat[1, 3, 2])
            elif query == 'depth_max':
                responses.append(cam_mat[1, 3, 3])
            elif query == 'extrinsic':
                responses.append(cam_mat[0])
            elif query == 'intrinsic':
                responses.append(cam_mat[1, :3, :3])
            elif query == 'R':
                responses.append(cam_mat[0, :3, :3])
            elif query == 'T':
                responses.append(cam_mat[0, :3, 3])
            else:
                logger.warn('unknown query: {}'.format(query))
                exit()

        return responses

    def _get_R_and_T(self):
        assert self.extrinsic_mat is not None, 'extrinsic matrix is None'

        R = self.extrinsic_mat[0:3, 0:3]
        T = self.extrinsic_mat[0:3, 3]
        assert R.shape == (3, 3), 'rotation matrix with shape: {}'.format(R.shape)
        assert T.shape == (3,), 'translation vector with shape: {}'.format(T.shape)

        return R, T

    def get_K(self):
        assert self.intrinsic_mat is not None, 'intrinsic matrix is None'
        assert self.intrinsic_mat.shape == (3, 3), 'intrinsic matrix with shape: {}'.format(self.intrinsic_mat.shape)

        return self.intrinsic_mat

    def get_mat_form(self):
        ret = np.zeros((2, 4, 4))
        ret[0] = self.extrinsic_mat
        ret[1, :3, :3] = self.intrinsic_mat
        ret[1, 3, 0] = self.depth_min
        ret[1, 3, 1] = self.depth_interval
        ret[1, 3, 2] = self.depth_num
        ret[1, 3, 3] = self.depth_max

        return ret

    def _load_cam_from_file(self, file_name):

        with open(file_name, 'r') as cam_file:
            words = cam_file.read().split()
            self.extrinsic_mat = np.zeros((4, 4), dtype=np.float32)
            self.intrinsic_mat = np.zeros((3, 3), dtype=np.float32)
            for i in range(0, 4):
                for j in range(0, 4):
                    extrinsic_index = 4 * i + j + 1
                    self.extrinsic_mat[i, j] = words[extrinsic_index]

            for i in range(0, 3):
                for j in range(0, 3):
                    intrinsic_index = 3 * i + j + 18
                    self.intrinsic_mat[i, j] = words[intrinsic_index]

            if len(words) == 29:
                self.depth_min = float(words[27])
                self.depth_interval = float(words[28]) * self.interval_scale
                assert self.max_d is not None, 'max_d should not be None when DEPTH_NUM is not provided'
                self.depth_num = self.max_d
                self.depth_max = self.depth_min + self.depth_interval * self.depth_num
                # print(self.depth_max, self.depth_min, self.depth_interval, self.depth_num)
            elif len(words) == 30:
                self.depth_min = float(words[27])
                self.depth_interval = float(words[28]) * self.interval_scale
                self.depth_num = round(float(words[29]))
                self.depth_max = self.depth_min + self.depth_interval * self.depth_num
            elif len(words) == 31:
                self.depth_min = float(words[27])
                self.depth_interval = float(words[28]) * self.interval_scale
                self.depth_num = round(float(words[29]))
                self.depth_max = float(words[30])
            else:
                self.depth_min = 0
                self.depth_interval = 0
                self.depth_num = 0
                self.depth_max = 0


class PFMReader(object):
    def __init__(self, file_name):
        self.file_name = file_name
        self.data = self._load_pfm_file()

    def _load_pfm_file(self):
        file = open(self.file_name, 'rb')
        header = str(file.readline(), encoding='utf-8').strip('\n ')
        # print(header)

        if header == 'PF':
            color = True
        elif header == 'Pf':
            color = False
        else:
            raise Exception('Not a PFM file.')
        dim_match = re.match(r'^(\d+)\s(\d+)\s$', str(file.readline(), encoding='utf-8'))
        if dim_match:
            width, height = map(int, dim_match.groups())
        else:
            raise Exception('Malformed PFM header.')
        # scale = float(file.readline().rstrip())
        scale = float((file.readline()).rstrip())
        if scale < 0:  # little-endian
            data_type = '<f'
        else:
            data_type = '>f'  # big-endian
        data_string = file.read()
        data = np.fromstring(data_string, data_type)
        # data = np.fromfile(file, data_type)
        shape = (height, width, 3) if color else (height, width)
        data = np.reshape(data, shape)
        data = cv2.flip(data, 0)
        file.close()

        return data
