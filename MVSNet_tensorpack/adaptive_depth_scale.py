import os
from os import path
import numpy as np
from test_utils import LogManager


def parse_obj_file(path):
    with open(path, 'r') as infile:
        obj_file_lines = infile.readlines()
    cors = []
    for obj_file_line in obj_file_lines:
        fields = obj_file_line.split()
        _, x, y, z, *_ = fields
        cors.append([float(x), float(y), float(z)])
    return np.array(cors)


def obtain_depths_with_R_T(R, T, cors):
    if cors.shape[1] == 3:
        cors = cors.T
    assert cors.shape[0] == 3, cors.shape
    _, num_cors = cors.shape
    camera_cors = np.matmul(R, cors) + np.tile(np.expand_dims(T, 1), [1, num_cors])
    depths = camera_cors[2, :]
    return depths


def obtain_depths_with_extrinsic(extrinsic, cors):
    R = extrinsic[:3, :3]
    T = extrinsic[:3, 3]
    return obtain_depths_with_R_T(R, T, cors)


def obtain_depths_with_parsed_dict(parsed_dict, cors):
    extrinsic = parsed_dict['extrinsic']
    return obtain_depths_with_extrinsic(extrinsic, cors)


def obtain_depths_with_log_path(log_path, cors):
    log_manager = LogManager(log_path)
    parsed_dict_list = log_manager.parse()
    depths_list = []
    for parsed_dict in parsed_dict_list:
        depths = obtain_depths_with_parsed_dict(parsed_dict, cors)
        depths_list.append(depths)
    return depths_list


def depth_statistics(depths):
    max_depth = max(depths)
    min_depth = min(depths)
    median_depth = np.median(depths)
    mean_depth = np.mean(depths)
    # percentile_4 = np.percentile(depths, 0.4)
    # percentile_6 = np.percentile()
    # print('max_depth: %f, min_depth: %f, mean_depth: %f, median_depth: %f' % (max_depth, min_depth, mean_depth, median_depth))
    quantile = [np.quantile(depths, per) for per in np.arange(0.05, 1, 0.05)]
    # print('percentile statistics: {}'.format(quantile))
    return quantile


if __name__ == "__main__":
    obj_path = '/data3/lyf/mvsnet_test/xuhantong20190503/selected_images2/selected_images/10real/image/' \
               'bundler_point_cloud.obj'
    log_path = '/data3/lyf/mvsnet_test/xuhantong20190503/selected_images2/selected_images/10real/image/' \
               'color_depth_log/camera_params.log'
    log_manager = LogManager(log_path)
    parsed_dict_list = log_manager.parse()
    cors = parse_obj_file(obj_path)
    print('original cors shape: {}'.format(cors.shape))
    cors = cors.T
    print('after transpose: {}'.format(cors.shape))

    for parsed_dict in parsed_dict_list:
        extrinsic = parsed_dict['extrinsic']
        R = extrinsic[:3, :3]
        T = extrinsic[:3, 3]
        # print(R.shape, R.dtype)
        # print(T.shape, T.dtype)
        # print(cors.dtype)
        _, num_cors = cors.shape
        camera_cors = np.matmul(R, cors) + np.tile(np.expand_dims(T, 1), [1, num_cors])
        print('camera_cors shape: {}'.format(cors.shape))
        depths = camera_cors[2, :]
        depth_statistics(depths)
