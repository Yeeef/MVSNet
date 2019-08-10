import cv2
from matplotlib import pyplot as plt
import os
from os import path
import re
import numpy as np
import shutil


class LogFile(object):
    """ log parser, come from plane_estimate.py """
    def __init__(self, log_id, content_lines):
        self.log_id = log_id
        self.content_lines = content_lines
        self.num_lines = len(content_lines)
        self.num_polygon = 0
        self.polygons = []
        self._file_ptr = 0

    def _ptr_advance(self):
        self._file_ptr += 1

    def _ptr_advance_by(self, step):
        self._file_ptr += step

    def _is_eof(self):
        if self._file_ptr > self.num_lines:
            return True
        else:
            return False

    def _ptr_line_content(self):
        if not self._is_eof():
            return self.content_lines[self._file_ptr]
        else:
            raise IOError('{} log, ptr {} exceeds eof!'.format(self.log_id, self._file_ptr))

    def _parse_polygon_meta(self, meta_str):
        fields = meta_str.split(' ')
        polygon_id = fields[0].strip()
        num_points = int(fields[1].strip())
        return polygon_id, num_points

    def _parse_polygon_info(self, num_points):
        point_list = []
        for i in range(num_points):
            self._ptr_advance()
            line = self._ptr_line_content()
            fields = line.split(' ')
            x = round(float(fields[0].lower()))
            y = round(float(fields[1].lower()))
            point_list.append((x, y))

        return point_list

    def parse_content(self):
        meta = self.content_lines[0]
        self.num_polygon = int(meta.split(' ')[-1].strip())
        polygon_list = []
        try:
            for i in range(self.num_polygon):
                self._ptr_advance()
                polygon_meta = self._ptr_line_content()
                _, num_points = self._parse_polygon_meta(polygon_meta)
                point_list = self._parse_polygon_info(num_points)
                if len(point_list) > 2:
                    polygon_list.append(point_list)
        except IOError as ioe:
            print('Error: ', ioe)
            exit()

        return polygon_list

NEW_DIR_COLLECTIONS = [
    '/media/yeeef/Seagate Expansion Drive/training_data/selected_images_1_camera_polygon',
    '/media/yeeef/Seagate Expansion Drive/training_data/selected_images_2_camera_polygon',
    '/media/yeeef/Seagate Expansion Drive/training_data/selected_images_3_camera_polygon',
    '/media/yeeef/Seagate Expansion Drive/training_data/selected_images_4_camera_polygon',
    '/media/yeeef/Seagate Expansion Drive/training_data/selected_images_5_camera_polygon',
    '/media/yeeef/Seagate Expansion Drive/training_data/selected_images_6_camera_polygon',
    '/media/yeeef/Seagate Expansion Drive/training_data/selected_images_8_camera_polygon',

]
IDX = 4
BASE_DIR = NEW_DIR_COLLECTIONS[IDX]
LOG_PATTERN = re.compile(r'(\d+).log')
VALID_DIR_PATTERN = re.compile(r'\d+$')
DATASET_DIR = "/data3/lyf/MVSNET/mvsnet_test/standard_dataset/illumination_part{}_adaptives".format(IDX)
OUT_BASE_DIR = "/data3/lyf/MVSNET/mvsnet_test/results/illumination_part{}_adaptives".format(IDX)


def generate_label_one_scene(log_dir, out_dir):
    demo_files = list(os.scandir(log_dir))
    label_log_files = [file for file in demo_files if
                       path.splitext(file.name)[1] == '.log' and file.name != 'camera_params.log']

    for label_log in label_log_files[:]:
        match = re.search(LOG_PATTERN, label_log.name)
        log_id = int(match.group(1))
        content_lines = open(label_log.path).readlines()
        log_parser = LogFile(log_id, content_lines)
        polygon_list = log_parser.parse_content()
        img_path = path.join(path.dirname(label_log.path), '{}.png'.format(log_id))
        img = cv2.imread(img_path)
        label = np.zeros([img.shape[0], img.shape[1]])
        cv2.fillConvexPoly(label, np.array(polygon_list[0]), 255)
        cv2.imwrite(path.join(out_dir, '{}_label.png'.format(log_id)), label)


def rename_output(dataset_dir, result_dir):
    """
    align the folder name between mvsnet output and mvsnet input
    :param dataset_dir:
    :param result_dir:
    :return:
    """
    assert path.basename(dataset_dir) == path.basename(result_dir)
    new_result_dir = path.join(path.dirname(result_dir), "new_" + path.basename(result_dir))

    dataset_groups = sorted(os.listdir(dataset_dir), key=int)
    result_groups = os.listdir(result_dir)
    result_groups = [group for group in result_groups if 'log' not in group]
    result_groups = sorted(result_groups, key=int)
    assert len(result_groups) == len(dataset_groups)
    idx_mapping = dict(zip(result_groups, dataset_groups))
    print(idx_mapping)
    for group in result_groups:
        if group in idx_mapping:
            shutil.move(path.join(result_dir, group), path.join(new_result_dir, idx_mapping[group]))
    shutil.move(path.join(result_dir, 'log.log'), path.join(new_result_dir, 'log.log'))
    shutil.rmtree(result_dir)
    shutil.move(new_result_dir, result_dir)


if __name__ == "__main__":
    import tqdm
    rename_output(DATASET_DIR, OUT_BASE_DIR)
    group_dirs = list(os.scandir(BASE_DIR))
    for group_dir in tqdm.tqdm(group_dirs[:]):
        log_dir = path.join(group_dir.path, 'image', 'color_depth_log')
        match = re.search(VALID_DIR_PATTERN, group_dir.name)
        if match:
            group_id = int(match.group())
            # print(group_id)
            # print(log_dir)
            generate_label_one_scene(log_dir, path.join(OUT_BASE_DIR, str(group_id)))
