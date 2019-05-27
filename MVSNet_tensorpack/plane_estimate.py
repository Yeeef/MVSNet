# -*- coding: utf-8 -*-
# File: plane_estimate.py

import re
import numpy as np
from os import path
import os
from matplotlib import pyplot as plt
import cv2
from test_utils import PointCloudGenerator, LogManager
from DataManager import Cam

""" data structs """


class Point(object):

    def __init__(self, x, y):
        self.x_ = x
        self.y_ = y
        self.x = int(np.clip(x, 0, 5999))
        self.y = int(np.clip(y, 0, 3999))
        self.pt = (self.x, self.y)

    def __iter__(self):
        yield self.x
        yield self.y

    def __str__(self):
        return 'Point: ({}, {})'.format(self.x, self.y)


class Polygon(object):

    def __init__(self, point_list):
        assert len(point_list) > 0, len(point_list)
        self.num_points = len(point_list)
        self.points = point_list
        self.boundaries = []
        self._construct_boundaries()

    def __str__(self):
        info = "Polygon with %d corners\n" % len(self.points)
        for point in self.points:
            x, y = point.pt
            info += 'x: {}, y: {}\n'.format(x, y)
        return info

    def print_info(self):
        print('=' * 70)
        print(self.__str__())

    def surround_box(self):
        min_x = 100000
        max_x = -1
        min_y = 100000
        max_y = -1

        for point in self.points:
            x, y = point.pt
            if x < min_x:
                min_x = x
            elif x > max_x:
                max_x = x

            if y < min_y:
                min_y = y
            elif y > max_y:
                max_y = y

        return min_x, max_x, min_y, max_y

    def _construct_boundaries(self):
        """ return the boundaries of a polygon """
        point_list = self.points
        mask = [False] * len(point_list)
        start_point = point_list[0]
        mask[0] = True
        idx_order = [0]

        next_idx = 1
        while sum(mask) != len(point_list):
            min_tan = 1e5
            for idx in range(1, len(point_list)):
                if mask[idx]:
                    continue
                next_point = point_list[idx]
                tan, *_ = calc_tan(start_point, next_point)
                if tan < min_tan:
                    min_tan = tan
                    next_idx = idx
            idx_order.append(next_idx)
            mask[next_idx] = True
        for idx in idx_order:
            p1 = point_list[idx]
            p2 = point_list[(idx + 1) % len(idx_order)]
            _, y_diff, x_diff = calc_tan(p1, p2)
            line = Line(y_diff, -x_diff, p1, p2)
            self.boundaries.append(line)


class Line(object):
    """
    ax + by + c = 0
    ax + by - ax0 - by0 = 0
    ax + by - a(x1 + x2) / 2 - b(y1 + y2) / 2= 0

    """
    def __init__(self, a, b, p1, p2):
        """

        :param a:
        :param b:
        :param p1: first boundary
        :param p2: second boundary
        """
        self.a, self.b = a, b
        self.p1 = p1
        self.p2 = p2
        self.c = -np.dot([a, b], (np.array(p1.pt) + np.array(p2.pt)) / 2)

    def __str__(self):
        info = 'Line: {}x + {}y + {} = 0'.format(self.a, self.b, self.c)
        return info

    def get_y_given_x(self, x):
        assert self.b != 0, 'infinite y values'
        val = (-self.c - self.a * x) / self.b
        return val

    def get_x_given_y(self, y):
        assert self.a != 0, 'infinite x values'
        val = (-self.c - self.b * y) / self.a
        return val


class LogFile(object):

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
            point_list.append(Point(x, y))

        polygon = Polygon(point_list)
        return polygon

    def parse_content(self):
        meta = self.content_lines[0]
        self.num_polygon = int(meta.split(' ')[-1].strip())
        polygon_list = []
        try:
            for i in range(self.num_polygon):
                self._ptr_advance()
                polygon_meta = self._ptr_line_content()
                _, num_points = self._parse_polygon_meta(polygon_meta)
                polygon = self._parse_polygon_info(num_points)
                polygon_list.append(polygon)
        except IOError as ioe:
            print('Error: ', ioe)
            exit()

        return polygon_list


""" utility funcs """


def find_and_parse_log_file(dir_path, log_pattern):
    all_files = os.listdir(test_log_dir)
    log_files = []

    for file in all_files:
        match = re.search(log_pattern, file)
        if not match:
            continue
        if match:
            img_id = int(match.group(1))
            log_files.append(file)

    assert len(log_files) == 1, (dir_path, log_files)
    log_file = log_files[0]
    abs_log_path = path.join(dir_path, log_file)

    with open(abs_log_path, 'r') as infile:
        content_lines = infile.readlines()
    parser = LogFile(img_id, content_lines)
    polygons = parser.parse_content()
    for polygon in polygons:
        polygon.print_info()
    return (img_id, polygons)


def calc_tan(p1, p2):
    """
    calculate the tan of point p1 and point p2
    """
    x1, y1 = p1
    x2, y2 = p2

    if x1 == x2:
        tan = 1e4
    else:
        tan = (y2 - y1) / (x2 - x1)
    return tan, y2 - y1, x2 - x1


def min_and_max(a, b):
    if a > b:
        return b, a
    else:
        return a, b


def extract_valid_depth_from_line(line, depth_map):
    p1, p2 = line.p1, line.p2
    x1, y1 = p1
    x2, y2 = p2
    xmin, xmax = min_and_max(x1, x2)
    ymin, ymax = min_and_max(y1, y2)
    valid_depths = []
    for y in range(ymin, ymax + 1):
        x_val = line.get_x_given_y(y)
        if int(x_val) == x_val:
            left_x = int(x_val) - 1
            right_x = int(x_val) + 1
        else:
            left_x = int(x_val)
            right_x = int(x_val) + 1
        depth_left = depth_map[y, left_x]
        depth_right = depth_map[y, right_x]
        if depth_left != 0:
            valid_depths.append((left_x, y, depth_left))
        if depth_right != 0:
            valid_depths.append((right_x, y, depth_right))

    for x in range(xmin, xmax + 1):
        y_val = line.get_y_given_x(x)
        if int(y_val) == y_val:
            up_y = int(y_val) - 1
            bottom_y = int(y_val) + 1
        else:
            up_y = int(y_val)
            bottom_y = int(y_val) + 1
        depth_up, depth_bottom = depth_map[up_y, x], depth_map[bottom_y, x]
        if depth_up != 0:
            valid_depths.append((x, up_y, depth_up))
        if depth_bottom != 0:
            valid_depths.append((x, bottom_y, depth_bottom))
    return valid_depths


if __name__ == "__main__":
    test_log_dir = '/home/yeeef/Desktop/part1/0'
    LOG_PATTERN = re.compile(r'([0-9]{1,2})_fused_rgb\.log')
    img_id, polygons = find_and_parse_log_file(test_log_dir, LOG_PATTERN)
    img = cv2.imread(path.join(test_log_dir, '{}_rgb.png'.format(img_id)))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cam = Cam(path.join(test_log_dir, '{}_cam.txt'.format(img_id)))
    intrinsic = cam.intrinsic_mat
    plt.imshow(img)
    for polygon in polygons:
        point_list = polygon.points
        for point in point_list:
            x, y = point.pt
            plt.scatter(x, y)
    # for polygon in polygons:
    #     boundaries = polygon.boundaries
    #
    #     for line in boundaries:
    #         print(line)
    #         p1, p2 = line.p1, line.p2
    #         x1, y1 = p1
    #         x2, y2 = p2
    #         plt.plot([x1, x2], [y1, y2])

    demo_polygon = polygons[10]
    boundaries = demo_polygon.boundaries

    quality_depth_path = path.join(test_log_dir, '{}_depth_quality.exr'.format(img_id))
    quality_depth = cv2.imread(quality_depth_path, cv2.IMREAD_UNCHANGED)
    assert quality_depth.shape[:2] == img.shape[:2], (img.shape, quality_depth.shape)
    threeD_point_list = []
    mask = np.zeros_like(quality_depth)
    valid_xs = []
    valid_ys = []
    for line in boundaries:
        print(line.p1, line.p2)
        print(line)
        p1, p2 = line.p1, line.p2
        x1, y1 = p1
        x2, y2 = p2
        plt.plot([x1, x2], [y1, y2])
        valid_depths = extract_valid_depth_from_line(line, quality_depth)
        valid_xs.extend([item[0] for item in valid_depths])
        valid_ys.extend([item[1] for item in valid_depths])
        mask[valid_ys, valid_xs] = 1
        print(len(valid_depths), valid_depths)

    plt.show()
    threeD_point_list = (PointCloudGenerator.gen_3d_point_with_rgb_v2(quality_depth, img, intrinsic, zip(valid_xs, valid_ys)))

    PointCloudGenerator.write_as_obj(threeD_point_list, 'test.obj')

