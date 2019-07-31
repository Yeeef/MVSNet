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
import shutil

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


class Plane(object):
    def __init__(self, a, b, c, d):
        """
        ax + by + cz + d = 0
        """
        self.a, self.b, self.c, self.d = a, b, c, d

    def norm(self):
        return np.array([self.a, self.b, self.c])

    def offset(self):
        return self.d

    def get_z_given_xy(self, x, y):
        assert self.c != 0
        return -(self.d + self.a * x + self.b * y) / self.c

    def __str__(self):
        return 'Plane: %f * x + %f * y + %f * z + %f = 0' % (self.a, self.b, self.c, self.d)


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
                if len(polygon.points) > 2:
                    polygon_list.append(polygon)
        except IOError as ioe:
            print('Error: ', ioe)
            exit()

        return polygon_list


""" utility funcs """


def find_and_parse_log_file(dir_path, log_pattern):
    all_files = os.listdir(dir_path)
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
    if ymin != ymax:
        for y in range(ymin, ymax + 1):
            x_val = line.get_x_given_y(y)
            if int(x_val) == x_val:
                middle_x = int(x_val)
                left_x = int(x_val) - 1
                right_x = int(x_val) + 1
                depth_middle = depth_map[y, middle_x]
            else:
                left_x = int(x_val)
                right_x = int(x_val) + 1
                depth_middle = 0
        
            depth_left = depth_map[y, left_x]
            depth_right = depth_map[y, right_x]
            if depth_middle != 0:
                valid_depths.append((middle_x, y, depth_middle))
            if depth_left != 0:
                valid_depths.append((left_x, y, depth_left))
            if depth_right != 0:
                valid_depths.append((right_x, y, depth_right))
    if xmin != xmax:
        for x in range(xmin, xmax + 1):
            y_val = line.get_y_given_x(x)
            if int(y_val) == y_val:
                middle_y = int(y_val)
                up_y = int(y_val) - 1
                bottom_y = int(y_val) + 1
                depth_middle = depth_map[middle_y, x]
            else:
                up_y = int(y_val)
                bottom_y = int(y_val) + 1
                depth_middle = 0
            depth_up, depth_bottom = depth_map[up_y, x], depth_map[bottom_y, x]
            if depth_middle != 0:
                valid_depths.append((x, middle_y, depth_middle))
            if depth_up != 0:
                valid_depths.append((x, up_y, depth_up))
            if depth_bottom != 0:
                valid_depths.append((x, bottom_y, depth_bottom))
    return valid_depths


def write_3d_points(file_name, points_list):
    if path.exists(file_name):
        print('{} already exists, check it first'.format(file_name))
    with open(file_name, 'w') as outfile:
        for point in points_list:
            x, y, z, *_ = point
            outfile.write('{} {} {}\n'.format(x, y, z))


def call_ransac_plane_estimator(points_path, bin_path, threeD_point_list):
    temp_out_path = './tmp.txt'
    os.system('{} {} {}'.format(bin_path, points_path, temp_out_path))
    with open(temp_out_path, 'r') as infile:
        lines = infile.readlines()
    a, b, c = [float(item) for item in lines[0].split()]
    x0, y0, z0 = [float(item) for item in lines[1].split()]
    d = -(a * x0 + b * y0 + c * z0)
    plane = Plane(a, b, c, d)
    idxs = [int(item) for item in lines[2:]]
    selected_points = np.array(threeD_point_list)[idxs]

    return plane, selected_points


def one_scene(scene_base_dir, log_pattern):
    # read image, log, depth, intrinsic
    img_id, polygons = find_and_parse_log_file(scene_base_dir, log_pattern)

    img = cv2.imread(path.join(scene_base_dir, '{}_rgb.png'.format(img_id)))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    quality_depth_path = path.join(scene_base_dir, '{}_depth_quality.exr'.format(img_id))
    quality_depth = cv2.imread(quality_depth_path, cv2.IMREAD_UNCHANGED)
    assert quality_depth.shape[:2] == img.shape[:2], (img.shape, quality_depth.shape)
    cam = Cam(path.join(scene_base_dir, '{}_cam.txt'.format(img_id)))
    intrinsic = cam.intrinsic_mat
    extrinsic = cam.extrinsic_mat

    out_base_path = path.join(scene_base_dir, f'{img_id}_polygon_post_process', 'verbose_output')
    if path.exists(out_base_path):
        shutil.rmtree(out_base_path)
    os.makedirs(out_base_path)

    planes = []
    corner_points_list = []
    # find
    for polygon_id, polygon in enumerate(polygons):
        # gen 3d points of a polygon
        boundaries = polygon.boundaries
        polygon_corners = polygon.points
        corners_3d = []
        for polygon_corner in polygon_corners:
            u, v = polygon_corner
            x, y, z = PointCloudGenerator.get_3d_point((u, v), quality_depth[v, u], intrinsic)
            corners_3d.append((x, y, z))

        valid_cors = []
        for line in boundaries:
            valid_depths = extract_valid_depth_from_line(line, quality_depth)
            valid_cors.extend([item[:2] for item in valid_depths])
        point_list_3D = (PointCloudGenerator.gen_3d_point_with_rgb_v2(quality_depth, img, intrinsic,
                                                                          valid_cors))

        # write 3d points for ransac algorithm
        write_3d_points(path.join(out_base_path, '{}_{}_3d_points.txt'.format(img_id, polygon_id)), point_list_3D)

        # write 3d points obj file for visualization and debug
        PointCloudGenerator.write_as_obj(point_list_3D,
                                         path.join(out_base_path, '{}_{}_3d_points.obj'.format(img_id, polygon_id)))

        # call the ransac program, read the output file, get the plane and the selected points
        plane, selected_points = call_ransac_plane_estimator(
                                    path.join(out_base_path, '{}_{}_3d_points.txt'.format(img_id, polygon_id)),
                                    '/home/yeeef/Desktop/ransac_program/cmake-build-debug/ransac_program',
                                    point_list_3D)
        planes.append(plane)

        # write the selected points for visualization and debug
        PointCloudGenerator.write_as_obj(selected_points,
                                         path.join(out_base_path, f'{img_id}_{polygon_id}_plane_3d_points.obj'))

        # new corner points
        corner_points = []
        for point in corners_3d:
            x, y, z = point
            new_z = plane.get_z_given_xy(x, y)
            corner_points.append((x, y, new_z))
            print('previous: {}, now: {}'.format(z, new_z))
        corner_points_list.append(corner_points)

    # write everything to a log file for future usage
    log_path = path.join(scene_base_dir, f'{img_id}_polygon_post_process', f'{img_id}_planes_info.log')
    write_polygon_plane_log_file(log_path, extrinsic, intrinsic, planes, corner_points_list)


def write_polygon_plane_log_file(log_path, extrinsic, intrinsic, planes, corner_points_list):
    assert extrinsic.shape == (4, 4), extrinsic.shape
    assert intrinsic.shape == (3, 3), intrinsic.shape
    assert len(planes) == len(corner_points_list)

    def to_str_iterable(iterable):
        return [str(item) for item in iterable]

    num_polygon = len(planes)
    with open(log_path, 'w') as outfile:
        outfile.write('extrinsic\n')
        for i in range(4):
            extrinsic_row = extrinsic[i, :]
            outfile.write(' '.join(to_str_iterable(extrinsic_row)))
            outfile.write('\n')
        outfile.write('\n')
        outfile.write('intrinsic\n')
        for i in range(3):
            intrinsic_row = intrinsic[i, :]
            outfile.write(' '.join(to_str_iterable(intrinsic_row)))
            outfile.write('\n')
        outfile.write('\n')
        outfile.write('polygon %d\n' % num_polygon)
        for idx in range(num_polygon):
            corner_points = corner_points_list[idx]
            outfile.write("\npoints %d\n" % len(corner_points))
            for corner_point in corner_points:
                outfile.write(' '.join(to_str_iterable(corner_point)))
                outfile.write('\n')
            outfile.write('\nplane\n')
            plane = planes[idx]
            outfile.write("{} {} {} {}\n".format(plane.a, plane.b, plane.c, plane.d))


if __name__ == "__main__":
    base_dir = '/home/yeeef/Desktop/part1'
    LOG_PATTERN = re.compile(r'([0-9]{1,2})_fused_rgb\.log')
    #
    all_dirs = os.listdir(base_dir)
    all_dirs = [path.join(base_dir, item) for item in all_dirs]
    all_dirs = [item for item in all_dirs if path.isdir(item)]
    all_dirs = sorted(all_dirs, key=lambda x: int(path.basename(x)))
    for base_dir in all_dirs:
        one_scene(base_dir, LOG_PATTERN)
    #
    # one_scene('/home/yeeef/Desktop/part1/20', LOG_PATTERN)


