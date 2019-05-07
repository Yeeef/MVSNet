import os
import cv2
import numpy as np
from DataManager import PFMReader, Cam


class LogReader(object):
    def __init__(self, file_path):
        self.file_path = file_path
        with open(file_path, 'r') as infile:
            self.file_content_lines = infile.readlines()
        self.pt = 0
        self.end_of_file = len(self.file_content_lines)
        print('end of file: {}'.format(self.end_of_file))

    def is_eof(self):
        return self.pt >= self.end_of_file

    def pt_advance(self):
        self.pt += 1

    def parse(self):
        """

        :return:
        """
        parsed_list = []
        try:
            while True:
                parsed_list.append(self.parse_one_component())
        except ValueError:
            return parsed_list

    def pt_content(self):
        if self.is_eof():
            # self.pt = self.end_of_file - 1
            # print('reach the end of file already')
            raise ValueError
        return self.file_content_lines[self.pt]

    def parse_extrinsic(self):
        extrinsic_mat = []
        for i in range(4):
            row = self.pt_content().split(' ')
            row = [float(x) for x in row]
            extrinsic_mat.append(row)
            self.pt_advance()
        return np.linalg.inv(np.array(extrinsic_mat))
        #
        # return np.array(extrinsic_mat)

    def parse_one_component(self):
        pic_id = int(self.pt_content())
        self.pt_advance()
        f_c_list = self.pt_content().split(' ')
        f_c_list = [float(x) for x in f_c_list]
        fx, fy, cx, cy = f_c_list
        self.pt_advance()
        h_w_d_list = self.pt_content().split(' ')
        h_w_d_list = [float(x) for x in h_w_d_list]
        h, w, depth_min, depth_max = h_w_d_list
        self.pt_advance()
        intrinsic = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        extrinsic = self.parse_extrinsic()
        # print(f_c_list)
        return {'id': pic_id, 'h': h, 'w': w, 'depth_min': depth_min, 'depth_max': depth_max,
                'intrinsic': intrinsic, 'extrinsic': extrinsic}


class LogFormatter(object):
    def __init__(self):
        pass

    @staticmethod
    def format(parsed_dict, depth_min=425.0, depth_interval=2.5):
        write_buffer = []
        write_buffer.append('extrinsic')
        extrinsic = parsed_dict['extrinsic']
        for row in extrinsic:
            write_buffer.append(' '.join([str(item) for item in row]))
        # write_buffer.append('\n')
        write_buffer.append('\nintrinsic')
        intrinsic = parsed_dict['intrinsic']
        for row in intrinsic:
            write_buffer.append(' '.join([str(item) for item in row]))
        write_buffer.append('\n' + str(depth_min) + ' ' + str(depth_interval))
        return write_buffer


class ScaleHandler(object):
    def __init__(self):
        pass

    @staticmethod
    def scale_with_max_depths(dtu_max_depth, parsed_dict):
        depth_max = parsed_dict['depth_max']
        scale = dtu_max_depth // depth_max
        extrinsic = parsed_dict['extrinsic']
        assert extrinsic.shape == (4, 4), extrinsic.shape

        extrinsic[:3, 3] = extrinsic[:3, 3] * scale

        # intrinsic = parsed_dict['intrinsic']
        # assert intrinsic.shape == (3, 3), intrinsic.shape
        # intrinsic[0, 0] *= scale
        # intrinsic[1, 1] *= scale

        return parsed_dict


class PointCloudGenerator(object):
    def __init__(self):
        pass

    @staticmethod
    def get_fx_fy_cx_cy(intrinsic):
        fx, fy, cx, cy = intrinsic[0, 0], intrinsic[1, 1], intrinsic[0, 2], intrinsic[1, 2]
        return fx, fy, cx, cy

    @staticmethod
    def gen_3d_from_depth(depth_map, intrinsic):
        assert len(depth_map.shape) in [2, 3], depth_map.shape
        depth_map = np.squeeze(depth_map)
        h, w = depth_map.shape
        fx, fy, cx, cy = intrinsic[0, 0], intrinsic[1, 1], intrinsic[0, 2], intrinsic[1, 2]
        point_list = []
        for row in range(h):
            for col in range(w):
                u, v = col, row
                z = depth_map[row, col]
                x = (u - cx) * z / fx
                y = (v - cy) * z / fy
                point_list.append((x, y, z))

        return point_list

    @staticmethod
    def gen_3d_point_with_rgb(depth_map, rgb, intrinsic):
        """

        :param depth_map:
        :param rgb: RGB not BGR!
        :param intrinsic:
        :return:
        """
        assert len(depth_map.shape) in [2, 3], depth_map.shape
        assert len(rgb.shape) == 3, rgb.shape
        depth_map = np.squeeze(depth_map)
        h, w = depth_map.shape
        _h, _w, c = rgb.shape
        assert c == 3 and h == _h and w == _w, (h, w, c)

        fx, fy, cx, cy = intrinsic[0, 0], intrinsic[1, 1], intrinsic[0, 2], intrinsic[1, 2]
        point_list = []
        for row in range(h):
            for col in range(w):
                r, g, b = rgb[row, col]
                u, v = col, row
                z = depth_map[row, col]
                x = (u - cx) * z / fx
                y = (v - cy) * z / fy
                point_list.append((x, y, z, r, g, b))

        return point_list

    @staticmethod
    def write_as_obj(point_list, file_name):
        if not file_name.endswith('.obj'):
            file_name += '.obj'
        with open(file_name, 'w') as objfile:
            for point in point_list:
                str_points = [str(cor) for cor in point]
                objfile.write('v ' + ' '.join(str_points) + '\n')


def convert_png_to_jpg(dir):
    png_files = os.listdir(dir)
    png_files = list(filter(lambda x: os.path.splitext(x)[1] == '.png', png_files))
    png_files = sorted(png_files, key=lambda x: int(os.path.splitext(x)[0]))
    for idx, png_file in enumerate(png_files):
        img = cv2.imread(os.path.join(dir, png_file))
        # img = cv2.resize(img, (1425, 950))
        cv2.imwrite(os.path.join(dir, '%08d.jpg') % idx, img)


def get_dataset(log_dir, out_dir, depth_min=425.0, depth_interval=2.5):
    DTU_MAX_DEPTH = 933.8
    log_reader = LogReader(log_dir)
    # print(log_reader.end_of_file)
    paresed_list = log_reader.parse()
    print(len(paresed_list))
    for componet_dict in paresed_list:
        print(componet_dict['h'])
        print(componet_dict['w'])
        print(componet_dict['depth_min'])
        print(componet_dict['depth_max'])
        print(componet_dict['extrinsic'])
        print(componet_dict['intrinsic'])
        print('-'*100)
    for parsed_dict in paresed_list:
        # scaled_parsed_dict = ScaleHandler.scale_with_max_depths(DTU_MAX_DEPTH, parsed_dict)
        # write_lines = LogFormatter.format(scaled_parsed_dict)
        write_lines = LogFormatter.format(parsed_dict, depth_min=depth_min, depth_interval=depth_interval)
        print(len(write_lines))
        with open(os.path.join(out_dir, '%08d_cam.txt' % parsed_dict['id']), 'w') as outfile:
            # outfile.writelines(write_lines)
            for line in write_lines:
                outfile.write(line)
                outfile.write('\n')


def generate_3d_point_cloud(rgb_path, depth_path, cam_path):
    # depth_id = os.path.splitext(depth_path)[0]
    img = cv2.imread(rgb_path)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cam_id = os.path.splitext(cam_path)[0]
    # assert depth_id == cam_id, (depth_id, cam_id)
    cam = Cam(cam_path)
    intrinsic = cam.intrinsic_mat
    print('intrinsic: ')
    print(intrinsic)
    pfm_reader = PFMReader(depth_path)
    depth_map = pfm_reader.data
    ma = np.ma.masked_equal(depth_map, 0.0, copy=False)
    print('value range: ', ma.min(), ma.max())
    point_list = PointCloudGenerator.gen_3d_point_with_rgb(depth_map, rgb, intrinsic)
    PointCloudGenerator.write_as_obj(point_list, '%s.obj' % cam_id)


def scale_camera(log_dir):
    log_files = os.listdir(log_dir)
    # log_files = sorted(log_files, key=lambda x: os.path.splitext())
    for log_file in log_files:
        cam = Cam(os.path.join(log_dir, log_file), max_d=192)
        intrinsic = cam.intrinsic_mat
        extrinsic = cam.extrinsic_mat
        depth_min = cam.depth_min
        depth_interval = cam.depth_interval
        print(intrinsic)
        print(extrinsic)
        print(depth_min, depth_interval)
        # fx, fy = intrinsic[0, 0], intrinsic[1, 1]
        # intrinsic[0, 0], intrinsic[1, 1] = fx / 3.0, fy / 3.0
        extrinsic[:3, 3] = extrinsic[:3, 3] / 3.0
        depth_min = depth_min / 3.0
        depth_interval = depth_interval / 3.0

        with open(os.path.join(log_dir, log_file), 'w') as outfile:
            outfile.write('extrinsic\n')
            for row in extrinsic:
                row = [str(item) for item in row]
                row = ' '.join(row)
                outfile.write(row + '\n')
            outfile.write('\n')
            outfile.write('intrinsic\n')
            for row in intrinsic:
                row = [str(item) for item in row]
                row = ' '.join(row)
                outfile.write(row + '\n')
            outfile.write('\n')
            outfile.write(str(depth_min) + ' ' + str(depth_interval))


if __name__ == "__main__":
    # convert_png_to_jpg('/data3/lyf/multi_view/test_0/images')

    # imgs = os.listdir('/data3/lyf/multi_view/test_16/images')
    # imgs = [os.path.join('/data3/lyf/multi_view/test_16/images', img) for img in imgs]
    # for img_file in imgs:
    #     img = cv2.imread(img_file)
    #     img = cv2.resize(img, None, fx=0.946372, fy=0.946372)
    # get_dataset('/data3/lyf/multi_view/1/realitycapture/color_depth_log/camera_params.log',
    #             '/data3/lyf/multi_view/test_1/cams',
    #             20, 0.1)
    # get_dataset('/data3/lyf/multi_view/color_depth_log/camera_params.log', '/data3/lyf/multi_view/test_0/cams', 10, 0.1)

    # root_dir = '/data3/lyf/mvsnet/scan9/scan9_scale/depths_mvsnet'
    # root_dir = '/data3/lyf/mvsnet/preprocessed_inputs/tankandtemples/intermediate/M60/depths_mvsnet'
    root_dir = '/data3/lyf/multi_view/test_1/depths_mvsnet'

    for i in range(5):
        generate_3d_point_cloud(os.path.join(root_dir, '0000000%d.jpg' % i),
                            os.path.join(root_dir, '0000000%d_init.pfm' % i),
                            os.path.join(root_dir, '0000000%d.txt' % i))
    # scale_camera('/data3/lyf/mvsnet/scan9/scan9_scale/cams')
