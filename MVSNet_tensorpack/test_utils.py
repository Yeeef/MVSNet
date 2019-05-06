import os
import cv2
import numpy as np


class LogReader(object):
    def __init__(self, file_path):
        self.file_path = file_path
        with open(file_path, 'r') as infile:
            self.file_content_lines = infile.readlines()
        self.pt = 0
        self.end_of_file = len(self.file_content_lines)

    def is_eof(self):
        return self.pt >= self.end_of_file

    def pt_advance(self):
        self.pt += 1
        if self.is_eof():
            self.pt = self.end_of_file - 1
            # print('reach the end of file already')
            raise ValueError

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
        return self.file_content_lines[self.pt]

    def parse_extrinsic(self):
        extrinsic_mat = []
        for i in range(4):
            row = self.pt_content().split(' ')
            row = [float(x) for x in row]
            extrinsic_mat.append(row)
            self.pt_advance()
        return np.array(extrinsic_mat)

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
    def format(parsed_dict, depth_min=425, depth_interval=2.5):
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


if __name__ == "__main__":
    path = '/data3/lyf/mutilview/1/realitycapture/color_depth_log/camera_params.log'
    log_reader = LogReader(path)
    # print(log_reader.end_of_file)
    paresed_list = log_reader.parse()
    # print(len(paresed_list))
    # for componet_dict in paresed_list:
    #     print(componet_dict['h'])
    #     print(componet_dict['w'])
    #     print(componet_dict['depth_min'])
    #     print(componet_dict['depth_max'])
    #     print(componet_dict['extrinsic'])
    #     print(componet_dict['intrinsic'])
    #     print('-'*100)
    for parsed_dict in paresed_list:
        write_lines = LogFormatter.format(parsed_dict)
        print(len(write_lines))
        with open('{}_cam.txt'.format(parsed_dict['id']), 'w') as outfile:
            # outfile.writelines(write_lines)
            for line in write_lines:
                outfile.write(line)
                outfile.write('\n')