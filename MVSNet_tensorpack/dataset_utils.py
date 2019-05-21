"""
File: dataset_utils.py
Contains utility function for dataset preparation, generation etc.
"""
import os
import re
from os import path
from test_utils import gen_dataset
from tensorpack.utils import logger
import tqdm
import numpy as np
from adaptive_depth_scale import *

"""
bad: 5 16 18 19 21 23
"""
DIR_COLLECTIONS = ['/data3/lyf/mvsnet_test/xuhantong20190503/selected_images',
                   '/data3/lyf/mvsnet_test/xuhantong20190503/selected_images2/selected_images',
                   '/data3/lyf/mvsnet_test/xuhantong20190503/selected_images2/selected_images_1',
                   ]
NUM_RE = re.compile(r'(\d+)[real]*')
OUT_BASE = '/data3/lyf/mvsnet_test/standard_dataset'


if __name__ == "__main__":
    cur_dir = DIR_COLLECTIONS[0]
    logger.info('cur_dir: %s' % cur_dir)
    dirs = os.listdir(cur_dir)
    dirs = [path.join(cur_dir, dir_) for dir_ in dirs if path.isdir(path.join(cur_dir, dir_))]
    out_base = path.join(OUT_BASE, 'part1_adaptives')
    invalid_dir = []
    for dir_ in tqdm.tqdm(dirs):
        basename = path.basename(dir_)
        num_match = re.search(NUM_RE, basename)
        if not num_match:
            continue
        try:
            num = int(num_match.group(1))
        except TypeError as te:
            # print(te)
            # print(num_match.group())
            print(basename)
            print(num_match.group(1))
            exit(-1)
        base_dir = path.join(dir_, 'realitycapture')
        # base_dir = path.join(dir_, 'images')
        # if not path.exists(base_dir):
        #     base_dir = path.join(dir_, 'image')
        sparse_point_cloud_obj_path = path.join(base_dir, 'bundler_point_cloud.obj')
        cors = parse_obj_file(sparse_point_cloud_obj_path)

        try:
            gen_dataset(base_dir, path.join(out_base, str(num)), depth_min=20, depth_interval=0.1)
        except FileNotFoundError:
            invalid_dir.append(dir_)

    logger.warn('invalid_dirs: %s' % invalid_dir)
    logger.warn('%d dirs are invalid' % len(invalid_dir))
    base_dir = '/data3/lyf/mvsnet_test/xuhantong20190503/selected_images2/selected_images/4/images/'
    out_dir = path.join(OUT_BASE, 'test_scale')
    gen_dataset(base_dir, out_dir, depth_min=5, depth_interval=0.1)

    base_dir = '/data3/lyf/mvsnet_test/xuhantong20190503/selected_images2/selected_images/10real/image/'
    out_dir = path.join(OUT_BASE, 'test_scale', '1')

    gen_dataset(base_dir, out_dir, depth_min=40, depth_interval=0.1)
