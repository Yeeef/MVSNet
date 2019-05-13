from tensorpack import *
from tensorpack.utils import logger
import argparse
from mvsnet_model import MVSNet
import datetime
from tensorpack.utils.gpu import get_num_gpu
from tensorpack.predict import FeedfreePredictor
from dataflow_utils import *
import multiprocessing
import os
import tensorflow as tf
from nn_utils import (uni_feature_extraction_branch, unet_feature_extraction_branch)
from tensorpack.tfutils.gradproc import SummaryGradient
from matplotlib import pyplot as plt
from os import path
from DataManager import Cam
import cv2
import numpy as np
from test_utils import PointCloudGenerator


def get_data(args, mode):
    assert mode in ['train', 'val', 'test', 'fake'], 'invalid mode: {}'.format(mode)

    parallel = min(40, multiprocessing.cpu_count() // 2)  # assuming hyperthreading
    if parallel < 16:
        logger.warn("DataFlow may become the bottleneck when too few processes are used.")
    if mode == 'train':
        # ds = PrefetchData(ds, 4, parallel)
        ds = DTU(args.data, args.view_num, mode, args.interval_scale, args.max_d)
        # ds = PrefetchDataZMQ(ds, nr_proc=parallel)

        ds = BatchData(ds, args.batch, remainder=False)
    elif mode == 'val':
        ds = DTU(args.data, args.view_num, mode, args.interval_scale, args.max_d)
        # ds = PrefetchData(ds, 4, parallel)
        ds = BatchData(ds, args.batch, remainder=True)
        # ds = FakeData([[3, 512, 640, 3], [3, 2, 4, 4], [512 // 4, 640 // 4, 1]], 1)
        # ds = BatchData(ds, args.batch, remainder=False)
    else:
        ds = FakeData([[3, 512, 640, 3], [3, 2, 4, 4], [512 // 4, 640 // 4, 1]], 20)
        ds = BatchData(ds, args.batch, remainder=False)
    return ds


def get_train_conf(model, args):
    nr_tower = max(get_num_gpu(), 1)
    batch = args.batch
    logger.info("Running on {} tower. Batch size per tower: {}".format(nr_tower, batch))
    if args.mode == 'fake':
        ds_train = get_data(args, 'fake')
        ds_val = get_data(args, 'fake')
    else:

        # ds_train = get_data(args, 'train') if args.mode == 'train' else get_data(args, 'fake')
        ds_train = get_data(args, 'train')
        ds_val = get_data(args, 'val')
    train_size = len(ds_train)
    # train_data = StagingInput(
    #     QueueInput(ds_train)
    # )

    train_data = QueueInput(ds_train)
    val_data = QueueInput(ds_val)
    steps_per_epoch = train_size // nr_tower
    logger.info("train_size={}. steps_per_epoch={}".format(train_size, steps_per_epoch))
    callbacks = [
        ModelSaver(),
        EstimatedTimeLeft(),
        # SummaryGradient()
    ]
    infs = [ScalarStats(names=['loss'], prefix='val')]
    if nr_tower == 1:
        # single-GPU inference with queue prefetch
        callbacks.append(
            InferenceRunner(
                val_data,
                infs)
        )
    else:
        # multi-GPU inference (with mandatory queue prefetch)
        callbacks.append(DataParallelInferenceRunner(
            val_data, infs, list(range(nr_tower))))
    callbacks.extend([
        GPUUtilizationTracker(),
        MinSaver('val_loss',
                 filename='min_val_loss.tfmodel'),

        # GraphProfiler(dump_tracing=True, dump_event=True)

    ])

    return TrainConfig(
        # session_creator=creator,
        model=model,
        data=train_data,
        callbacks=callbacks,
        extra_callbacks=[ProgressBar(names=['loss', 'less_one_accuracy', 'less_three_accuracy']),
                         MovingAverageSummary(),
                         MergeAllSummaries(period=100 if steps_per_epoch > 100 else steps_per_epoch),
                         RunUpdateOps()],
        steps_per_epoch=steps_per_epoch,
        max_epoch=6,
    )


def evaluate(model, sess_init, args):
    """
    use feedforward trainconfig of tensorpack
    :return:
    """
    out_path = args.out
    if not os.path.exists(out_path):
        logger.warn(f'{out_path} does not exist, aumatically create for you')
        os.makedirs(out_path)
    else:
        logger.warn(f'{out_path} exists, it will be overwritten, y or n?')
        response = input()
        if response != 'y':
            logger.info(f'get {response} as answer, exit -1')
            exit(-1)
        else:
            logger.info(f'{out_path} will be overwritten')

    pred_conf = PredictConfig(
        model=model,
        session_init=sess_init,
        input_names=['imgs', 'cams', 'gt_depth'],
        output_names=['prob_map', 'coarse_depth', 'refine_depth', 'imgs', 'coarse_loss', 'refine_loss',
                      'less_one_accuracy', 'less_three_accuracy']
    )
    ds_val = get_data(args, 'val')
    pred_func = FeedfreePredictor(pred_conf, QueueInput(ds_val), device='/gpu:0')
    global_count = 0
    avg_loss = 0.
    avg_less_one_acc = 0.
    avg_less_three_acc = 0.
    ds_len = len(ds_val)
    for i in range(ds_len):
        prob_map, coarse_depth, refine_depth, imgs, coarse_loss, refine_loss, less_one_accuracy, less_three_accuracy = pred_func()
        batch_size, h, w, *_ = prob_map.shape
        ref_img = imgs[0]
        assert ref_img.shape[2] == 3, ref_img.shape
        for _ in range(batch_size):
            plt.imsave(path.join(out_path, str(global_count) + '_prob.png'), prob_map, cmap='rainbow')
            plt.imsave(path.join(out_path, str(global_count) + '_depth.png'), coarse_depth, cmap='rainbow')
            plt.imsave(path.join(out_path, str(global_count) + '_rgb.png'), ref_img.astype('uint8'))

            global_count += 1
            avg_loss += 0.5 * (coarse_loss + refine_depth)
            avg_less_one_acc += less_one_accuracy
            avg_less_three_acc += less_three_accuracy
    avg_loss /= ds_len
    avg_less_one_acc /= ds_len
    avg_less_three_acc /= ds_len
    with open(path.join(out_path, '!log.txt'), 'w') as out_file:
        out_file.write(f'loss: {avg_loss}\n')
        out_file.write(f'less_one_acc: {avg_less_one_acc}\n')
        out_file.write(f'less_three_acc: {avg_less_three_acc}\n')

    return avg_loss, avg_less_three_acc, avg_less_one_acc


def test(model, sess_init, args):
    """
    outputs prob_map, depth_map, rgb, and meshlab .obj file
    :param model:
    :param sess_init:
    :param args:
    :return:
    """
    data_dir = args.data
    out_dir = args.out
    view_num = args.view_num
    max_h = args.max_h
    max_w = args.max_w
    max_d = args.max_d
    interval_scale = args.interval_scale
    logger.info('data_dir: %s, out_dir: %s' % (data_dir, out_dir))
    if not os.path.exists(out_dir):
        logger.warn(f'{out_dir} does not exist, aumatically create for you')
        os.makedirs(out_dir)
    else:
        logger.warn(f'{out_dir} exists, it will be overwritten, y or n?')
        response = input()
        if response != 'y':
            logger.info(f'get {response} as answer, exit -1')
            exit(-1)
        else:
            logger.info(f'{out_dir} will be overwritten')
    pred_conf = PredictConfig(
        model=model,
        session_init=sess_init,
        input_names=['imgs', 'cams'],
        output_names=['prob_map', 'coarse_depth', 'refine_depth']
    )
    # create imgs and cams data
    data_points = list(DTU.make_test_data(data_dir, view_num, max_h, max_w, max_d, interval_scale))
    pred_func = OfflinePredictor(pred_conf)
    batch_prob_map, batch_coarse_depth, batch_refine_depth = pred_func(data_points)

    for i in range(len(batch_prob_map)):
        imgs, cams = data_points[i]
        prob_map, coarse_depth, refine_depth = batch_prob_map[i], batch_coarse_depth[i], batch_refine_depth[i]
        ref_img, ref_cam = imgs[0], cams[0]
        rgb = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)
        plt.imsave(path.join(out_dir, str(i) + '_prob.png'), prob_map, cmap='rainbow')
        plt.imsave(path.join(out_dir, str(i) + '_depth.png'), coarse_depth, cmap='rainbow')
        plt.imsave(path.join(out_dir, str(i) + '_rgb.png'), rgb.astype('uint8'))
        Cam.write_cam(ref_cam, path.join(out_dir, str(i) + '_cam.txt'))

        intrinsic = Cam.get_depth_meta(ref_cam, 'intrinsic')
        ma = np.ma.masked_equal(coarse_depth, 0.0, copy=False)
        logger.info('value range: %f -> %f' % (ma.min(), ma.max()))
        depth_point_list = PointCloudGenerator.gen_3d_point_with_rgb(coarse_depth, rgb, intrinsic)
        PointCloudGenerator.write_as_obj(depth_point_list, path.join(out_dir, '%s_depth.obj' % str(i)))


def mvsnet_main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', help='path to save model ckpt', default='.')
    parser.add_argument('--data', help='path to dataset', required=True)
    parser.add_argument('--load', help='load a model for training or evaluation')
    parser.add_argument('--exp_name', help='model ckpt name')
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
    parser.add_argument('--mode', '-m', help='train / val / test', default='train', choices=['train', 'val', 'test', 'fake'])
    parser.add_argument('--out', default='./',
                        help='output path for evaluation and test, default to current folder')
    parser.add_argument('--batch', default=1, type=int, help="Batch size per tower.")
    parser.add_argument('--max_d', help='depth num for MVSNet', required=True, type=int)
    parser.add_argument('--interval_scale', required=True, type=float)
    parser.add_argument('--view_num', required=True, type=int)
    parser.add_argument('--refine', default=False)
    parser.add_argument('--feature', help='feature extraction branch', choices=['uninet', 'unet'], default='unet')

    args = parser.parse_args()

    if args.feature == 'unet':
        feature_branch_function = unet_feature_extraction_branch
    else:
        feature_branch_function = uni_feature_extraction_branch

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if args.mode == 'train' or 'fake':

        model = MVSNet(depth_num=args.max_d, bn_training=None, bn_trainable=None, batch_size=args.batch,
                       branch_function=feature_branch_function, is_refine=args.refine)

        if args.exp_name is None:
            if not args.refine:
                exp_name = '{}-{}-b{}-{}-{}-{}-no-refine'.format(args.max_d, args.interval_scale, args.batch, os.path.basename(args.data),
                                                                args.feature,
                                            datetime.datetime.now().strftime("%m%d-%H%M"))
            else:
                exp_name = '{}-{}-b{}-{}-{}-{}-refine'.format(args.max_d, args.interval_scale, args.batch,
                                                    os.path.basename(args.data),
                                                    args.feature,
                                                    datetime.datetime.now().strftime("%m%d-%H%M"))
        else:
            exp_name = args.exp_name
        logger.set_logger_dir(os.path.join(args.logdir, exp_name))
        config = get_train_conf(model, args)
        if args.load:
            config.session_init = get_model_loader(args.load)
        gpus_id = args.gpu.split(',')
        gpus = len(gpus_id)
        logger.info('num of gpus to use: {}'.format(gpus))
        if gpus > 1:
            trainer = SyncMultiGPUTrainerParameterServer(gpus)
            # trainer = SyncMultiGPUTrainerReplicated(gpus, mode='cpu')
        else:
            trainer = SimpleTrainer()
        # trainer = SimpleTrainer()
        launch_train_with_config(config, trainer)

    elif args.mode == 'val':
        assert args.load, 'in eval mode, you have to specify a trained model'
        assert args.out, 'in eval mode, you have to specify the output dir path'
        model = MVSNet(depth_num=args.max_d, bn_training=None, bn_trainable=None, batch_size=args.batch,
                       branch_function=feature_branch_function, is_refine=args.refine)
        sess_init = get_model_loader(args.load)
        avg_loss, avg_less_three_acc, avg_less_one_acc = evaluate(model, sess_init, args)
        logger.info(f'val loss: {avg_loss}')
        logger.info(f'val less three acc: {avg_less_three_acc}')
        logger.info(f'val less one acc: {avg_less_one_acc}')

    else:  # test
        assert args.load, 'in eval mode, you have to specify a trained model'
        assert args.out, 'in eval mode, you have to specify the output dir path'
        assert args.data, 'in eval mode, you have to specify the data dir path'
        model = MVSNet(depth_num=args.max_d, bn_training=None, bn_trainable=None, batch_size=args.batch,
                       branch_function=feature_branch_function, is_refine=args.refine)
        sess_init = get_model_loader(args.load)
        test(model, sess_init, args)


if __name__ == '__main__':
    mvsnet_main()