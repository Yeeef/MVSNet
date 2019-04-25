from tensorpack import *
from tensorpack.utils import logger
import argparse
from mvsnet_model import MVSNet
import datetime
from tensorpack.utils.gpu import get_num_gpu
from dataflow_utils import *
import multiprocessing
import os
import tensorflow as tf
from nn_utils import (uni_feature_extraction_branch, unet_feature_extraction_branch)


def get_data(args, mode):
    assert mode in ['train', 'val', 'test', 'fake'], 'invalid mode: {}'.format(mode)

    parallel = min(40, multiprocessing.cpu_count() // 2)  # assuming hyperthreading
    if parallel < 16:
        logger.warn("DataFlow may become the bottleneck when too few processes are used.")
    if mode == 'train':
        # ds = PrefetchData(ds, 4, parallel)
        ds = DTU(args.data, args.view_num, mode, args.interval_scale, args.max_d)
        ds = PrefetchDataZMQ(ds, nr_proc=parallel)

        ds = BatchData(ds, args.batch, remainder=False)
    elif mode == 'val':
        ds = DTU(args.data, args.view_num, mode, args.interval_scale, args.max_d)
        ds = PrefetchData(ds, 4, parallel)
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
        extra_callbacks=[ProgressBar(names=['loss']),
                         MovingAverageSummary(),
                         MergeAllSummaries(period=100 if steps_per_epoch > 100 else steps_per_epoch),
                         RunUpdateOps()],
        steps_per_epoch=steps_per_epoch,
        max_epoch=6,
    )


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
    parser.add_argument('--batch', default=2, type=int, help="Batch size per tower.")
    parser.add_argument('--max_d', help='depth num for MVSNet', required=True, type=int)
    parser.add_argument('--interval_scale', required=True, type=float)
    parser.add_argument('--view_num', required=True, type=int)
    parser.add_argument('--refine', default=False)
    parser.add_argument('--feature', help='feature extraction branch', choices=['uninet', 'unet'], default='unet')

    args = parser.parse_args()

    if args.feature == 'unet':
        feature_branch_function = unet_feature_extraction_branch
    else:
        feature_branch_function = uni_feature_extraction_branch()

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if args.mode == 'train' or 'fake':

        model = MVSNet(depth_num=args.max_d, bn_training=None, bn_trainable=None, batch_size=args.batch,
                       branch_function=feature_branch_function, is_refine=args.refine)

        if args.exp_name is None:
            if not args.refine:
                exp_name = '{}-{}-b{}-{}-{}-no-refine'.format(args.max_d, args.interval_scale, args.batch, os.path.basename(args.data),
                                            datetime.datetime.now().strftime("%m%d-%H%M"))
            else:
                exp_name = '{}-{}-b{}-{}-{}-refine'.format(args.max_d, args.interval_scale, args.batch,
                                                    os.path.basename(args.data),
                                                    datetime.datetime.now().strftime("%m%d-%H%M"))
        else:
            exp_name = args.exp_name
        logger.set_logger_dir(os.path.join(args.logdir, exp_name))
        config = get_train_conf(model, args)
        if args.load:
            config.session_init = get_model_loader(args.load)
        gpus_id = args.gpu.split(',')
        gpus = len(gpus_id)
        if gpus > 1:
            trainer = SyncMultiGPUTrainerParameterServer(gpus)
            # trainer = SyncMultiGPUTrainerReplicated(gpus, mode='cpu')
        else:
            trainer = SimpleTrainer()
        # trainer = SimpleTrainer()
        launch_train_with_config(config, trainer)

    elif args.mode == 'val':
        pass
    else:  # test
        pass


if __name__ == '__main__':
    mvsnet_main()