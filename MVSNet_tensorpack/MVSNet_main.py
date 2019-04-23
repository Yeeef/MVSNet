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


def get_data(args, mode):
    assert mode in ['train', 'val', 'test'], 'invalid mode: {}'.format(mode)

    ds = DTU(args.data, args.view_num, mode, args.interval_scale, args.max_d)
    parallel = min(40, multiprocessing.cpu_count() // 2)  # assuming hyperthreading
    if parallel < 16:
        logger.warn("DataFlow may become the bottleneck when too few processes are used.")
    if mode == 'train':
        ds = PrefetchData(ds, 4, parallel)
        ds = BatchData(ds, args.batch, remainder=False)
    else:
        ds = PrefetchData(ds, 4, parallel)
        ds = BatchData(ds, args.batch, remainder=True)

    return ds


def get_train_conf(model, args):
    nr_tower = max(get_num_gpu(), 1)
    batch = args.batch
    logger.info("Running on {} tower. Batch size per tower: {}".format(nr_tower, batch))
    ds_train = get_data(args, 'train')
    ds_val = get_data(args, 'val')
    train_size = len(ds_train)
    train_data = StagingInput(
        QueueInput(ds_train)
    )
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
                 filename='min_val_loss.tfmodel')
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
    parser.add_argument('--mode', '-m', help='train / val / test', default='train', choices=['train', 'val', 'test'])
    parser.add_argument('--out', default='./',
                        help='output path for evaluation and test, default to current folder')
    parser.add_argument('--batch', default=2, type=int, help="Batch size per tower.")
    parser.add_argument('--max_d', help='depth num for MVSNet', required=True, type=int)
    parser.add_argument('--interval_scale', required=True, type=float)
    parser.add_argument('--view_num', required=True, type=int)



    args = parser.parse_args()

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    model = MVSNet(depth_num=args.max_d)

    if args.mode == 'train':
        if args.exp_name is None:
            exp_name = '{}-{}-b{}-{}-{}'.format(args.max_d, args.interval_scale, args.batch, os.path.basename(args.data),
                                            datetime.datetime.now().strftime("%m%d-%H%M"))
        else:
            exp_name = args.exp_name
        logger.set_logger_dir(os.path.join(args.logdir, exp_name))
        config = get_train_conf(model, args)
        if args.load:
            config.session_init = get_model_loader(args.load)
        gpus = get_num_gpu()
        if gpus > 1:
            trainer = SyncMultiGPUTrainerParameterServer(gpus)
        else:
            trainer = SimpleTrainer()
        launch_train_with_config(config, trainer)

    elif args.mode == 'val':
        pass
    else:  # test
        pass


if __name__ == '__main__':
    mvsnet_main()