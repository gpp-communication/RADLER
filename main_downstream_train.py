#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import argparse
import builtins
import os
import random
import shutil
import time
import warnings
import subprocess

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import numpy as np

from networks.downstream import RadarObjectDetector
from networks.downstream.visualization.visualize_training_and_testing import visualize_training
from data_tools.downstream import DownstreamDataset
from models.ssl_encoder import radar_transform

parser = argparse.ArgumentParser(description="PyTorch Radar Object Detection Training with Semantic Depth Tensor")
parser.add_argument("data", metavar="DIR", help="path to dataset")
parser.add_argument(
    "-j",
    "--workers",
    default=32,
    type=int,
    metavar="N",
    help="number of data loading workers (default: 32)",
)
parser.add_argument(
    "--epochs", default=100, type=int, metavar="N", help="number of total epochs to run"
)
parser.add_argument(
    "--start-epoch",
    default=0,
    type=int,
    metavar="N",
    help="manual epoch number (useful on restarts)",
)
parser.add_argument(
    "-b",
    "--batch-size",
    default=256,
    type=int,
    metavar="N",
    help="mini-batch size (default: 256), this is the total "
         "batch size of all GPUs on the current node when "
         "using Data Parallel or Distributed Data Parallel",
)
parser.add_argument(
    "--lr",
    "--learning-rate",
    default=30.0,
    type=float,
    metavar="LR",
    help="initial learning rate",
    dest="lr",
)
parser.add_argument(
    "--schedule",
    default=[60, 80],
    nargs="*",
    type=int,
    help="learning rate schedule (when to drop lr by a ratio)",
)
parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
parser.add_argument(
    "--wd",
    "--weight-decay",
    default=0.0,
    type=float,
    metavar="W",
    help="weight decay (default: 0.)",
    dest="weight_decay",
)
parser.add_argument(
    "-p",
    "--print-freq",
    default=10,
    type=int,
    metavar="N",
    help="print frequency (default: 10)",
)
parser.add_argument(
    "--save-frequency",
    default='1',
    type=int,
    help="checkpoint file save frequency (default: 1)"
)
parser.add_argument(
    "--resume",
    default="",
    type=str,
    metavar="PATH",
    help="path to latest checkpoint (default: none)",
)
parser.add_argument(
    "--world-size",
    default=-1,
    type=int,
    help="number of nodes for distributed training",
)
parser.add_argument(
    "--rank", default=-1, type=int, help="node rank for distributed training"
)
parser.add_argument(
    "--dist-url",
    default="env://",
    type=str,
    help="url used to set up distributed training",
)
parser.add_argument(
    "--dist-backend", default="nccl", type=str, help="distributed backend"
)
parser.add_argument(
    "--seed", default=None, type=int, help="seed for initializing training. "
)
parser.add_argument("--gpu", default=None, type=int, help="GPU id to use.")
parser.add_argument(
    "--multiprocessing-distributed",
    action="store_true",
    help="Use multi-processing distributed training to launch "
         "N processes per node, which has N GPUs. This is the "
         "fastest way to use PyTorch for either single node or "
         "multi node data parallel training",
)
parser.add_argument(
    "--pretrained", default="", type=str, help="path to moco pretrained checkpoint"
)
parser.add_argument(
    "--fuse-semantic-depth-tensor", default=False, action="store_true",
    help="whether to fuse semantic depth tensor"
)
parser.add_argument(
    "--checkpoints-dir",
    default='./logs/checkpoints/downstream',
    type=str,
    help="folder path to save checkpoints"
)


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn(
            "You have chosen to seed training. "
            "This will turn on the CUDNN deterministic setting, "
            "which can slow down your training considerably! "
            "You may see unexpected behavior when restarting "
            "from checkpoints."
        )

    if args.gpu is not None:
        warnings.warn(
            "You have chosen a specific GPU. This will completely "
            "disable data parallelism."
        )

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    # suppress printing if not master
    if args.multiprocessing_distributed and args.gpu != 0:
        def print_pass(*args):
            pass

        builtins.print = print_pass

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["SLURM_PROCID"])
        if "MASTER_ADDR" not in os.environ:
            node_list = os.environ["SLURM_NODELIST"]
            os.environ["MASTER_ADDR"] = subprocess.getoutput(f"scontrol show hostname {node_list} | head -n1")
            if "MASTER_PORT" in os.environ:
                pass  # use MASTER_PORT in the environment variable
            else:
                os.environ["MASTER_PORT"] = "29500"
        dist.init_process_group(
            backend=args.dist_backend,
            init_method=args.dist_url,
            world_size=args.world_size,
            rank=args.rank,
        )
    # create model
    print("=> creating model '{}'".format("Radar Object Detector"))
    model = RadarObjectDetector(args.pretrained, 'train', 3, args.fuse_semantic_depth_tensor)

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[args.gpu]
            )
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.BCELoss().cuda(args.gpu)

    optimizer = torch.optim.Adam(
        model.parameters(), args.lr, weight_decay=args.weight_decay
    )

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = "cuda:{}".format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint["epoch"]
            model.load_state_dict(checkpoint["state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            print(
                "=> loaded checkpoint '{}' (epoch {})".format(
                    args.resume, checkpoint["epoch"]
                )
            )
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    args.checkpoints_dir = os.path.join(args.checkpoints_dir,
                                        '-'.join(['training', str(args.batch_size * ngpus_per_node), str(args.lr),
                                                  'fuse_semantic_depth_tensor_' + str(args.fuse_semantic_depth_tensor)
                                                  ]))
    os.makedirs(args.checkpoints_dir, exist_ok=True)

    cudnn.benchmark = True

    # Data loading code
    train_dataset_dir = args.data
    radar_transforms = radar_transform()
    semantic_depth_transforms = transforms.Compose([transforms.ToTensor()])
    train_dataset = DownstreamDataset(train_dataset_dir, radar_transforms, semantic_depth_transforms)

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=args.workers,
        pin_memory=True,
        sampler=train_sampler,
    )

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args)

        if not args.multiprocessing_distributed or (
                args.multiprocessing_distributed and args.rank % ngpus_per_node == 0
        ):
            if (epoch + 1) % args.save_frequency == 0:
                save_checkpoint(
                    {
                        "epoch": epoch + 1,
                        "arch": "Radar Object Detector",
                        "state_dict": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "fuse_semantic_depth_tensor": args.fuse_semantic_depth_tensor,
                    },
                    is_best=False,
                    checkpoints_dir=args.checkpoints_dir,
                    filename="checkpoint_{:04d}.pth.tar".format(epoch)
                )


def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4f")
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses],
        os.path.join(args.checkpoints_dir, "train.log"),
        prefix="Epoch: [{}]".format(epoch),
    )

    end = time.time()
    for i, (image_paths, radar_data, semantic_depth_tensors, gt_confmaps) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            radar_data = radar_data.cuda(args.gpu, non_blocking=True)
        gt_confmaps = gt_confmaps.cuda(args.gpu, non_blocking=True)

        # compute output
        output_confmap = model(radar_data, semantic_depth_tensors)
        loss = criterion(output_confmap, gt_confmaps)

        # visualize training process
        if i % 100 == 0:
            fig_path = os.path.join(args.checkpoints_dir, '%d_%d.png' % (epoch, i))
            image_path = image_paths[0]
            radar_path = image_path.replace('IMAGES_0', 'RADAR_RA_H').replace('png', 'npy')
            gt_confmap_path = image_path.replace('IMAGES_0', 'GT_CONFMAPS').replace('png', 'npy')
            raw_radar_data = np.load(radar_path)
            gt_confmap = np.load(gt_confmap_path)
            visualize_training(fig_path, image_path, raw_radar_data, output_confmap[0].cpu().detach().numpy(),
                               gt_confmap[:3, :, :])

        # measure accuracy and record loss
        losses.update(loss.item(), radar_data.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


def save_checkpoint(state, is_best, checkpoints_dir, filename="checkpoint.pth.tar"):
    torch.save(state, os.path.join(checkpoints_dir, filename))
    if is_best:
        shutil.copyfile(os.path.join(checkpoints_dir, filename), os.path.join(checkpoints_dir, "model_best.pth.tar"))


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


class ProgressMeter:
    def __init__(self, num_batches, meters, train_log, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.train_log = train_log
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        with open(self.train_log, 'a+') as f_log:
            f_log.write("\t".join(entries))
            f_log.write("\n")

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    for milestone in args.schedule:
        lr *= 0.1 if epoch >= milestone else 1.0
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


if __name__ == "__main__":
    main()
