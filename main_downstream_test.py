#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import argparse
import builtins
import os
import random
import warnings
import subprocess
import time
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms

from networks.downstream import RadarObjectDetector
from data_tools.downstream import DownstreamDataset
from models.ssl_encoder import radar_transform
from networks.downstream.post_processing import post_process_single_frame, write_single_frame_detection_results
from networks.downstream.visualization import visualize_test_img

parser = argparse.ArgumentParser(description="PyTorch Radar Object Detection Testing with Semantic Depth Tensor")
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
    "-p",
    "--print-freq",
    default=10,
    type=int,
    metavar="N",
    help="print frequency (default: 10)",
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
    "--pretrained", required=True, default="", type=str, help="path to Radar Object Detection checkpoint"
)
parser.add_argument(
    "--results-dir",
    default='./logs/results/',
    type=str,
    help="folder path to save results"
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
    model = RadarObjectDetector(None, 'test', 3)

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

    if os.path.isfile(args.pretrained):
        print("=> loading checkpoint '{}'".format(args.pretrained))
        if args.gpu is None:
            checkpoint = torch.load(args.pretrained)
        else:
            # Map model to be loaded to specified single gpu.
            loc = "cuda:{}".format(args.gpu)
            checkpoint = torch.load(args.pretrained, map_location=loc)
        model.load_state_dict(checkpoint["state_dict"])
        print(
            "=> loaded checkpoint '{}'".format(args.pretrained)
        )
    else:
        print("=> no checkpoint found at '{}'".format(args.pretrained))

    args.results_dir = os.path.join(args.results_dir, '-'.join(['testing', str(args.batch_size * ngpus_per_node)]))
    os.makedirs(args.results_dir, exist_ok=True)
    with open(os.path.join(args.results_dir, 'weight_source.txt'), 'w+') as f:
        f.write(args.pretrained)

    cudnn.benchmark = True

    # Data loading code
    test_dataset_dir = args.data
    radar_transforms = radar_transform()
    semantic_depth_transforms = transforms.Compose([transforms.ToTensor()])
    test_dataset = DownstreamDataset(test_dataset_dir, radar_transforms, semantic_depth_transforms)

    if args.distributed:
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
    else:
        test_sampler = None

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=(test_sampler is None),
        num_workers=args.workers,
        pin_memory=True,
        sampler=test_sampler,
    )

    # train for one epoch
    test(test_loader, model, args)


def test(test_loader, model, args):
    load_tic = time.time()
    for i, (image_paths, radar_data, semantic_depth_tensors, gt_confmaps) in enumerate(test_loader):
        load_time = time.time() - load_tic
        if args.gpu is not None:
            radar_data = radar_data.cuda(args.gpu, non_blocking=True)

        # compute output
        model.eval()
        inference_tic = time.time()
        with torch.no_grad():
            output_confmap = model(radar_data, semantic_depth_tensors)
        inference_time = time.time() - inference_tic
        output_confmap = output_confmap.detach().cpu().numpy()
        proc_tic = time.time()
        for j in range(output_confmap.shape[0]):
            results = post_process_single_frame(output_confmap[j])
            folder = os.path.join(os.path.join(args.results_dir, os.path.basename(os.path.dirname(os.path.dirname(image_paths[j])))))
            os.makedirs(folder, exist_ok=True)
            write_single_frame_detection_results(results, os.path.join(args.results_dir, os.path.basename(os.path.dirname(os.path.dirname(image_paths[j]))) + '.txt'),
                                                 os.path.basename(image_paths[j]).rstrip('.png'))
            image_path = image_paths[j]
            radar_path = image_path.replace('IMAGES_0', 'RADAR_RA_H').replace('png', 'npy')
            gt_confmap_path = image_path.replace('IMAGES_0', 'GT_CONFMAPS').replace('png', 'npy')
            raw_radar_data = np.load(radar_path)
            gt_confmap = np.load(gt_confmap_path)
            test_img_path = os.path.join(folder, os.path.basename(image_path))
            output_confmap_path = test_img_path.replace('png', 'npy')
            visualize_test_img(test_img_path, image_path, raw_radar_data, output_confmap[j], gt_confmap[:3, :, :], results)
            with open(output_confmap_path, 'wb') as f:
                np.save(f, output_confmap[j])

        proc_time = time.time() - proc_tic
        print("Testing: step:%d/%d | Load time: %.4f | Inference time: %.4f | Process time: %.4f" %
              (i, 2000 / args.batch_size, load_time, inference_time, proc_time))
        load_tic = time.time()


if __name__ == "__main__":
    main()
