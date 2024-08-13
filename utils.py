# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import logging
import os
import sys
import torch
from data import ImDataset

def make_dataloader(cfg, is_train=True):
    dataset_list = cfg.DATASETS.TRAIN if is_train else cfg.DATASETS.TEST
    data_loaders = []
    if is_train:
        batch_size = cfg.SOLVER.BATCHSIZE
    else:
        batch_size = cfg.SOLVER.TEST_BATCH
    num_workers = cfg.DATALOADER.NUM_WORKERS
    datasets = []
    for dataset_name in dataset_list:
        dataset = ImDataset(dataset_name, is_train=is_train)
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            
        )
        data_loaders.append(data_loader)
        datasets.append(dataset)
    if is_train:
        # during training, a single (possibly concatenated) data_loader is returned
        assert len(data_loaders) == 1, print(len(data_loader))
        return data_loaders[0], None
    return data_loaders, datasets

def setup_logger(name, save_dir=None, distributed_rank=0, filename="log.txt"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    # don't log results for the non-master process
    if distributed_rank > 0:
        return logger
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if save_dir is not None:
        fh = logging.FileHandler(os.path.join(save_dir, filename))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger
