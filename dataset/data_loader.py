from cgi import test
from itertools import tee
import os
import random

# import nori2 as nori
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from reds_dataset import REDSDataset

def worker_init_fn(worker_id, num_workers, rank, seed):
    # Set the worker seed to num_workers * rank + worker_id + seed
    worker_seed = num_workers * rank + worker_id + seed
    random.seed(worker_seed)
    np.random.seed(worker_seed)

def fetch_dataloader(params):

    if params.dataset_type == 'REDS':
        train_ds = REDSDataset(params.datasets)
        val_ds = REDSDataset(params)
        test_ds = REDSDataset(params)

    dataloaders = {}
    train_dl = DataLoader(train_ds,
                          batch_size=params.batch_size, 
                          shuffle=False, 
                          num_workers=params.num_workers,
                          drop_last=True,
                          pin_memory=params.cuda,
                          worker_init_fn=worker_init_fn
                )

    dataloaders['train'] = train_dl

    for split in ['val', 'test']:
        if split in params.eval_type:
            if split == 'val':
                dl = DataLoader(val_ds,
                                batch_size=params.eval_batch_size, # 1
                                shuffle=False,
                                num_workers=params.num_workers,
                                pin_memory=params.cuda,
                    )
            if split == 'test':
                dl = DataLoader(test_ds,
                                batch_size=params.eval_batch_size, # 1
                                shuffle=False,
                                num_workers=params.num_workers,
                                pin_memory=params.cuda,
                    )
            else:
                raise ValueError("Unknow eval_type in params, should in [val, test]")
            dataloaders[split] = dl
        else:
            dataloaders[split] = None

    return dataloaders