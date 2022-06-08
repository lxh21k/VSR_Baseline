import argparse
import os
import common.utils as utils
from dataset.data_sampler import EnlargedSampler
from dataset.reds_dataset import REDSDataset
import random
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', help="Directory containing params.json")

def worker_init_fn(worker_id, num_workers, rank, seed):
    # Set the worker seed to num_workers * rank + worker_id + seed
    worker_seed = num_workers * rank + worker_id + seed
    random.seed(worker_seed)
    np.random.seed(worker_seed)

if __name__ == '__main__':
    # Load the parameters from json file
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    # use GPU if available
    params.cuda = torch.cuda.is_available()
    
    train_ds = REDSDataset(params.datasets)

    print(len(train_ds))

    train_sampler = EnlargedSampler(train_ds, 1, [1])
    print(len(train_sampler))

    train_dl = DataLoader(train_ds,
                          batch_size=params.batch_size, 
                          shuffle=False, 
                          sampler= train_sampler,
                          num_workers=params.num_workers,
                          drop_last=True,
                          pin_memory=params.cuda,
                          worker_init_fn=worker_init_fn
                )

