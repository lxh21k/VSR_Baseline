from pkg_resources import working_set
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
import random

from utils.img_utils import imfrombytes, img2tensor
from transforms import augment, paired_random_crop


class REDSDataset(Dataset):
    """
    Dataset class for REDS dataset.
    """
    def __init__(self, gt_root, lq_root, meta_info_file, num_frame):
        """
        Args:
            gt_root (string): Data root path for gt.
            lq_root (string): Data root path for lq.

            meta_info_file (string): Path for meta information file.

            num_frame (int): Window size for input frames.
        """

        self.gt_root = gt_root
        self.lq_root = lq_root
        self.num_frame = num_frame
        self.meta_info_file = meta_info_file

        self.gt_patch_size = 256
        self.scale = 4

        # generate frame index from meta info file
        self.keys = []
        with open(self.meta_info_file, 'r') as f:
            for line in f:
                folder, frame_num, _ = line.split(' ')
                self.keys.extend([f'{folder}/{i:08d}' for i in range(int(frame_num))])

        # remove the video clips used in validation set (ref from REDS4)
        val_partition = ['000', '011', '015', '020']

        self.key = [v for v in self.keys if v.split('/')[0] not in val_partition]


    

    def __getitem__(self, idx):
        
        key = self.key[idx]    # key example: '000/00000000'
        clip_name, frame_name = key.split('/') 
        center_frame_idx = int(frame_name)

        # ensure not exceeding the boundary
        start_frame_idx = center_frame_idx - self.num_frame // 2
        end_frame_idx = center_frame_idx + self.num_frame // 2
        # why not break loop when indexes exceed the boundary?
        while (start_frame_idx < 0) or (end_frame_idx > 99):
            center_frame_idx = random.randint(0, 99)
            start_frame_idx = center_frame_idx - self.num_frame // 2
            end_frame_idx = center_frame_idx + self.num_frame // 2
        frame_name = f'center_frame_idx:08d'
        neighbor_list = list(range(start_frame_idx, end_frame_idx+1))

        # random reverse
        if random.random() > 0.5:
            neighbor_list.reverse()

        # get the GT frame (as the center frame)
        img_gt_path = f'{self.gt_root}/{clip_name}/{frame_name}.png'
        with open(img_gt_path, 'rb') as f:
            img_bytes = f.read()
        img_gt = imfrombytes(img_bytes, float32=True)
        
        # get the LQ frames
        img_lqs = []
        for neighbor_frame_idx in neighbor_list:
            img_lq_path = f'{self.lq_root}/{clip_name}/{neighbor_frame_idx:08d}.png'
            with open(img_lq_path, 'rb') as f:
                img_bytes = f.read()
            img_lq = imfrombytes(img_bytes, float32=True)
            img_lqs.append(img_lq)


        img_gt, img_lqs = paired_random_crop(img_gt, img_lqs, self.gt_patch_size, self.scale, img_gt_path)

        # augmentation - flip, rotate
        img_lqs.append(img_gt)
        img_results = augment(img_lqs, hflip=True, rotate=True)

        img_results = img2tensor(img_results)
        img_lqs = torch.stack(img_results[:-1], dim=0)
        img_gt = img_results[-1] 

        return {'lq': img_lqs, 'gt': img_gt, 'key': key}    

    def __len__(self):
        return len(self.keys)

def build_dataset(dataset_type):
    dataset = REDSDataset()

    return dataset

def build_dataloader(dataset, params):
    dataloaders = {}

    train_dl = DataLoader(
        dataset=dataset,
        batch_size=params['batch_size'],
        shuffle=True,
        num_workers=params['num_workers'],
        pin_memory=params['cuda'],
        drop_last=True
        worker_init_fn= # ?
    )
