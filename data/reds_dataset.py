import torch
import numpy as np
from torch.utils.data import Dataset
import os


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

        return {'lq': img_lqs, 'gt': img_gts, 'key': key}

    def __len__(self):
        return len(self.keys)