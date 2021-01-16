from pathlib import Path
from typing import Dict, List, Optional

import albumentations as albu
import cv2
import numpy as np
import torch
from src.bev_processing.bev_utils import SampleMeta

from src.config.config import BEV_TRAIN_SUFFIX


class BEVImageDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        input_filepaths: List[Path],
        input_meta_dict: Dict[str, SampleMeta],
        target_filepaths: Optional[List[Path]],
        transforms: albu.core.composition.Compose,
        map_filepaths: Optional[List[Path]] = None,
        multi_channel_filepaths: Optional[List] = None,
    ) -> None:
        self.input_filepaths = input_filepaths
        self.target_filepaths = target_filepaths
        self.input_meta_dict = input_meta_dict

        self.map_filepaths = map_filepaths
        self.transforms = transforms
        self.multi_channel_filepaths = multi_channel_filepaths

        if map_filepaths is not None:
            assert len(input_filepaths) == len(map_filepaths)
        self.test_mode = target_filepaths is None

    def __len__(self):
        return len(self.input_filepaths)

    def __getitem__(self, idx: int) -> dict:
        input_filepath = self.input_filepaths[idx]

        sample_token = input_filepath.name.replace(f"_{BEV_TRAIN_SUFFIX}.png", "")
        input_meta = self.input_meta_dict[sample_token]

        if self.multi_channel_filepaths is not None:
            im_list = []
            for i in range(len(self.multi_channel_filepaths)):
                input_filepath_ = self.multi_channel_filepaths[i][idx]
                im_list.append(cv2.imread(input_filepath_, cv2.IMREAD_UNCHANGED))
            im = np.concatenate(im_list, axis=2)

        else:
            im = cv2.cvtColor(
                cv2.imread(str(input_filepath), cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB
            )

        if self.map_filepaths is not None:
            map_filepath = self.map_filepaths[idx]
            map_im = cv2.cvtColor(
                cv2.imread(str(map_filepath), cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB
            )
            im = np.concatenate((im, map_im), axis=2)

        if self.target_filepaths is None:
            target = np.zeros_like(im[:, :, 0], dtype=np.int64)
        else:
            target_filepath = self.target_filepaths[idx]
            # target is grey image
            target = cv2.imread(str(target_filepath), cv2.IMREAD_UNCHANGED)

        if self.transforms:
            augmented = self.transforms(image=im, mask=target)
            im = augmented["image"]
            target = augmented["mask"]
            target = target.astype(np.int64)

        target = torch.from_numpy(target)

        im = torch.from_numpy(im.transpose(2, 0, 1))
        return {
            "image": im,
            "target": target,
            "sample_token": sample_token,
            "global_from_voxel": torch.tensor(input_meta.global_from_voxel),
            "ego_translation": torch.tensor(input_meta.ego_pose["translation"]),
            "ego_rotation": torch.tensor(input_meta.ego_pose["rotation"]),
        }
