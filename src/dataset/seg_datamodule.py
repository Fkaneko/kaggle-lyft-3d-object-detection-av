import json

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import albumentations as albu
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
from omegaconf.omegaconf import OmegaConf
from torch.utils.data import DataLoader

from src.config.config import (
    BEV_TRAIN_SUFFIX,
    BEV_TARGET_SUFFIX,
    INPUT_META_JSON_NAME,
)

from src.dataset.bev_dataset import BEVImageDataset
from src.bev_processing.bev_utils import SampleMeta

IMG_MEAN = (0.485, 0.456, 0.406)
IMG_STD = (0.229, 0.224, 0.225)


class Lyft3DdetSegDatamodule(pl.LightningDataModule):
    def __init__(
        self,
        bev_data_dir: Path,
        val_hosts: int,
        batch_size: int = 64,
        num_workers: int = 16,
        aug_mode: int = 0,
        is_debug: bool = False,
        use_map: bool = False,
    ) -> None:
        super().__init__()
        self.bev_data_dir = Path(bev_data_dir)
        self.batch_size = batch_size
        self.aug_mode = aug_mode
        self.num_workers = num_workers
        self.is_debug = is_debug
        self.use_map = use_map
        self.val_hosts = self.get_val_hosts(val_hosts)
        self.input_size = 320

    def prepare_data(self):
        # check
        assert self.bev_data_dir.is_dir()

    def setup(self, stage: Optional[str] = None):
        # Assign Train/val split(s) for use in Dataloaders
        self.bev_config = OmegaConf.load(Path(self.bev_data_dir, "config.yaml"))
        self.bev_config.bev_data_dir = str(self.bev_data_dir)

        if stage == "fit" or stage is None:
            prefix = "train"
            input_json = self._load_meta_json(prefix)
            train_meta, val_meta = self._add_key_and_split(input_json, prefix)
            train_input_paths, train_target_paths = self._get_paths(
                train_meta, prefix=prefix
            )
            val_input_paths, val_target_paths = self._get_paths(val_meta, prefix=prefix)
            self.train_dataset = BEVImageDataset(
                train_input_paths,
                train_meta,
                train_target_paths,
                transforms=self.train_transform(),
            )
            self.val_dataset = BEVImageDataset(
                val_input_paths,
                val_meta,
                val_target_paths,
                transforms=self.val_transform(),
            )
            self.plot_dataset(self.train_dataset)

        # Assign Test split(s) for use in Dataloaders
        if stage == "test" or stage is None:
            prefix = "test"
            input_json = self._load_meta_json(prefix)
            test_meta, _ = self._add_key_and_split(input_json, prefix)
            test_input_paths, _ = self._get_paths(test_meta, prefix=prefix)
            self.test_dataset = BEVImageDataset(
                test_input_paths,
                test_meta,
                target_filepaths=None,
                transforms=self.test_transform(),
            )
            self.plot_dataset(self.test_dataset)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def train_transform(self):
        return self.get_transforms(mode=self.aug_mode)

    def val_transform(self):
        return self.get_transforms(mode=0)

    def test_transform(self):
        return self.get_transforms(mode=0)

    def get_val_hosts(self, val_choice: int = 0) -> Tuple[str, ...]:
        if val_choice == 0:
            return ("host-a007", "host-a008", "host-a009")
        elif val_choice == 1:
            return ("host-a004",)
        elif val_choice == 2:
            return ("host-a015", "host-a101", "host-a102")
        elif val_choice == 3:
            return ("host-a011",)
        else:
            print(f"unexpected val_choice:{val_choice}, so the default will be used")
            return ("host-a007", "host-a008", "host-a009")

    def _load_meta_json(self, prefix: str = "train") -> List[SampleMeta]:
        json_path = Path(self.bev_data_dir, prefix + "_images", INPUT_META_JSON_NAME)
        with open(json_path) as f:
            input_json = json.load(f)
        input_sample_meta: List[SampleMeta] = []
        sample_token_check: List[str] = []

        for input_ in input_json:
            input_sample_meta.append(SampleMeta(**input_))
            assert input_["sample_token"] not in sample_token_check
            sample_token_check.append(input_["sample_token"])
        return input_sample_meta

    def _add_key_and_split(
        self,
        input_json: List[SampleMeta],
        prefix: str = "train",
    ) -> Tuple[Dict[str, SampleMeta], Dict[str, SampleMeta]]:
        input_meta = {}
        val_meta = {}
        for sample_meta in input_json:
            if prefix == "train":
                if sample_meta.host in self.val_hosts:
                    val_meta[sample_meta.sample_token] = sample_meta
                    continue
            input_meta[sample_meta.sample_token] = sample_meta
        return input_meta, val_meta

    def _get_paths(
        self,
        input_meta: dict,
        prefix: str = "train"
        # ) -> Tuple[List[Path], List[Optional[Path]]]:
    ) -> Tuple[List[Path], List[Path]]:
        input_filepaths = []
        target_filepaths = []
        for key, values in input_meta.items():
            input_filepaths.append(
                Path(
                    self.bev_data_dir,
                    f"{prefix}_images",
                    values.host,
                    "{}_{}.png".format(key, BEV_TRAIN_SUFFIX),
                )
            )
            if prefix == "train":
                target_filepaths.append(
                    Path(
                        self.bev_data_dir,
                        f"{prefix}_images",
                        values.host,
                        "{}_{}.png".format(key, BEV_TARGET_SUFFIX),
                    )
                )
        return input_filepaths, target_filepaths

    def plot_dataset(
        self,
        dataset: BEVImageDataset,
        plot_num: int = 10,
    ) -> None:
        inds = np.random.choice(len(dataset), plot_num)
        for i in inds:
            plt.figure(figsize=(16, 8))
            data = dataset[i]
            im = data["image"][:3, :, :].numpy().transpose(1, 2, 0)
            im = im * np.array(IMG_STD) + np.array(IMG_MEAN)
            target_as_rgb = data["target"].numpy().clip(0.0, 1.0)
            target_as_rgb = np.repeat(target_as_rgb[:, :, np.newaxis], 3, axis=-1)
            if self.use_map:
                im_map = data["image"][-3:, :, :].numpy().transpose(1, 2, 0)
                im_map = im_map * np.array(IMG_STD) + np.array(IMG_MEAN)
                plt.imshow(np.hstack((im, im_map, target_as_rgb)))
            else:
                plt.imshow(np.hstack((im, target_as_rgb)))
            plt.title(data["sample_token"])
            if self.is_debug:
                plt.show()
            plt.close()

    def get_transforms(self, mode: int = 0) -> albu.core.composition.Compose:
        if mode == 0:
            transforms = [
                albu.Resize(self.input_size, self.input_size, p=1.0),
                albu.Normalize(mean=IMG_MEAN, std=IMG_STD),
            ]
        elif mode == 1:
            transforms = [
                albu.HorizontalFlip(p=0.5),
                albu.VerticalFlip(p=0.5),
                albu.RandomRotate90(p=1),
                albu.Transpose(p=0.5),
                albu.Resize(self.input_size, self.input_size, p=1.0),
                albu.Normalize(mean=IMG_MEAN, std=IMG_STD),
            ]
        else:
            raise NotImplementedError

        return albu.Compose(transforms)


def get_transforms(
    mode: int = 0, input_size: int = 320
) -> albu.core.composition.Compose:
    if mode == 0:
        transforms = [
            albu.Resize(input_size, input_size, p=1.0),
            albu.Normalize(mean=IMG_MEAN, std=IMG_STD),
        ]
    elif mode == 1:
        transforms = [
            albu.HorizontalFlip(p=0.5),
            albu.VerticalFlip(p=0.5),
            albu.RandomRotate90(p=1),
            albu.Transpose(p=0.5),
            albu.Resize(input_size, input_size, p=1.0),
            albu.Normalize(mean=IMG_MEAN, std=IMG_STD),
        ]
    else:
        raise NotImplementedError

    return albu.Compose(transforms)
