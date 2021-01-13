#!/usr/bin/env python
# coding: utf-8

import argparse
import json
from random import sample


from make_image import transform_points
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import albumentations as albu
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch
from omegaconf.omegaconf import OmegaConf
from torch.utils.data import DataLoader

import lyft_3d_utils
from lyft_3d_utils import (
    BEV_TARGET_SUFFIX,
    BEV_TRAIN_SUFFIX,
    CLASSES,
    CLASS_LOSS_WEIGHTS,
    INPUT_META_JSON_NAME,
    BEVImageDataset,
    SEED,
    SampleMeta,
)

# output path for test mode
CSV_PATH = "./submission.csv"


def calc_detection_box(
    prediction_opened: np.ndarray,
    raw_prediction: np.ndarray,
    classes: Tuple[str, ...] = CLASSES,
) -> Tuple[List[np.ndarray], List[float], List[str]]:

    sample_boxes = []
    sample_detection_scores = []
    sample_detection_classes = []
    contours, _ = cv2.findContours(
        prediction_opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    probability_wo_bkg = raw_prediction[:, :, 1:]
    for cnt in contours:
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)

        # Let's take the center pixel value as the confidence value
        box_center_index = np.int0(np.mean(box, axis=0))

        # Be carfull! y and x correspond to the first and second pixel
        max_class_index = np.argmax(
            probability_wo_bkg[box_center_index[1], box_center_index[0], :]
        )
        box_center_value = probability_wo_bkg[
            box_center_index[1], box_center_index[0], max_class_index
        ]
        sample_detection_classes.append(classes[max_class_index])
        sample_detection_scores.append(box_center_value)
        sample_boxes.append(box)

    return sample_boxes, sample_detection_scores, sample_detection_classes


class Lyft3DdetSegDatamodule(pl.LightningDataModule):
    def __init__(
        self,
        bev_data_dir: Path,
        val_hosts: Tuple[str, ...],
        batch_size: int = 440,
        num_workers: int = 16,
        is_debug: bool = False,
        use_map: bool = False,
        img_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        img_std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    ) -> None:
        super().__init__()
        self.bev_data_dir = Path(bev_data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.is_debug = is_debug
        self.use_map = use_map
        self.val_hosts = val_hosts
        self._img_mean = img_mean
        self._img_std = img_std
        self.input_size = 320

    def prepare_data(self):
        # check
        assert self.bev_data_dir.is_dir()

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
            self.plot_dataset(self.val_dataset)

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
        transforms = [
            # albu.HorizontalFlip(p=0.5),
            # albu.VerticalFlip(p=0.5),
            # albu.RandomRotate90(p=1),
            # albu.Transpose(p=0.5),
            albu.Resize(self.input_size, self.input_size, p=1.0),
            albu.Normalize(mean=self._img_mean, std=self._img_std),
        ]
        return albu.Compose(transforms)

    def val_transform(self):
        transforms = [
            albu.Resize(self.input_size, self.input_size, p=1.0),
            albu.Normalize(mean=self._img_mean, std=self._img_std),
        ]
        return albu.Compose(transforms)

    def test_transform(self):
        transforms = [
            albu.Resize(self.input_size, self.input_size, p=1.0),
            albu.Normalize(mean=self._img_mean, std=self._img_std),
        ]
        return albu.Compose(transforms)

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
            im = im * np.array(self._img_std) + np.array(self._img_mean)
            target_as_rgb = data["target"].numpy().clip(0.0, 1.0)
            target_as_rgb = np.repeat(target_as_rgb[:, :, np.newaxis], 3, axis=-1)
            if self.use_map:
                im_map = data["image"][-3:, :, :].numpy().transpose(1, 2, 0)
                im_map = im_map * np.array(self._img_std) + np.array(self._img_mean)
                plt.imshow(np.hstack((im, im_map, target_as_rgb)))
            else:
                plt.imshow(np.hstack((im, target_as_rgb)))
            plt.title(data["sample_token"])
            if self.is_debug:
                plt.show()
            plt.close()


class LitModel(pl.LightningModule):
    def __init__(
        self,
        bev_config: dict,
        val_hosts: Tuple[str, ...],
        dataset_len: int,
        ba_size: int = 128,
        lr: float = 3.0e-4,
        class_weights: List[float] = CLASS_LOSS_WEIGHTS,
        backbone_name: str = "efficientnet-b1",
        epochs: int = 15,
        optim_name: str = "adam",
        in_channels: int = 3,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model = smp.FPN(
            encoder_name=backbone_name,
            encoder_weights="imagenet",
            in_channels=in_channels,
            classes=len(CLASSES) + 1,
        )
        self.criterion = torch.nn.CrossEntropyLoss(torch.tensor(class_weights))

        self.background_threshold = 200
        self.morph_kernel_size = 3
        self.kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (self.morph_kernel_size, self.morph_kernel_size)
        )
        # self.f1 = pl.metrics.F1(num_classes=len(CLASSES) + 1)
        self.f1 = pl.metrics.F1()

    def forward(self, x):
        x = self.model(x)
        return x

    def training_step(self, batch, batch_idx):
        inputs = batch["image"]
        targets = batch["target"]

        outputs = self.model(inputs)
        loss = self.criterion(outputs, targets)
        non_bkg_proba = 1 - outputs.softmax(dim=1)[:, 0]
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        self.log(
            "train_f1",
            self.f1(non_bkg_proba, (targets > 0).type(torch.int64)),
            on_step=True,
            on_epoch=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        inputs = batch["image"]
        targets = batch["target"]

        outputs = self.model(inputs)
        loss = self.criterion(outputs, targets)
        self.log("val_loss", loss)
        non_bkg_proba = 1 - outputs.softmax(dim=1)[:, 0]
        self.log("val_f1", self.f1(non_bkg_proba, (targets > 0).type(torch.int64)))
        if batch_idx == 0:
            num_ = 2
            non_bkg_proba = non_bkg_proba.detach()[:num_]
            # (num_, H, W) -> (num_, 3, H, W)
            class_rgb = torch.repeat_interleave(non_bkg_proba[:, None], 3, dim=1)
            class_rgb[:, 2] = 0
            class_rgb[:, 1] = targets[:num_]
            # (num_, 3, H, W) -> (3, H*num_, W)
            class_rgb = torch.cat([class_rgb[i] for i in range(num_)], dim=1)
            inputs_rgb = torch.cat([inputs[i] for i in range(num_)], dim=1)
            # (3, H*num_, W)*2 -> (3, H*num_, W*2)
            img = torch.cat((inputs_rgb, class_rgb), dim=2)
            # img_ch = torch.cat((class_rgb[1], class_rgb[0]), dim=2)
            self.logger.experiment.add_image(
                "prediction",
                img.type(torch.float32).cpu(),
                global_step=self.global_step,
            )
            # self.logger.experiment.add_image(
            #     "image_ch",
            #     img_ch.type(torch.float32).cpu(),
            #     global_step=self.global_step,
            # )
        return loss

    def test_step(self, batch, batch_idx):
        inputs = batch["image"]
        outputs = self.model(inputs)

        outputs = outputs.softmax(dim=1)
        predictions = np.round(outputs.cpu().numpy() * 255).astype(np.uint8)
        predictions = np.transpose(predictions, (0, 2, 3, 1))

        global_from_voxel = batch["global_from_voxel"].numpy()
        ego_pose = batch["ego_pose"].numpy()

        detection_dimensions = []
        detection_centers = []
        detection_scores = []
        detection_classes = []
        for i, raw_pred in enumerate(predictions):
            raw_pred = cv2.resize(
                raw_pred,
                dsize=(
                    self.hparams.bev_config.image_size,
                    self.hparams.bev_config.image_size,
                ),
                interpolation=cv2.INTER_LINEAR,
            )
            # [336, 336]
            bkg_probability = 255 - raw_pred[:, :, 0]
            thresholded_p = (bkg_probability > self.background_threshold).astype(
                np.uint8
            )
            predictions_opened = cv2.morphologyEx(
                thresholded_p, cv2.MORPH_OPEN, self.kernel
            )
            # get all 2D boxes from predicted image
            (
                sample_boxes,
                sample_detection_scores,
                sample_detection_classes,
            ) = calc_detection_box(predictions_opened, raw_pred, CLASSES)

            # store the all boxes on the list
            (
                sample_boxes_centers,
                sample_boxes_dimensions,
            ) = self.create_3d_boxes_from_2d(
                np.array(sample_boxes),
                sample_detection_scores,
                sample_detection_classes,
                ego_pose,
                global_from_voxel,
            )
            detection_centers.append(sample_boxes_centers)
            detection_dimensions.append(sample_boxes_dimensions)
            detection_scores.append(sample_detection_scores)
            detection_classes.append(sample_detection_classes)
        return {
            "sample_token": batch["sample_token"],
            "boxes_centers": detection_centers,
            "boxes_dimenstions": detection_dimensions,
            "detection_scores": detection_scores,
            "detection_classes": detection_classes,
        }

    def create_3d_boxes_from_2d(
        self,
        sample_boxes,
        sample_detection_classes,
        ego_pose,
        global_from_voxel,
        box_scale: float = 0.8,
    ):
        # make json pred
        sample_boxes = sample_boxes.reshape(-1, 2)  # (N, 4, 2) -> (N*4, 2)
        sample_boxes = sample_boxes.transpose(1, 0)  # (N*4, 2) -> (2, N*4)
        # Add Z dimension,  (2, N*4) -> (3, N*4)
        sample_boxes = np.vstack((sample_boxes, np.zeros(sample_boxes.shape[1])))
        # Transform box cordinate into global (3, N*4)
        sample_boxes = transform_points(sample_boxes, global_from_voxel)

        # We don't know at where the boxes are in the scene on the z-axis (up-down), let's assume all of them are at
        # the same height as the ego vehicle.
        sample_boxes[2, :] = ego_pose["translation"][2]
        # (3, N*4) -> (N, 4, 3)
        sample_boxes = sample_boxes.transpose(1, 0).reshape(-1, 4, 3)

        # We don't know the height of our boxes, let's assume every object is the same height.
        # box_height = 1.75
        box_height = np.array(
            [CLASS_AVG_HEIGHTS[cls_] for cls_ in sample_detection_classes]
        )

        # Note: Each of these boxes describes the ground corners of a 3D box.
        # To get the center of the box in 3D, we'll have to add half the height to it.
        sample_boxes_centers = sample_boxes.mean(axis=1)  # (N, 3)
        sample_boxes_centers[:, 2] += box_height / 2

        # Width and height is arbitrary - we don't know what way the vehicles are pointing from our prediction segmentation
        # It doesn't matter for evaluation, so no need to worry about that here.
        # Note: We scaled our targets to be 0.8 the actual size, we need to adjust for that
        sample_lengths = (
            np.linalg.norm(sample_boxes[:, 0, :] - sample_boxes[:, 1, :], axis=1)
            * 1
            / box_scale
        )  # N
        sample_widths = (
            np.linalg.norm(sample_boxes[:, 1, :] - sample_boxes[:, 2, :], axis=1)
            * 1
            / box_scale
        )  # N

        sample_boxes_dimensions = np.zeros_like(sample_boxes_centers)  # (N, 3)
        sample_boxes_dimensions[:, 0] = sample_widths
        sample_boxes_dimensions[:, 1] = sample_lengths
        sample_boxes_dimensions[:, 2] = box_height
        return sample_boxes_centers, sample_boxes_dimensions

    # def writing_each_3d_box(self):
    #     for i in range(len(sample_boxes)):
    #         translation = sample_boxes_centers[i]
    #         size = sample_boxes_dimensions[i]
    #         class_name = sample_detection_class[i]

    #         if args.width_len_average_set > 0:
    #             if class_name in width_len_average_set:
    #                 size[0] = class_widths[class_name]
    #                 size[1] = class_lengths[class_name]

    #         # Determine the rotation of the box
    #         # TODO: OUTPUT-PREDICTION, only z axis rotaions are exist?
    #         # TODO: OUTPUT-PREDICTION, yaw_deg consistent check <DONE>
    #         v = sample_boxes[i, 0] - sample_boxes[i, 1]  # (3, )
    #         v /= np.linalg.norm(v)
    #         r = R.from_dcm(
    #             [
    #                 [v[0], -v[1], 0],
    #                 [v[1], v[0], 0],
    #                 [0, 0, 1],
    #             ]
    #         )
    #         quat = r.as_quat()
    #         # XYZW -> WXYZ order of elements
    #         quat = quat[[3, 0, 1, 2]]

    #         detection_score = float(sample_detection_scores[i])

    #         box3d = Box3D(
    #             sample_token=sample_token,
    #             translation=list(translation),
    #             size=list(size),
    #             rotation=list(quat),
    #             name=class_name,
    #             score=detection_score,
    #         )
    #         pred_box3ds.append(box3d)

    #     pred = [b.serialize() for b in pred_box3ds]
    #     with open(os.path.join(model_out_dir, pred_json_file_name), "w") as f:
    #         json.dump(pred, f)

    # def test_epoch_end(self, outputs):
    #     """from https://www.kaggle.com/pestipeti/pytorch-baseline-inference"""
    #     # convert into world coordinates and compute offsets
    #     for outputs, confidences, batch in outputs:
    #         outputs = outputs.cpu().numpy()

    #     sample_boxes = sample_boxes.reshape(-1, 2)  # (N, 4, 2) -> (N*4, 2)
    #     sample_boxes = sample_boxes.transpose(1, 0)  # (N*4, 2) -> (2, N*4)

    def configure_optimizers(self):
        if self.hparams.optim_name == "sgd":
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.hparams.lr,
                momentum=0.9,
                weight_decay=4e-5,
            )
        elif self.hparams.optim_name == "adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        else:
            raise NotImplementedError
        steps_per_epoch = self.hparams.dataset_len // self.hparams.ba_size
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.hparams.lr,
            epochs=self.hparams.epochs,
            steps_per_epoch=steps_per_epoch,
        )
        return [optimizer], [scheduler]


def main(args: argparse.Namespace) -> None:

    lyft_3d_utils.set_random_seed()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_gpus

    # ===== Configure LYFT dataset
    val_hosts = lyft_3d_utils.get_val_hosts(args.val_hosts)

    # mypy error due to pl.DataModule.transfer_batch_to_device
    det_dm = Lyft3DdetSegDatamodule(  # type: ignore[abstract]
        args.bev_data_dir,
        val_hosts=val_hosts,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        is_debug=args.is_debug,
    )
    det_dm.prepare_data()
    det_dm.setup(stage="test" if args.is_test else "fit")
    if args.is_test:
        print("\t\t ==== TEST MODE ====")
        print("load from: ", args.ckpt_path)
        model = LitModel.load_from_checkpoint(args.ckpt_path)
        trainer = pl.Trainer(gpus=len(args.visible_gpus.split(",")))
        trainer.test(model, datamodule=det_dm)

        test_gt_path = os.path.join(os.path.dirname(det_dm.test_path), "gt.csv")
        if os.path.exists(test_gt_path):
            print("test mode with validation chopped dataset, and check the metrics")
            print("validation ground truth path: ", test_gt_path)

    else:
        print("\t\t ==== TRAIN MODE ====")
        print(
            "training samples: {}, valid samples: {}".format(
                len(det_dm.train_dataset), len(det_dm.val_dataset)
            )
        )

        model = LitModel(
            det_dm.bev_config,
            val_hosts,
            len(det_dm.train_dataset),
            lr=args.lr,
            backbone_name=args.backbone_name,
            optim_name=args.optim_name,
            ba_size=args.batch_size,
            epochs=args.epochs,
        )

        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            monitor="val_loss",
            save_last=True,
            mode="min",
            verbose=True,
        )
        pl.trainer.seed_everything(seed=SEED)
        trainer = pl.Trainer(
            gpus=len(args.visible_gpus.split(",")),
            max_epochs=args.epochs,
            precision=args.precision,
            benchmark=True,
            deterministic=False,
            checkpoint_callback=checkpoint_callback,
        )

        # Run lr finder
        if args.find_lr:
            lr_finder = trainer.tuner.lr_find(model, datamodule=det_dm)
            lr_finder.plot(suggest=True)
            plt.show()
            sys.exit()

        # Run Training
        trainer.fit(model, datamodule=det_dm)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run training for lyft 3d detection with bev image",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--bev_data_dir",
        default="/your/dataset/path",
        type=str,
        help="root directory path for bev, ",
    )
    parser.add_argument(
        "--optim_name",
        choices=["adam", "sgd"],
        default="adam",
        help="optimizer name",
    )
    parser.add_argument("--lr", default=3.0e-3, type=float, help="learning rate")
    parser.add_argument("--batch_size", type=int, default=96, help="batch size")
    parser.add_argument("--epochs", type=int, default=50, help="epochs for training")
    parser.add_argument(
        "--backbone_name",
        choices=["efficientnet-b1", "seresnext26d_32x4d"],
        default="efficientnet-b1",
        help="backbone name",
    )
    parser.add_argument("--is_test", action="store_true", help="test mode")
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="./model.pth",
        help="path for model checkpoint at test mode",
    )
    parser.add_argument(
        "--precision",
        default=16,
        choices=[16, 32],
        type=int,
        help="float precision at training",
    )
    parser.add_argument(
        "--val_hosts",
        default=0,
        choices=[0, 1, 2, 3],
        type=int,
        help="validation hosts configuration for train/val split",
    )
    parser.add_argument(
        "--visible_gpus",
        type=str,
        default="0",
        help="Select gpu ids with comma separated format",
    )
    parser.add_argument(
        "--find_lr",
        action="store_true",
        help="find lr with fast ai implementation",
    )
    parser.add_argument(
        "--num_workers",
        default="16",
        type=int,
        help="number of cpus for DataLoader",
    )
    parser.add_argument("--is_debug", action="store_true", help="debug mode")

    args = parser.parse_args()

    if args.is_debug:
        DEBUG = True
        print("\t ---- DEBUG RUN ---- ")
        VAL_INTERVAL_SAMPLES = 5000
        args.batch_size = 16
    else:
        DEBUG = False
        print("\t ---- NORMAL RUN ---- ")
    lyft_3d_utils.print_argparse_arguments(args)

    main(args)
