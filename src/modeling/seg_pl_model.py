import json
import math
import os
from typing import List, Tuple, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch
from omegaconf.dictconfig import DictConfig
from omegaconf.listconfig import ListConfig
from tqdm import tqdm

from src.bev_processing.bev_to_3d import (
    calc_detection_box,
    convert_into_nuscene_3dbox,
    create_3d_boxes_from_2d,
)
from src.config.config import CLASS_LOSS_WEIGHTS, CLASSES, CSV_NAME
from src.dataset.seg_datamodule import IMG_MEAN, IMG_STD


class LitModel(pl.LightningModule):
    def __init__(
        self,
        bev_config: Union[DictConfig, ListConfig],
        val_hosts: Tuple[str, ...],
        dataset_len: int,
        ba_size: int = 64,
        lr: float = 3.0e-4,
        aug_mode: int = 0,
        class_weights: List[float] = CLASS_LOSS_WEIGHTS,
        backbone_name: str = "efficientnet-b2",
        epochs: int = 50,
        optim_name: str = "adam",
        in_channels: int = 3,
        output_dir: str = "./",
        is_debug: bool = False,
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
        self.is_debug = is_debug

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

            self._set_image_normalization()
            inputs_rgb = torch.cat(
                [inputs[i] * self._img_std + self._img_mean for i in range(num_)], dim=1
            )
            # (3, H*num_, W)*2 -> (3, H*num_, W*2)
            img = torch.cat((inputs_rgb, class_rgb), dim=2)
            self.logger.experiment.add_image(
                "prediction",
                img.type(torch.float32).cpu(),
                global_step=self.global_step,
            )
        return loss

    def _set_image_normalization(self) -> None:
        self._img_std = torch.tensor(
            np.array(IMG_STD)[:, None, None], device=self.device
        )
        self._img_mean = torch.tensor(
            np.array(IMG_MEAN)[:, None, None], device=self.device
        )

    def test_step(self, batch, batch_idx):
        inputs = batch["image"]
        outputs = self.model(inputs)

        outputs = outputs.softmax(dim=1)
        predictions = np.round(outputs.cpu().numpy() * 255).astype(np.uint8)
        predictions = np.transpose(predictions, (0, 2, 3, 1))

        global_from_voxel = batch["global_from_voxel"].cpu().numpy()
        ego_translation = batch["ego_translation"].cpu().numpy()

        pred_box3ds = []

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
            non_bkg_proba = 255 - raw_pred[:, :, 0]
            thresholded_p = (non_bkg_proba > self.background_threshold).astype(np.uint8)

            predictions_opened = cv2.morphologyEx(
                thresholded_p, cv2.MORPH_OPEN, self.kernel
            )
            # get all 2D boxes from predicted image
            (
                sample_boxes,
                sample_detection_scores,
                sample_detection_classes,
            ) = calc_detection_box(predictions_opened, raw_pred, CLASSES)
            if self.is_debug:
                t = np.zeros_like(predictions_opened)
                for sample_b in sample_boxes:
                    box_pix = np.int0(sample_b)
                    cv2.drawContours(t, [box_pix], 0, (255), 2)
                plt.imshow(np.hstack([predictions_opened * 255, t]))
                plt.show()
            # store the all boxes on the list
            (
                sample_boxes,
                sample_boxes_centers,
                sample_boxes_dimensions,
            ) = create_3d_boxes_from_2d(
                np.array(sample_boxes),
                sample_detection_classes,
                ego_translation[i],
                global_from_voxel[i],
                box_scale=self.hparams.bev_config.box_scale,
            )
            pred_box3ds.extend(
                convert_into_nuscene_3dbox(
                    batch["sample_token"][i],
                    sample_boxes,
                    sample_boxes_centers,
                    sample_boxes_dimensions,
                    sample_detection_classes,
                    sample_detection_scores,
                    is_debug=self.is_debug,
                )
            )
        return pred_box3ds

    def test_epoch_end(self, outputs):
        # convert into world coordinates and compute offsets
        pred_box3ds = []
        for boxes in outputs:
            pred_box3ds.extend(boxes)
        pred = [b.serialize() for b in pred_box3ds]
        pred_json_path = os.path.join(
            self.hparams.output_dir, CSV_NAME.replace(".csv", ".json")
        )
        with open(pred_json_path, "w") as f:
            json.dump(pred, f)

        sub = {}
        for i in tqdm(range(len(pred_box3ds))):
            yaw = 2 * np.arccos(pred_box3ds[i].rotation[0])
            yaw = math.copysign(yaw, pred_box3ds[i].rotation[-1])
            pred = " ".join(
                [
                    str(pred_box3ds[i].score / 255),
                    str(pred_box3ds[i].center_x),
                    str(pred_box3ds[i].center_y),
                    str(pred_box3ds[i].center_z),
                    str(pred_box3ds[i].width),
                    str(pred_box3ds[i].length),
                    str(pred_box3ds[i].height),
                    str(yaw),
                    str(pred_box3ds[i].name),
                    " ",
                ]
            )[:-1]
            if pred_box3ds[i].sample_token in sub.keys():
                sub[pred_box3ds[i].sample_token] += pred
            else:
                sub[pred_box3ds[i].sample_token] = pred

        sample_sub = pd.read_csv(
            os.path.join(
                self.hparams.bev_config.dataset_root_dir, "sample_submission.csv"
            )
        )
        for token in set(sample_sub.Id.values).difference(sub.keys()):
            sub[token] = ""

        sub = pd.DataFrame(list(sub.items()))
        sub.columns = sample_sub.columns
        sub_csv_path = os.path.join(self.hparams.output_dir, CSV_NAME)
        sub.to_csv(sub_csv_path, index=False)
        print("save submission file on:", sub_csv_path)

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
