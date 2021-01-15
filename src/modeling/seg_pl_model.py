from src.dataset.seg_datamodule import IMG_MEAN, IMG_STD
from typing import List, Tuple, Union

import cv2
import numpy as np
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch

from make_image import transform_points
from src.bev_processing.bev_to_3d import calc_detection_box
from src.config.config import CLASS_AVG_HEIGHTS, CLASS_LOSS_WEIGHTS, CLASSES
from omegaconf.dictconfig import DictConfig
from omegaconf.listconfig import ListConfig


class LitModel(pl.LightningModule):
    def __init__(
        self,
        bev_config: Union[DictConfig, ListConfig],
        val_hosts: Tuple[str, ...],
        dataset_len: int,
        ba_size: int = 128,
        lr: float = 3.0e-4,
        aug_mode: int = 0,
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
                sample_detection_classes,
                ego_pose,
                global_from_voxel,
                box_scale=self.hparams.bev_config.box_scale,
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

        # We don't know at where the boxes are in the scene on the z-axis (up-down),
        # let's assume all of them are at the same height as the ego vehicle.
        sample_boxes[2, :] = ego_pose["translation"][2]
        # (3, N*4) -> (N, 4, 3)
        sample_boxes = sample_boxes.transpose(1, 0).reshape(-1, 4, 3)

        # We don't know the height of our boxes, let's assume every object is the same
        # height.
        # box_height = 1.75
        box_height = np.array(
            [CLASS_AVG_HEIGHTS[cls_] for cls_ in sample_detection_classes]
        )

        # Note: Each of these boxes describes the ground corners of a 3D box.
        # To get the center of the box in 3D, we'll have to add half the height to it.
        sample_boxes_centers = sample_boxes.mean(axis=1)  # (N, 3)
        sample_boxes_centers[:, 2] += box_height / 2

        # Width and height is arbitrary - we don't know what way the vehicles are
        # pointing from our prediction segmentation It doesn't matter for evaluation,
        # so no need to worry about that here. Note: We scaled our targets to be 0.8
        # the actual size, we need to adjust for that
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
