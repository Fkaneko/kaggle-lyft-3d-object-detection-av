#!/usr/bin/env python
# coding: utf-8
import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pytorch_lightning as pl

from src.config.config import SEED
from src.dataset.seg_datamodule import Lyft3DdetSegDatamodule
from src.modeling.seg_pl_model import LitModel
from src.utils.util import print_argparse_arguments, set_random_seed


def main(args: argparse.Namespace) -> None:

    set_random_seed(SEED)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_gpus

    # mypy error due to pl.DataModule.transfer_batch_to_device
    det_dm = Lyft3DdetSegDatamodule(  # type: ignore[abstract]
        args.bev_data_dir,
        val_hosts=args.val_hosts,
        batch_size=args.batch_size,
        aug_mode=args.aug_mode,
        num_workers=args.num_workers,
        is_debug=args.is_debug,
    )

    det_dm.prepare_data()
    if args.is_test:
        det_dm.setup(stage="test" if not args.test_with_val else "fit")
        print("\t\t ==== TEST MODE ====")
        print("load from: ", args.ckpt_path)
        model = LitModel.load_from_checkpoint(
            args.ckpt_path,
            output_dir=str(Path(args.ckpt_path).parent),
            flip_tta=args.flip_tta,
            background_threshold=args.background_threshold,
        )
        # Check the image resolution. Train and test bev resolution should be the same.
        assert model.hparams.bev_config.voxel_size_xy == det_dm.bev_config.voxel_size_xy
        assert model.hparams.bev_config.voxel_size_z == det_dm.bev_config.voxel_size_z
        assert model.hparams.bev_config.box_scale == det_dm.bev_config.box_scale

        # Image size can be different between training and test.
        model.hparams.bev_config.image_size = det_dm.bev_config.image_size

        trainer = pl.Trainer(gpus=len(args.visible_gpus.split(",")))

        if args.test_with_val:
            trainer.test(model, test_dataloaders=det_dm.val_dataloader())
        else:
            trainer.test(model, datamodule=det_dm)

        # test_gt_path = os.path.join(os.path.dirname(det_dm.test_path), "gt.csv")
        # if os.path.exists(test_gt_path):
        #     print("test mode with validation chopped dataset, and check the metrics")
        #     print("validation ground truth path: ", test_gt_path)

    else:
        print("\t\t ==== TRAIN MODE ====")
        print(
            "training samples: {}, valid samples: {}".format(
                len(det_dm.train_dataset), len(det_dm.val_dataset)
            )
        )

        model = LitModel(
            det_dm.bev_config,
            det_dm.val_hosts,
            len(det_dm.train_dataset),
            lr=args.lr,
            aug_mode=args.aug_mode,
            backbone_name=args.backbone_name,
            optim_name=args.optim_name,
            ba_size=args.batch_size,
            epochs=args.epochs,
        )

        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            monitor="val_f1",
            save_last=True,
            mode="max",
            verbose=True,
        )
        pl.trainer.seed_everything(seed=SEED)
        trainer = pl.Trainer(
            resume_from_checkpoint=args.resume_from_checkpoint,
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
    parser.add_argument("--lr", default=1.0e-2, type=float, help="learning rate")
    parser.add_argument("--batch_size", type=int, default=96, help="batch size")
    parser.add_argument("--epochs", type=int, default=50, help="epochs for training")
    parser.add_argument(
        "--backbone_name",
        choices=[
            "efficientnet-b1",
            "efficientnet-b2",
            "timm-resnest50d",
            "timm-resnest269e",
        ],
        default="timm-resnest50d",
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
        "--test_with_val", action="store_true", help="test mode with validation data"
    )
    parser.add_argument(
        "--flip_tta", action="store_true", help="test time augmentation h/vflip"
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
        "--aug_mode",
        default=0,
        choices=[0, 1],
        type=int,
        help="augmentation mode",
    )
    parser.add_argument(
        "--background_threshold",
        default=200,
        type=int,
        help="background threshold for 2d predicted mask, only used at test mode",
    )
    parser.add_argument(
        "--visible_gpus",
        type=str,
        default="0",
        help="Select gpu ids with comma separated format",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="resume training from a specific checkpoint path",
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
    print_argparse_arguments(args)
    main(args)
