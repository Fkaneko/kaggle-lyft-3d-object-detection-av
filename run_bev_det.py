#!/usr/bin/env python
# coding: utf-8
import argparse
import os
import sys

import matplotlib.pyplot as plt
import pytorch_lightning as pl

from src.config.config import CSV_PATH, SEED
from src.dataset.seg_datamodule import Lyft3DdetSegDatamodule
from src.modeling.seg_pl_model import LitModel
from src.utils.util import set_random_seed, print_argparse_arguments


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
        "--aug_mode",
        default=0,
        choices=[0, 1],
        type=int,
        help="augmentation mode",
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
    print_argparse_arguments(args)
    main(args)
