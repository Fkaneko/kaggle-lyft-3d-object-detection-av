import argparse
import json
import os
from pathlib import Path
from typing import List

from lyft_dataset_sdk.lyftdataset import LyftDataset
from tqdm import tqdm

from src.bev_processing.bev_utils import LyftBEVFolderConfiguration, SampleMeta
from src.bev_processing.points_to_bev import convert_scene_into_bev_images
from src.config.config import DATSET_LINKS_DIR, GT_JSON_NAME, INPUT_META_JSON_NAME
from src.evaluation.mAP_evaluation import Box3D
from src.utils.util import (
    make_dataset_links,
    make_version_folder,
    save_config_data,
    set_random_seed,
)


def main(args):
    set_random_seed()
    # Disable multiprocesing for numpy/opencv. We already multiprocess ourselves,
    # this would mean every subprocess produces  even more threads which would lead
    # to a lot of context switching, slowing things down a lot.
    os.environ["OMP_NUM_THREADS"] = "1"

    # Some hyperparameters we'll need to define for the system, to generate BEV
    # "bev" stands for birds eye view
    voxel_size = (args.voxel_size_xy, args.voxel_size_xy, args.voxel_size_z)
    z_offset = args.z_offset
    bev_shape = (args.image_size, args.image_size, args.image_channel)
    # We scale down each box so they are more separated when projected into our coarse
    # voxel space.
    box_scale = args.box_scale

    dataset_root = Path(args.dataset_root_dir)

    # output directory configuration
    output_dir = make_version_folder(Path(args.output_dir))
    save_config_data(args, output_dir / Path("config.yaml"))

    data_prefix = "test" if args.is_test else "train"
    make_dataset_links(dataset_root, data_prefix)
    level5data = LyftDataset(
        data_path=str(Path(DATSET_LINKS_DIR, data_prefix)),
        json_path=str(dataset_root.joinpath(data_prefix + "_data")),
        verbose=True,
    )
    folder_configuration = LyftBEVFolderConfiguration(
        level5data, output_dir, data_prefix
    )
    folder_configuration.extract_scene_info()
    df_and_folder_list = folder_configuration.create_folders()
    map_mask = level5data.map[0]["mask"]

    for data_folder, df in df_and_folder_list:
        print("Generating BEV images from lidar 3d points on {}".format(data_folder))
        first_samples = df.first_sample_token.values
        hosts = df.host.values
        sample_meta: List[SampleMeta] = []
        gt: List[Box3D] = []
        for i in tqdm(range(first_samples.shape[0])):
            sample_meta_data, gt_box3ds = convert_scene_into_bev_images(
                first_samples[i],
                hosts[i],
                level5data=level5data,
                output_folder=data_folder,
                test_mode=args.is_test,
                debug_mode=args.is_debug,
                bev_shape=bev_shape,
                map_mask=map_mask,
                voxel_size=voxel_size,
                z_offset=z_offset,
                box_scale=box_scale,
                max_intensity=args.max_intensity,
            )
            if args.is_debug:
                if i > 5:
                    break
            sample_meta.extend(sample_meta_data)
            gt.extend(gt_box3ds)

        # making input meta json for training
        sample_meta = [meta_._asdict() for meta_ in sample_meta]
        with open(Path(data_folder, INPUT_META_JSON_NAME), "w") as f:
            json.dump(sample_meta, f)

        if not args.is_test:
            # making ground truth json file for mAP evaluation
            gt = [b.serialize() for b in gt]
            with open(Path(data_folder, GT_JSON_NAME), "w") as f:
                json.dump(gt, f)

        # process_func = partial(
        #     prepare_training_data_for_scene,
        #     level5data=level5data,
        #     output_folder=data_folder,
        #     bev_shape=bev_shape,
        #     map_mask=map_mask,
        #     voxel_size=voxel_size,
        #     z_offset=z_offset,
        #     box_scale=box_scale,
        #     max_intensity=args.max_intensity,
        #     test_mode=args.is_test,
        # )
        # pool = Pool(NUM_WORKERS)
        # for _ in tqdm(
        #     pool.imap_unordered(process_func, first_samples), total=len(first_samples)
        # ):
        #     pass
        # pool.close()


if __name__ == "__main__":
    ####################################################################################
    # Setting arguments
    ####################################################################################
    parser = argparse.ArgumentParser(
        description="making simple BEV image for lyft 3d object detection",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./BEV",
        help="directory for generated bev images",
    )
    parser.add_argument(
        "--dataset_root_dir",
        type=str,
        default="dataset root directory",
        help="directory for bev images",
    )
    parser.add_argument(
        "--voxel_size_xy",
        type=float,
        default=0.4,
        help="voxel unit size in meter for x and y",
    )
    parser.add_argument(
        "--voxel_size_z",
        type=float,
        default=1.5,
        help="voxel unit size in meter for z",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=336,
        help="BEV image size",
    )
    parser.add_argument(
        "--image_channel",
        type=int,
        default=3,
        help="input image channel",
    )
    parser.add_argument(
        "--box_scale",
        type=float,
        default=0.8,
        help="box scale factor",
    )
    parser.add_argument(
        "--max_intensity",
        type=int,
        default=16,
        help="max point intensity for each voxel",
    )
    parser.add_argument(
        "--z_offset",
        type=float,
        default=-2.0,
        help="z ground offset at voxel definition",
    )
    parser.add_argument(
        "--is_test",
        action="store_true",
        help="test mode image generation",
    )
    parser.add_argument(
        "--is_debug",
        action="store_true",
        help="debug mode, only few scenes are processed",
    )
    # parser.add_argument(
    #     "--workers",
    #     default=4,
    #     type=int,
    #     help="number of data loading workers",
    # )
    args = parser.parse_args()
    main(args)
