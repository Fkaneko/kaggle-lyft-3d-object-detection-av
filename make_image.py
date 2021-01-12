import argparse
import os
import json
from datetime import datetime
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from typing import List, Tuple, NamedTuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lyft_dataset_sdk.lyftdataset import LyftDataset
from lyft_dataset_sdk.utils.data_classes import LidarPointCloud, Quaternion
from lyft_dataset_sdk.utils.geometry_utils import transform_matrix
from lyft_dataset_sdk.eval.detection.mAP_evaluation import Box3D
from PIL import Image
from tqdm import tqdm

import lyft_3d_utils
from lyft_3d_utils import (
    BEV_MAP_SUFFIX,
    BEV_TARGET_SUFFIX,
    BEV_TRAIN_SUFFIX,
    make_version_folder,
    save_config_data,
    set_random_seed,
    SampleMeta,
    INPUT_META_JSON_NAME,
    GT_JSON_NAME,
    CLASSES,
)


def load_groundtruth_boxes(level5data: LyftDataset, sample_tokens: str) -> List[Box3D]:
    """load ground  truth with nuscenes Box3D format

    Args:
        level5data (LyftDataset): dataset contains ground truth
        sample_tokens (str): target samples

    Returns:
        List[Box3D]: ground truth box with nuscenes format
    """
    gt_box3ds = []
    for sample_token in tqdm(sample_tokens):
        gt_box3ds.extend(make_gt_boxes_from_sample(level5data, sample_token))
    return gt_box3ds


def make_gt_boxes_from_sample(
    level5data: LyftDataset, sample_token: str
) -> List[Box3D]:
    """load ground  truth from LyftDataset and convert it into nuscenes Box3D format

    Args:
        level5data (LyftDataset): dataset contains ground truth
        sample_token (str): target sample

    Returns:
        List[Box3D]: ground truth box with nuscenes format
    """
    sample = level5data.get("sample", sample_token)
    sample_annotation_tokens = sample["anns"]
    gt_box3ds = []
    for sample_annotation_token in sample_annotation_tokens:
        sample_annotation = level5data.get("sample_annotation", sample_annotation_token)
        sample_annotation_translation = sample_annotation["translation"]
        class_name = sample_annotation["category_name"]
        box3d = Box3D(
            sample_token=sample_token,
            translation=sample_annotation_translation,
            size=sample_annotation["size"],
            rotation=sample_annotation["rotation"],
            name=class_name,
        )
        gt_box3ds.append(box3d)
    return gt_box3ds


def get_semantic_map_around_ego(map_mask, ego_pose, voxel_size, output_shape):
    def crop_image(
        image: np.array, x_px: int, y_px: int, axes_limit_px: int
    ) -> np.array:
        x_min = int(x_px - axes_limit_px)
        x_max = int(x_px + axes_limit_px)
        y_min = int(y_px - axes_limit_px)
        y_max = int(y_px + axes_limit_px)

        cropped_image = image[y_min:y_max, x_min:x_max]

        return cropped_image

    # Pixel position from 40000x40000
    pixel_coords = map_mask.to_pixel_coords(
        ego_pose["translation"][0], ego_pose["translation"][1]
    )

    # Requared half map: size*0.5 at global coordinate, "mm".
    extent = voxel_size * output_shape[0] * 0.5

    # Requared map size*0.5 with pixel, "pixel".
    scaled_limit_px = int(extent * (1.0 / (map_mask.resolution)))
    mask_raster = map_mask.mask()  # 40000x40000
    # Keep larger pixel than requared size, scaled_limit_px, for rotation
    cropped = crop_image(
        mask_raster, pixel_coords[0], pixel_coords[1], int(scaled_limit_px * np.sqrt(2))
    )

    ypr_rad = Quaternion(ego_pose["rotation"]).yaw_pitch_roll
    # z axis rotation angle, minus is necessary.
    yaw_deg = -np.degrees(ypr_rad[0])
    rotated_cropped = np.array(Image.fromarray(cropped).rotate(yaw_deg))
    # Be careful pixel and x, y coordinate corresponding rule pixel[0],y and pixel[1],x
    # And y flip with [::-1]

    # TODO: INPUT-MAP, Why y flip is required? yes, increase the y coord corresponding
    # to decrease of the y pixel at the original map <DONE>

    ego_centric_map = crop_image(
        rotated_cropped,
        rotated_cropped.shape[1] / 2,
        rotated_cropped.shape[0] / 2,
        scaled_limit_px,
    )[::-1]

    ego_centric_map = cv2.resize(ego_centric_map, output_shape[:2], cv2.INTER_NEAREST)
    return ego_centric_map.astype(np.float32) / 255


def create_transformation_matrix_to_voxel_space(
    shape: Tuple[int, int, int] = (336, 336, 3),
    voxel_size: Tuple[float, float, float] = (0.4, 0.4, 1.5),
    offset: Tuple[float, float, float] = (0.0, 0.0, -2.0),
) -> np.ndarray:
    """
    Constructs a transformation matrix given an output voxel shape such that (0,0,0)
    ends up in the center. Voxel_size defines how large every voxel is in world
    coordinate, (1,1,1) would be the same as Minecraft voxels.
    An offset per axis in world coordinates (metric) can be provided, this is useful
    for Z (up-down) in lidar points.
    """
    shape, voxel_size, offset = np.array(shape), np.array(voxel_size), np.array(offset)
    # 4x4 Diagnal matrix
    tm = np.eye(4, dtype=np.float32)
    # Make the center of voxel space (0mm, 0mm, 0mm) when no offset
    # Make the center of voxel space (0mm, 0mm,-offset mm) with offset

    translation = shape / 2 + offset / voxel_size

    # Scaling
    tm = tm * np.array(np.hstack((1 / voxel_size, [1])))

    tm[:3, 3] = np.transpose(translation)
    return tm


def transform_points(points, transf_matrix):
    """
    Transform (3,N) or (4,N) points using transformation matrix.
    """
    if points.shape[0] not in [3, 4]:
        raise Exception(
            "Points input should be (3,N) or (4,N) shape, received {}".format(
                points.shape
            )
        )
    return transf_matrix.dot(np.vstack((points[:3, :], np.ones(points.shape[1]))))[
        :3, :
    ]


def car_to_voxel_coords(
    points: np.ndarray,
    shape: Tuple[int, int, int] = (336, 336, 3),
    voxel_size: Tuple[float, float, float] = (0.4, 0.4, 1.5),
    z_offset: float = -2.0,
) -> np.ndarray:
    if len(shape) != 3:
        raise Exception("Voxel volume shape should be 3 dimensions (x,y,z)")

    if len(points.shape) != 2 or points.shape[0] not in [3, 4]:
        raise Exception(
            "Input points should be (3,N) or (4,N) in shape, found {}".format(
                points.shape
            )
        )

    tm = create_transformation_matrix_to_voxel_space(
        shape, voxel_size, (0, 0, z_offset)
    )
    p = transform_points(points, tm)
    return p


def create_voxel_pointcloud(
    points: np.ndarray,
    shape: Tuple[int, int, int] = (336, 336, 3),
    voxel_size: Tuple[float, float, float] = (0.4, 0.4, 1.5),
    z_offset: float = -2.0,
    int_first: bool = True,
) -> np.ndarray:
    """create bev from pointcolud

    Args:
        points (np.ndarray): lidar points
        shape (Tuple[int, int, int], optional): bev shape. Defaults to (336, 336, 3).
        voxel_size (Tuple[float, float, float], optional): voxel saize in meter.
            Defaults to (0.4, 0.4, 1.5).
        z_offset (float, optional): z offset. Defaults to -2.0.
        int_first (bool, optional): index timing control. Defaults to True.

    Returns:
        np.ndarray: bev image with the shape
    """
    points_voxel_coords = car_to_voxel_coords(
        points.copy(), shape, voxel_size, z_offset
    )
    # 3xN -> Nx3
    points_voxel_coords = points_voxel_coords[:3].transpose(1, 0)
    if int_first:
        points_voxel_coords = np.int0(points_voxel_coords)

    bev = np.zeros(shape, dtype=np.float32)
    bev_shape = np.array(shape)

    # mask for within bev_shape or voxel scope <337 and >=0
    within_bounds = np.all(points_voxel_coords >= 0, axis=1) * np.all(
        points_voxel_coords < bev_shape, axis=1
    )
    # Nx3 -> Mx3 and M, M < N
    points_voxel_coords = points_voxel_coords[within_bounds]

    # same resolution for edge pixcels
    if not int_first:
        points_voxel_coords = np.int0(points_voxel_coords)
    coord, count = np.unique(points_voxel_coords, axis=0, return_counts=True)

    # Note X and Y are flipped:
    bev[coord[:, 1], coord[:, 0], coord[:, 2]] = count

    return bev


def normalize_voxel_intensities(bev: np.ndarray, max_intensity: int = 16) -> np.ndarray:
    return (bev / max_intensity).clip(0, 1)


def move_boxes_to_car_space(boxes, ego_pose) -> None:
    """
    Move boxes from world space to car space.
    Note: mutates input boxes.
    """
    translation = -np.array(ego_pose["translation"])
    rotation = Quaternion(ego_pose["rotation"]).inverse

    for box in boxes:
        # Bring box to car space
        box.translate(translation)
        box.rotate(rotation)


def scale_boxes(boxes, factor) -> None:
    """
    Note: mutates input boxes
    """
    for box in boxes:
        box.wlh = box.wlh * factor


def draw_boxes(im, voxel_size, boxes, classes, z_offset=0.0, box_scale=0.8):
    for box in boxes:
        # We only care about the bottom corners
        corners = box.bottom_corners()  # (3, 4)

        # z_bottom = np.mean(corners, axis=1)[2]
        height = box.wlh[2] * 1.0 / box_scale
        z_bottom = box.center[2] - height / 2

        corners_voxel = car_to_voxel_coords(
            corners, im.shape, voxel_size, z_offset
        ).transpose(1, 0)
        corners_voxel = corners_voxel[:, :2]  # Drop z coord

        class_color = classes.index(box.name) + 1
        if class_color == 0:
            raise Exception("Unknown class: {}".format(box.name))

        # for advanced study, not used at default for these two values
        z_bottom = int(np.clip((z_bottom + 3.0) / 6.0 * 255, 0, 255))
        height = int(np.clip(height / 6.0 * 255, 0, 255))

        cv2.drawContours(
            im, np.int0([corners_voxel]), 0, (class_color, z_bottom, height), -1
        )


def prepare_training_data_for_scene(
    first_sample_token: str,
    host: str,
    level5data: LyftDataset,
    output_folder: Path,
    map_mask: np.ndarray,
    test_mode: bool,
    debug_mode: bool,
    bev_shape: Tuple[int, int, int] = (336, 336, 3),
    voxel_size: Tuple[float, float, float] = (0.4, 0.4, 1.5),
    z_offset: float = -2.0,
    box_scale: float = 0.8,
    max_intensity: int = 16,
) -> Tuple[List[SampleMeta], List[Box3D]]:
    """
    Given a first sample token (in a scene), output rasterized input volumes and
    targets in birds-eye-view perspective.
    """
    sample_token = first_sample_token
    gt_box3ds: List[Box3D] = []
    sample_meta_data: List[SampleMeta] = []

    while sample_token:
        # extract necessary info from dataset
        sample = level5data.get("sample", sample_token)
        sample_lidar_token = sample["data"]["LIDAR_TOP"]
        lidar_data = level5data.get("sample_data", sample_lidar_token)
        lidar_filepath = level5data.get_sample_data_path(sample_lidar_token)

        calibrated_sensor = level5data.get(
            "calibrated_sensor", lidar_data["calibrated_sensor_token"]
        )
        ego_pose = level5data.get("ego_pose", lidar_data["ego_pose_token"])

        car_from_sensor = transform_matrix(
            calibrated_sensor["translation"],
            Quaternion(calibrated_sensor["rotation"]),
            inverse=False,
        )
        try:
            lidar_pointcloud = LidarPointCloud.from_file(lidar_filepath)
            lidar_pointcloud.transform(car_from_sensor)
        except Exception as e:
            print("Failed to load Lidar Pointcloud for {}: {}:".format(sample_token, e))
            sample_token = sample["next"]
            continue

        # create bev input
        # (336x336x3) (y, x, z)
        bev = create_voxel_pointcloud(
            lidar_pointcloud.points,
            bev_shape,
            voxel_size=voxel_size,
            z_offset=z_offset,
        )
        # (336x336x3) (y, x, z) [0, 1]
        bev = normalize_voxel_intensities(bev, max_intensity=max_intensity)
        bev_im = np.round(bev * 255).astype(np.uint8)
        cv2.imwrite(
            os.path.join(
                output_folder, host, "{}_{}.png".format(sample_token, BEV_TRAIN_SUFFIX)
            ),
            cv2.cvtColor(bev_im, cv2.COLOR_RGB2BGR),
        )
        # extract map
        semantic_im = get_semantic_map_around_ego(
            map_mask, ego_pose, voxel_size[0], (bev.shape[0], bev.shape[1])
        )
        semantic_im = np.round(semantic_im * 255).astype(np.uint8)
        cv2.imwrite(
            os.path.join(
                output_folder, host, "{}_{}.png".format(sample_token, BEV_MAP_SUFFIX)
            ),
            cv2.cvtColor(semantic_im, cv2.COLOR_RGB2BGR),
        )

        global_from_car = transform_matrix(
            ego_pose["translation"], Quaternion(ego_pose["rotation"]), inverse=False
        )
        car_from_voxel = np.linalg.inv(
            create_transformation_matrix_to_voxel_space(
                bev_shape, voxel_size, (0, 0, z_offset)
            )
        )
        global_from_voxel = np.dot(global_from_car, car_from_voxel).tolist()
        sample_meta_data.append(
            SampleMeta(sample_token, host, ego_pose, global_from_voxel)
        )
        if debug_mode:
            plt.figure(figsize=(16, 8))
            plt.imshow(np.hstack((bev_im, semantic_im)))
            plt.show()
            plt.close()

        # extract annotation and create bev tareget
        if not test_mode:
            # (3xN)
            boxes = level5data.get_boxes(sample_lidar_token)
            target = np.zeros_like(bev[:, :, :3])

            # change the frame from global to car
            move_boxes_to_car_space(boxes, ego_pose)
            # scale boxes for low resolution image
            scale_boxes(boxes, box_scale)
            # (336x336x3) (y, x, class_color) no z information
            draw_boxes(
                target,
                voxel_size,
                boxes=boxes,
                classes=CLASSES,
                z_offset=z_offset,
            )
            cv2.imwrite(
                os.path.join(
                    output_folder,
                    host,
                    "{}_{}.png".format(sample_token, BEV_TARGET_SUFFIX),
                ),
                target[:, :, 0],
            )
            if debug_mode:
                plt.figure(figsize=(8, 8))
                plt.imshow((target[:, :, 0] > 0).astype(np.float32), cmap="Set2")
                plt.show()
                # These are the annotations in the same top-down frame, Below we plot
                # the same scene using the NuScenes SDK. Don't worry about it being
                # flipped.
                plt.close()
                level5data.render_sample_data(sample_lidar_token, axes_limit=80)
            # for mAP evaluation
            gt_box3ds.extend(make_gt_boxes_from_sample(level5data, sample_token))

        sample_token = sample["next"]

    return sample_meta_data, gt_box3ds


class Lyft3dFolderConfigure:
    def __init__(
        self, level5data: LyftDataset, output_dir: Path, data_prefix: str
    ) -> None:
        """make folder for each host from LyftDataset

        Args:
            level5data (LyftDataset): lidar dataset with meta data
            output_dir (Path): output root directory for generated bev images
        """
        self.output_dir = output_dir
        self.level5data = level5data
        self.data_prefix = data_prefix

    def extract_scene_info(self) -> None:
        """create a pandas dataframe with a row for each of the scenes"""
        records = [
            (self.level5data.get("sample", rec["first_sample_token"])["timestamp"], rec)
            for rec in self.level5data.scene
        ]

        entries = []
        for start_time, record in sorted(records):
            start_time = (
                self.level5data.get("sample", record["first_sample_token"])["timestamp"]
                / 1000000
            )

            token = record["token"]
            name = record["name"]
            date = datetime.utcfromtimestamp(start_time)
            host = "-".join(record["name"].split("-")[:2])
            first_sample_token = record["first_sample_token"]

            entries.append((host, name, date, token, first_sample_token))

        self.df = pd.DataFrame(
            entries,
            columns=["host", "scene_name", "date", "scene_token", "first_sample_token"],
        )
        host_count_df = self.df.groupby("host")["scene_token"].count()
        print("the number of host", host_count_df)

    def create_folders(self) -> List[Tuple[Path, pd.DataFrame]]:
        df_and_folder_list = []
        df_and_folder_list.append(self._create_folders(self.data_prefix + "_images"))
        return df_and_folder_list

    def _create_folders(self, name: str) -> Tuple[Path, pd.DataFrame]:
        """make output directory for each host and return with corresponding dataframe
        Args:
            df (pd.DataFrame): dataframe from level5data
            name (str):  folder name
        """
        data_folder = self.output_dir / Path(name)
        data_folder.mkdir(exist_ok=True)
        for host_name in self.df["host"].unique():
            Path(data_folder, host_name).mkdir(exist_ok=True)
        df_and_folder_list = (data_folder, self.df)
        return df_and_folder_list


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

    data_prefix = "test" if args.test_mode else "train"
    lyft_3d_utils.make_dataset_links(dataset_root, data_prefix)
    level5data = LyftDataset(
        data_path=str(Path(lyft_3d_utils.DATSET_LINKS_DIR, data_prefix)),
        json_path=str(dataset_root.joinpath(data_prefix + "_data")),
        verbose=True,
    )
    folder_configure = Lyft3dFolderConfigure(level5data, output_dir, data_prefix)
    folder_configure.extract_scene_info()
    df_and_folder_list = folder_configure.create_folders()
    map_mask = level5data.map[0]["mask"]

    for data_folder, df in df_and_folder_list:
        print("Generating bev from lidar 3d points into {}".format(data_folder))
        first_samples = df.first_sample_token.values
        hosts = df.host.values
        sample_meta: List = []
        gt: List = []
        for i in tqdm(range(first_samples.shape[0])):
            sample_meta_data, gt_box3ds = prepare_training_data_for_scene(
                first_samples[i],
                hosts[i],
                level5data=level5data,
                output_folder=data_folder,
                test_mode=args.test_mode,
                debug_mode=args.debug_mode,
                bev_shape=bev_shape,
                map_mask=map_mask,
                voxel_size=voxel_size,
                z_offset=z_offset,
                box_scale=box_scale,
                max_intensity=args.max_intensity,
            )
            if args.debug_mode:
                if i > 5:
                    break
            sample_meta.extend(sample_meta_data)
            gt.extend(gt_box3ds)

        # making input meta json for training
        sample_meta = [meta_._asdict() for meta_ in sample_meta]
        with open(Path(data_folder, INPUT_META_JSON_NAME), "w") as f:
            json.dump(sample_meta, f)

        if not args.test_mode:
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
        #     test_mode=args.test_mode,
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
        description="making BEV image for lyft 3d object detection",
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
        "--display_images",
        action="store_true",
        help="check the intermediate images",
    )
    parser.add_argument(
        "--add_z_info",
        action="store_true",
        help="add z info on target image",
    )
    parser.add_argument(
        "--test_mode",
        action="store_true",
        help="test mode image generation",
    )
    parser.add_argument(
        "--debug_mode",
        action="store_true",
        help="debug mode, only few scenes are processed",
    )
    parser.add_argument(
        "--workers",
        default=4,
        type=int,
        help="number of data loading workers",
    )
    args = parser.parse_args()
    main(args)
