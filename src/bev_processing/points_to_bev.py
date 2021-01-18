import os
from pathlib import Path
from typing import List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from lyft_dataset_sdk.eval.detection.mAP_evaluation import Box3D
from lyft_dataset_sdk.lyftdataset import LyftDataset
from lyft_dataset_sdk.utils.data_classes import LidarPointCloud, Quaternion
from lyft_dataset_sdk.utils.geometry_utils import transform_matrix
from PIL import Image

from src.bev_processing.bev_utils import SampleMeta, transform_points
from src.config.config import (
    BEV_MAP_SUFFIX,
    BEV_TARGET_SUFFIX,
    BEV_TRAIN_SUFFIX,
    CLASSES,
)
from src.evaluation.ground_trouth_box import make_gt_boxes_from_sample


def convert_scene_into_bev_images(
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
    output_map: bool = False,
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
        # extract map, but it needs additonal processing time and does not contribute
        # to accuracy.
        if output_map:
            semantic_im = get_semantic_map_around_ego(
                map_mask, ego_pose, voxel_size[0], (bev.shape[0], bev.shape[1])
            )
            semantic_im = np.round(semantic_im * 255).astype(np.uint8)
            cv2.imwrite(
                os.path.join(
                    output_folder,
                    host,
                    "{}_{}.png".format(sample_token, BEV_MAP_SUFFIX),
                ),
                cv2.cvtColor(semantic_im, cv2.COLOR_RGB2BGR),
            )
        else:
            semantic_im = None

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
            img = np.hstack((bev_im, semantic_im)) if output_map else bev_im
            plt.imshow(img)
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
