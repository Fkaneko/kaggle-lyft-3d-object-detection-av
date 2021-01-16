from typing import List, Tuple

import cv2
import numpy as np
from lyft_dataset_sdk.eval.detection.mAP_evaluation import Box3D
from scipy.spatial.transform import Rotation as R

from make_image import transform_points
from src.config.config import CLASSES, CLASS_AVG_HEIGHTS


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


def create_3d_boxes_from_2d(
    sample_boxes,
    sample_detection_classes,
    ego_translation,
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
    sample_boxes[2, :] = ego_translation[2]
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

    return sample_boxes, sample_boxes_centers, sample_boxes_dimensions


def convert_into_nuscene_3dbox(
    sample_token,
    sample_boxes,
    sample_boxes_centers,
    sample_boxes_dimensions,
    sample_detection_class,
    sample_detection_scores,
):
    pred_box3ds = []
    for i in range(len(sample_boxes)):
        translation = sample_boxes_centers[i]
        size = sample_boxes_dimensions[i]
        class_name = sample_detection_class[i]

        # Determine the rotation of the box
        v = sample_boxes[i, 0] - sample_boxes[i, 1]  # (3, )
        v /= np.linalg.norm(v)
        r = R.from_dcm(
            [
                [v[0], -v[1], 0],
                [v[1], v[0], 0],
                [0, 0, 1],
            ]
        )
        quat = r.as_quat()
        # XYZW -> WXYZ order of elements
        quat = quat[[3, 0, 1, 2]]

        detection_score = float(sample_detection_scores[i])

        box3d = Box3D(
            sample_token=sample_token,
            translation=list(translation),
            size=list(size),
            rotation=list(quat),
            name=class_name,
            score=detection_score,
        )
        pred_box3ds.append(box3d)
    return pred_box3ds

    # pred = [b.serialize() for b in pred_box3ds]
    # with open(os.path.join(model_out_dir, pred_json_file_name), "w") as f:
    #     json.dump(pred, f)
