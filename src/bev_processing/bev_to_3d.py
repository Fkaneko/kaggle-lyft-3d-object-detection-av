from typing import List, Tuple

import cv2
import numpy as np

from src.config.config import CLASSES


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
