from typing import List

from lyft_dataset_sdk.eval.detection.mAP_evaluation import Box3D
from lyft_dataset_sdk.lyftdataset import LyftDataset
from tqdm import tqdm


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
