import os
import pickle
import random
from argparse import Namespace
from pathlib import Path
from pprint import pprint
from typing import Dict, List, NamedTuple, Optional, Tuple, Union

import albumentations as albu
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from omegaconf import OmegaConf

# lyft dataset class definition
CLASSES = (
    "car",
    "motorcycle",
    "bus",
    "bicycle",
    "truck",
    "pedestrian",
    "other_vehicle",
    "animal",
    "emergency_vehicle",
)
CLASS_AVG_HEIGHTS = {
    "animal": 0.51,
    "bicycle": 1.44,
    "bus": 3.44,
    "car": 1.72,
    "emergency_vehicle": 2.39,
    "motorcycle": 1.59,
    "other_vehicle": 3.23,
    "pedestrian": 1.78,
    "truck": 3.44,
}
CLASS_AVG_WIDTHS = {
    "animal": 0.36,
    "bicycle": 0.63,
    "bus": 2.96,
    "car": 1.93,
    "emergency_vehicle": 2.45,
    "motorcycle": 0.96,
    "other_vehicle": 2.79,
    "pedestrian": 0.77,
    "truck": 2.84,
}

CLASS_AVG_LENGTHS = {
    "animal": 0.73,
    "bicycle": 1.76,
    "bus": 12.34,
    "car": 4.76,
    "emergency_vehicle": 6.52,
    "motorcycle": 2.35,
    "other_vehicle": 8.20,
    "pedestrian": 0.81,
    "truck": 10.24,
}
# all host
ALL_HOSTS = (
    "host-a004",
    "host-a005",
    "host-a006",
    "host-a007",
    "host-a008",
    "host-a009",
    "host-a011",
    "host-a012",
    "host-a015",
    "host-a017",
    "host-a101",
    "host-a102",
)

# dataset link
DATSET_LINKS_DIR = Path.cwd() / Path("Lyft3D_detection_links")

# bev dirctory name
BEV_FOLDER_VERSION_PREFIX = "version_"
BEV_TARGET_SUFFIX = "target"
BEV_TRAIN_SUFFIX = "input_0"
BEV_MAP_SUFFIX = "map"

# input meta json for training
INPUT_META_JSON_NAME = Path("bev_input_meta.json")

# ground truth json name
GT_JSON_NAME = Path("ground_truth_boxes.json")

CLASS_LOSS_WEIGHTS = [0.4] + [1.0] * len(CLASSES)

# random seed
SEED = 42


def set_random_seed(seed: int = SEED):
    os.environ["PYTHONHASHSEED"] = str(seed)
    # set Python random seed
    random.seed(seed)
    # set NumPy random seed
    np.random.seed(seed)


def print_argparse_arguments(p, bar=50, log=None):
    """
    Visualize argparse arguments.
    Arguments:
        p : parse_arge() object.
        bar : the number of bar on the output.

    argparseの parse_args() で生成されたオブジェクトを入力すると、
    integersとaccumulateを自動で取得して表示する
    [in] p: parse_args()で生成されたオブジェクト
    [in] bar: 区切りのハイフンの数
    """
    if log:
        log.write("PARAMETER SETTING\n")
        log.write("-" * bar)
        log.write("\n")
        args = [(i, getattr(p, i)) for i in dir(p) if not "_" in i[0]]
        for i, j in args:
            if isinstance(j, list):
                log.write("{0}[{1}]:".format(i, len(j)))
                [log.write("\t{}".format(k)) for k in j]
                log.write("\n")
            else:
                log.write("{0:25}:{1}\n".format(i, j))
        log.write("-" * bar)
        log.write("\n")
    else:
        print("PARAMETER SETTING")
        print("-" * bar)
        args = [(i, getattr(p, i)) for i in dir(p) if not "_" in i[0]]
        for i, j in args:
            if isinstance(j, list):
                print("{0}[{1}]:".format(i, len(j)))
            else:
                print("{0:25}:{1}".format(i, j))
        return print("-" * bar)


# def load_config_data(path: str) -> dict:
#     with open(path) as f:
#         cfg: dict = yaml.load(f, Loader=yaml.FullLoader)
#     return cfg


def load_image_description_from_dict(args, save_dir):
    with open(
        os.path.join(os.path.join(save_dir, "image_description_dict.dump")), "rb"
    ) as f:
        image_description_dict = pickle.load(f)

    args.voxel_size_xy = image_description_dict["voxel_size_xy"]
    args.voxel_size_z = image_description_dict["voxel_size_z"]
    args.image_size = image_description_dict["image_size"]
    args.image_channel = image_description_dict["image_channel"]
    args.z_offset = image_description_dict["z_offset"]
    args.max_intensity = image_description_dict["max_intensity"]
    args.box_scale = image_description_dict["box_scale"]


def save_config_data(conf: Union[dict, Namespace], path: Path) -> None:
    # with fs.open(config_yaml, "w", encoding="utf-8") as fp:
    if isinstance(conf, Namespace):
        conf = vars(conf)
    print("saving the following configuration:", path.name)
    pprint(conf)
    conf = OmegaConf.create(conf)
    # with open(path, "w", encoding="utf-8") as fp:
    with path.open("w", encoding="utf-8") as fp:
        OmegaConf.save(conf, fp, resolve=True)


def make_version_folder(
    output_dir: Path, ver_prefix: str = BEV_FOLDER_VERSION_PREFIX
) -> Path:
    output_dir.mkdir(exist_ok=True)
    ver_list = list(output_dir.glob(ver_prefix + "*"))
    if len(ver_list) > 0:
        vers = [int(x.name.replace(ver_prefix, "")) for x in ver_list]
        version_cnt = sorted(vers, key=int)[-1] + 1
    else:
        version_cnt = 0
    version_dir = output_dir / Path(ver_prefix + str(version_cnt))
    version_dir.mkdir()
    return version_dir


def make_dataset_links(dataset_root: Path, data_suffix: str) -> None:
    Path(DATSET_LINKS_DIR, data_suffix).mkdir(parents=True, exist_ok=True)
    folder_types = ("images", "maps", "lidar")
    for type_ in folder_types:
        target_folder = dataset_root / Path(data_suffix + "_" + type_)
        symbol_file = Path(DATSET_LINKS_DIR, data_suffix, type_)
        if symbol_file.is_symlink():
            continue
        symbol_file.symlink_to(target_folder, target_is_directory=True)


def get_val_hosts(val_choice: int = 0) -> Tuple[str, ...]:
    if val_choice == 0:
        return ("host-a007", "host-a008", "host-a009")
    elif val_choice == 1:
        return ("host-a004",)
    elif val_choice == 2:
        return ("host-a015", "host-a101", "host-a102")
    elif val_choice == 3:
        return ("host-a011",)
    else:
        print(f"unexptected val_choice:{val_choice}, so the default will be used")
        return ("host-a007", "host-a008", "host-a009")


class SampleMeta(NamedTuple):
    """bev meta infomation for training"""

    sample_token: str
    host: str
    ego_pose: dict
    global_from_voxel: list


class BEVImageDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        input_filepaths: List[Path],
        input_meta_dict: Dict[str, SampleMeta],
        target_filepaths: Optional[List[Path]],
        transforms: albu.core.composition.Compose,
        map_filepaths: Optional[List[Path]] = None,
        multi_channel_filepaths: Optional[List] = None,
    ) -> None:
        self.input_filepaths = input_filepaths
        self.target_filepaths = target_filepaths
        self.input_meta_dict = input_meta_dict

        self.map_filepaths = map_filepaths
        self.transforms = transforms
        self.multi_channel_filepaths = multi_channel_filepaths

        if map_filepaths is not None:
            assert len(input_filepaths) == len(map_filepaths)
        self.test_mode = target_filepaths is None

    def __len__(self):
        return len(self.input_filepaths)

    def __getitem__(self, idx: int) -> dict:
        input_filepath = self.input_filepaths[idx]

        sample_token = input_filepath.name.replace(f"_{BEV_TRAIN_SUFFIX}.png", "")
        input_meta = self.input_meta_dict[sample_token]

        if self.multi_channel_filepaths is not None:
            im_list = []
            for i in range(len(self.multi_channel_filepaths)):
                input_filepath_ = self.multi_channel_filepaths[i][idx]
                im_list.append(cv2.imread(input_filepath_, cv2.IMREAD_UNCHANGED))
            im = np.concatenate(im_list, axis=2)

        else:
            im = cv2.cvtColor(
                cv2.imread(str(input_filepath), cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB
            )

        if self.map_filepaths is not None:
            map_filepath = self.map_filepaths[idx]
            map_im = cv2.cvtColor(
                cv2.imread(str(map_filepath), cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB
            )
            im = np.concatenate((im, map_im), axis=2)

        if self.target_filepaths is None:
            target = None
        else:
            target_filepath = self.target_filepaths[idx]
            # target is grey image
            target = cv2.imread(str(target_filepath), cv2.IMREAD_UNCHANGED)

        if self.transforms:
            augmented = self.transforms(image=im, mask=target)
            im = augmented["image"]
            if not self.test_mode:
                target = augmented["mask"]
                target = target.astype(np.int64)

        if self.target_filepaths is not None:
            target = torch.from_numpy(target)

        im = torch.from_numpy(im.transpose(2, 0, 1))
        return {
            "image": im,
            "target": target,
            "sample_token": sample_token,
            "global_from_voxel": torch.tensor(input_meta.global_from_voxel),
            "ego_pose": input_meta.ego_pose,
        }


def visualize_predictions(
    input_image: torch.Tensor,
    prediction: torch.Tensor,
    target: torch.Tensor = None,
    n_images: int = 2,
    background_threshold=122,
):
    """
    Takes as input 3 PyTorch tensors, plots the input image, predictions and targets.
    You can interpret the above visualizations as follows:
    There are four different visualizations stacked on top of eachother:
    1. The top images have two color channels: red for predictions, green for targets, with red+green=yellow. In other words:
    > **Black**: True Negative
    **Green**: False Negative
    **Yellow**: True Positive
    **Red**: False Positive
    2. The input image
    3. The input image or semantic input map blended together with targets+predictions
    4. The predictions thresholded at 0.5 probability.
    """
    # Only select the first n images
    prediction = prediction[:n_images]

    # if test_mode:
    #     target = prediction[:n_images]
    # else:
    #     target = target[:n_images]
    target = target[:n_images]
    input_image = input_image[:n_images]

    # target: (batch_size, H, W) -> (H, W*batch_size)
    # prediction: (batch_size, 10, H, W) -> (batch_size, H, W) -> (H, W*batch_size)
    prediction = prediction.detach().cpu().numpy()

    target = target.detach().cpu().numpy()
    # "car" prob and batch direction is stacked horizontally on the last axis
    class_one_preds = 1 - prediction[:, 0]
    # (H, W*batch_size) -> (H, W*batch_size, 3)
    class_rgb = np.repeat(class_one_preds[..., None], 3, axis=2)
    class_rgb[..., 2] = 0
    class_rgb[..., 1] = target

    # (batch_size, 6, H, W) -> (6, H, W*batch_size) -> (H, W*batch_size, 6)
    input_im = np.hstack(input_image.cpu().numpy().transpose(0, 2, 3, 1))

    if input_im.shape[2] == 3:  # Without semantic map input
        input_im_grayscale = np.repeat(input_im.mean(axis=2)[..., None], 3, axis=2)
        overlayed_im = (input_im_grayscale * 0.6 + class_rgb * 0.7).clip(0, 1)
    else:
        input_map = input_im[..., -3:]  # With semantic map input
        overlayed_im = (input_map * 0.6 + class_rgb * 0.7).clip(0, 1)

    thresholded_pred = np.repeat(
        class_one_preds[..., None] > background_threshold / 255.0, 3, axis=2
    )

    fig = plt.figure(figsize=(12, 26))
    plot_im = (
        np.vstack([class_rgb, input_im[..., :3], overlayed_im, thresholded_pred])
        .clip(0, 1)
        .astype(np.float32)
    )
    plt.imshow(plot_im)
    plt.axis("off")
    return fig
