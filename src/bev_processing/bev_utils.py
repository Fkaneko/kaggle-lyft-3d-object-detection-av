from datetime import datetime
from pathlib import Path
from typing import List, NamedTuple, Tuple

import numpy as np
import pandas as pd
from lyft_dataset_sdk.lyftdataset import LyftDataset


class SampleMeta(NamedTuple):
    """bev meta infomation for training"""

    sample_token: str
    host: str
    ego_pose: dict
    global_from_voxel: list


class LyftBEVFolderConfiguration:
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
