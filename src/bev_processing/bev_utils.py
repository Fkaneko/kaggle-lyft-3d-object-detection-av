from typing import NamedTuple


class SampleMeta(NamedTuple):
    """bev meta infomation for training"""

    sample_token: str
    host: str
    ego_pose: dict
    global_from_voxel: list
