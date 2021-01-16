from pathlib import Path

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

# output path for test mode
CSV_NAME = "submission.csv"
