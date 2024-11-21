from pathlib import Path
from typing import Optional

from lib.data.dataset_structure import DatasetStructure

ARCHIVE_FILE_EXTENSION = ".zip"
IMAGE_FILE_EXTENSION = ".tif"
DEFAULT_MASK_FILE_EXTENSION = ".xml"
PARSED_MASK_FILE_EXTENSION = ".npy"
STATE_DICT_FILE_EXTENSION = ".pth"
METRICS_FILE_EXTENSION = ".pkl"
PATCH_INFIX = "-PTCH-"

MODELS_PATH = Path("models").absolute()
MODEL_CHECKPOINTS_PATH = MODELS_PATH / "checkpoints"
MODEL_METRICS_PATH = MODELS_PATH / "metrics"

DATA_PATH = Path("data").absolute()
DATA_DEFAULT_PATH = DATA_PATH / "default"
DATA_PATCHES_PATH = DATA_PATH / "patches"

TRAINING_DATA_FOLDER = DATA_DEFAULT_PATH / "MoNuSeg 2018 Training Data"
TRAINING_DATA_STRUCTURE = DatasetStructure(
    TRAINING_DATA_FOLDER / "Tissue Images", TRAINING_DATA_FOLDER / "Annotations"
)
TESTING_DATA_FOLDER = DATA_DEFAULT_PATH / "MoNuSegTestData"
TESTING_DATA_STRUCTURE = DatasetStructure(TESTING_DATA_FOLDER, TESTING_DATA_FOLDER)


def image_filename(identifier: str) -> str:
    return f"{identifier}{IMAGE_FILE_EXTENSION}"


def default_mask_filename(identifier: str) -> str:
    return f"{identifier}{DEFAULT_MASK_FILE_EXTENSION}"


def parsed_mask_filename(identifier: str, color: bool) -> str:
    return f"{identifier}_{'color' if color else 'binary'}{PARSED_MASK_FILE_EXTENSION}"


def losses_filename(timestamp: str) -> str:
    return f"unet_{timestamp}_losses{METRICS_FILE_EXTENSION}"


def dice_scores_filename(timestamp: str) -> str:
    return f"unet_{timestamp}_dice_scores{METRICS_FILE_EXTENSION}"


def state_dict_filename(timestamp: str, epoch: Optional[int] = None) -> str:
    if epoch is None:
        return f"unet_{timestamp}{STATE_DICT_FILE_EXTENSION}"

    return f"unet_{timestamp}_epoch_{epoch:03d}{STATE_DICT_FILE_EXTENSION}"


def state_dict_path(timestamp: str, epoch: Optional[int] = None) -> Path:
    if epoch is None:
        return MODELS_PATH

    return MODEL_CHECKPOINTS_PATH / timestamp


def patch_suffix(patch_idx: int) -> str:
    if patch_idx < 0:
        return ""

    return f"{PATCH_INFIX}{patch_idx:02d}"
