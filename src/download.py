import os
import shutil
from pathlib import Path

import gdown

from lib.data.data_subset import DataSubset
from lib.util.file import (
    ARCHIVE_FILE_EXTENSION,
    DATA_DEFAULT_PATH,
    DEFAULT_MASK_FILE_EXTENSION,
    IMAGE_FILE_EXTENSION,
    TESTING_DATA_FOLDER,
    TESTING_DATA_STRUCTURE,
    TRAINING_DATA_FOLDER,
    TRAINING_DATA_STRUCTURE,
)

TRAINING_DATA_URL = "https://drive.google.com/uc?id=1ZgqFJomqQGNnsx7w7QBzQQMVA16lbVCA"
TESTING_DATA_URL = "https://drive.google.com/uc?id=1NKkSQ5T0ZNQ8aUhh0a8Dt2YKYCQXIViw"


def download_data(url: str, subset: DataSubset) -> Path:
    filepath = DATA_DEFAULT_PATH / f"{subset.value}{ARCHIVE_FILE_EXTENSION}"
    gdown.download(url, output=str(filepath), quiet=False)

    return filepath


def unpack_and_copy_data(filepath: Path, subset: DataSubset) -> None:
    print(f"Unpacking {filepath}")
    shutil.unpack_archive(filepath, extract_dir=DATA_DEFAULT_PATH)

    folder = TRAINING_DATA_FOLDER if subset == DataSubset.TRAIN else TESTING_DATA_FOLDER
    structure = (
        TRAINING_DATA_STRUCTURE
        if subset == DataSubset.TRAIN
        else TESTING_DATA_STRUCTURE
    )
    target_path = DATA_DEFAULT_PATH / subset.value

    print(f"Moving files to {target_path}")
    for image_path in structure.images.glob(f"*{IMAGE_FILE_EXTENSION}"):
        shutil.move(image_path, target_path)

    for mask_path in structure.annotations.glob(f"*{DEFAULT_MASK_FILE_EXTENSION}"):
        shutil.move(mask_path, target_path)

    print("Cleaning up")
    os.remove(filepath)
    shutil.rmtree(folder)


if __name__ == "__main__":
    training_data_filepath = download_data(TRAINING_DATA_URL, DataSubset.TRAIN)
    unpack_and_copy_data(training_data_filepath, DataSubset.TRAIN)

    testing_data_filepath = download_data(TESTING_DATA_URL, DataSubset.TEST)
    unpack_and_copy_data(testing_data_filepath, DataSubset.TEST)
