from dataclasses import dataclass
from pathlib import Path


@dataclass
class DatasetStructure:
    images: Path
    annotations: Path
