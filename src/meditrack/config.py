import os
from dataclasses import dataclass
from pathlib import Path

@dataclass
class MediTrackConfig:
    data_dir: Path = Path(os.getenv("MEDTRACK_DATA_DIR", "./data"))
    outputs_dir: Path = Path(os.getenv("MEDTRACK_OUTPUT_DIR", "./data/outputs"))

    @property
    def sample_wounds_dir(self) -> Path:
        return self.data_dir / "sample_wounds"

CONFIG = MediTrackConfig()
