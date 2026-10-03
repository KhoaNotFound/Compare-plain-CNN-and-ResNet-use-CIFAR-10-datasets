"""Runtime paths independent of the installed package location; no import side effects."""
import os
from pathlib import Path

DATA_DIR = Path(os.environ.get("IMG_CLASSIFICATION_DATA_DIR", "data")).expanduser().resolve()
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

from dataclasses import asdict, dataclass, fields
import math
import tomllib


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 20
    batch_size: int = 128
    lr: float = 0.001
    seed: int = 42
    val_fraction: float = 0.1
    amp: bool = True

    def __post_init__(self):
        for name in ("epochs", "batch_size", "seed"):
            value = getattr(self, name)
            if type(value) is not int or value < (0 if name == "seed" else 1):
                raise ValueError(f"{name} must be an integer >= {0 if name == 'seed' else 1}")
        if type(self.lr) not in (int, float) or not math.isfinite(self.lr) or self.lr <= 0:
            raise ValueError("lr must be positive and finite")
        if type(self.val_fraction) not in (int, float) or not 0 < self.val_fraction < 1:
            raise ValueError("val_fraction must be between 0 and 1")
        if type(self.amp) is not bool:
            raise ValueError("amp must be a boolean")

    def to_dict(self):
        return asdict(self)


def load_train_config(path: Path | None = None, **overrides) -> TrainConfig:
    values = {}
    if path is not None:
        with path.open("rb") as stream:
            document = tomllib.load(stream)
        if set(document) != {"training"}:
            raise ValueError("Config must contain only a [training] section")
        values = document["training"]
        if not isinstance(values, dict):
            raise ValueError("[training] must be a TOML table")
    values.update({key: value for key, value in overrides.items() if value is not None})
    unknown = set(values) - {field.name for field in fields(TrainConfig)}
    if unknown:
        raise ValueError(f"Unknown training settings: {sorted(unknown)}")
    return TrainConfig(**values)
