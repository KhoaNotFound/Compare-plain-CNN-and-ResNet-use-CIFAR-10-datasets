from datasets import load_dataset
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

#load data
dataset = load_dataset("uoft-cs/cifar10")

#inspect schema
print(dataset)
classes = dataset["train"].features["label"].names
print(classes)