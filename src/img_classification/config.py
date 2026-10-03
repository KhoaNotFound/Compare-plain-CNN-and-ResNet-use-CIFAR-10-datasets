#%%
from pathlib import Path
import sys
#%%
PROJECT_ROOT = Path(__file__).resolve().parents[2]
print("Project root: ", PROJECT_ROOT)
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)