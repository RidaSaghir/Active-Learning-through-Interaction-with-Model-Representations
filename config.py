import os
import yaml

CONFIG_PATH = os.getenv("CONFIG_PATH", "../config.yaml")

with open(CONFIG_PATH, "r") as f:
    cfg = yaml.safe_load(f)

# Expose values as constants
DATA_DIR = os.path.abspath(cfg.get("data_dir", "data/UrbanSound8K"))
NUM_CLASSES = cfg.get("num_classes", 10)
SAMPLE_RATE = cfg.get("sample_rate", 16000)
TARGET_DURATION = cfg.get("target_duration", 4.0)
EMBEDDING_DIM = cfg.get("embedding_dim", 1024)
BROADCAST_INTERVAL = cfg.get("broadcast_interval", 20)
