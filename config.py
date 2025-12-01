import os
import yaml

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")

with open(CONFIG_PATH, "r") as f:
    cfg = yaml.safe_load(f)

# Expose values as constants
DATA_DIR = os.environ.get("DATA_DIR", os.path.abspath(cfg.get("data_dir", "data/UrbanSound8K")))
NUM_CLASSES = cfg.get("num_classes", 10)
SAMPLE_RATE = cfg.get("sample_rate", 16000)
TARGET_DURATION = cfg.get("target_duration", 4.0)
EMBEDDING_DIM = cfg.get("embedding_dim", 1024)
BROADCAST_INTERVAL = cfg.get("broadcast_interval", 20)
BASE_URL = cfg.get("base_url", "http://localhost:8000")
CHECKPOINT = cfg.get("check_point")
HUMAN_ANNOTATIONS = cfg.get("human_annotations")
LEARNING_RATE = float(cfg.get("learning_rate"))
BATCH_SIZE = cfg.get("batch_size")
INITIAL_LABELS_PER_CLASS_COUNT = cfg.get("initial_labels_per_class_count")
NUM_ANNOTATION_SUGGESTIONS = cfg.get("num_annotation_suggestions")
#NUM_ITERATIONS = cfg.get("num_iterations")
HELD_OUT_FOLD = cfg.get("held_out_fold", 10)
EPOCHS_BEFORE_QUERY = cfg.get("epochs_before_query")
ACCURACY_TARGET = cfg.get("accuracy_target")
CSV_FILENAME = cfg.get("csv_filename")
PNG_FILENAME = cfg.get("png_filename")
LABELS_PER_ROUND = cfg.get("labels_per_round")
