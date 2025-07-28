import librosa
import librosa.display
import os
import numpy as np
import yaml
import pandas as pd
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Dataset
import sounddevice as sd

CONFIG_PATH = os.getenv("CONFIG_PATH", "../config.yaml")
with open(CONFIG_PATH, "r") as f:
    cfg = yaml.safe_load(f)

# Config variables
DATA_DIR = os.path.abspath(cfg.get("data_dir", "data/UrbanSound8K"))
DESIRED_SAMPLE_RATE = cfg.get("sample_rate", 16000)
TARGET_DURATION = cfg.get("target_duration", 4.0)

# PyTorch dataset
class UrbanSoundEmbeddingDataset(Dataset):
    def __init__(self, paths, labels, embedder):
        self.paths = paths
        self.labels = labels
        self.embedder = embedder

    def __getitem__(self, idx):
        audio = self.load_audio_file(self.paths[idx])
        embedding = self.embedder.get_embedding(audio)
        filename = os.path.basename(self.paths[idx])
        return (
            torch.tensor(embedding, dtype=torch.float32),
            torch.tensor(self.labels[idx], dtype=torch.long),
            filename,
            idx
        )


    def __len__(self):
        return len(self.paths)

    def load_audio_file(self, path, sr=DESIRED_SAMPLE_RATE, duration=TARGET_DURATION):
        waveform, _ = librosa.load(path, sr=sr, mono=True)
        target_length = int(sr * duration)
        if len(waveform) > target_length:
            waveform = waveform[:target_length]
        else:
            waveform = np.pad(waveform, (0, target_length - len(waveform)))

        return waveform

class UrbanSoundLoader:
    def __init__(self, embedder):
        self.embedder = embedder
        self.metadata_path = os.path.join(DATA_DIR, "UrbanSound8K.csv")

    def get_labeled_unlabeled_datasets(self, held_out_fold, labeled_count=50):
        df = pd.read_csv(self.metadata_path)
        df['class_code'] = df['class'].astype('category').cat.codes

        train_df = df[df['fold'] != held_out_fold]
        test_df = df[df['fold'] == held_out_fold]

        # Taking 5 samples from every class for training
        labeled_df = train_df.groupby('class_code', group_keys=False).apply(lambda x: x.sample(n=10, random_state=42))
        unlabeled_df = train_df.drop(labeled_df.index)
        combined_df = pd.concat([labeled_df, unlabeled_df]).reset_index(drop=True)
        labeled_indices = list(range(len(labeled_df)))
        unlabeled_indices = list(range(len(labeled_df), len(combined_df)))

        train_paths = [Path(DATA_DIR) / f"fold{row['fold']}" / row['slice_file_name'] for _, row in combined_df.iterrows()]
        train_labels = combined_df['class_code'].to_numpy()
        full_train_dataset = UrbanSoundEmbeddingDataset(train_paths, train_labels, self.embedder)

        test_paths = [Path(DATA_DIR) / f"fold{row['fold']}" / row['slice_file_name'] for _, row in test_df.iterrows()]
        test_labels = test_df['class_code'].to_numpy()
        test_dataset = UrbanSoundEmbeddingDataset(test_paths, test_labels, self.embedder)

        return full_train_dataset, labeled_indices, unlabeled_indices, test_dataset


