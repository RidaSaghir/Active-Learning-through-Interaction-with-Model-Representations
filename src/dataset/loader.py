import scipy
from scipy.io import wavfile
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

# Access configuration variables
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
        # returns embedding and label
        return (
            torch.tensor(embedding, dtype=torch.float32),
            torch.tensor(self.labels[idx], dtype=torch.long),
            filename
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

    def get_labeled_unlabeled_datasets(self, labeled_count=50, batch_size=16):


        df = pd.read_csv(self.metadata_path)
        if 'manual_label' in df.columns:
            # Once there are manual labels from the user
            labeled_df = df[df['manual_label'] == True]
            unlabeled_df = df[df['manual_label'] != True]
        else:
            # Cold start fallback
            labeled_df = df.iloc[:labeled_count]
            unlabeled_df = df.iloc[labeled_count:]

        # Convert to paths and labels
        labeled_paths = [Path(DATA_DIR) / f"fold{row['fold']}" / row['slice_file_name'] for _, row in
                         labeled_df.iterrows()]
        labeled_labels = labeled_df['class'].astype('category').cat.codes.to_numpy()

        unlabeled_paths = [Path(DATA_DIR) / f"fold{row['fold']}" / row['slice_file_name'] for _, row in
                           unlabeled_df.iterrows()]
        unlabeled_labels = unlabeled_df['class'].astype('category').cat.codes.to_numpy()

        # Initializing class for fetching data with suitable sample rate and length
        labeled_ds = UrbanSoundEmbeddingDataset(labeled_paths, labeled_labels, self.embedder)
        unlabeled_ds = UrbanSoundEmbeddingDataset(unlabeled_paths, unlabeled_labels, self.embedder)

        # Wrap in DataLoaders (Pytorch specific)
        labeled_loader = DataLoader(labeled_ds, batch_size=batch_size, shuffle=True)

        # For active learning, track indices of unlabeled samples
        indexed_unlabeled = [(x[0], i) for i, x in enumerate(DataLoader(unlabeled_ds, batch_size=1))]
        unlabeled_loader = DataLoader(indexed_unlabeled, batch_size=batch_size)

        return labeled_ds, unlabeled_ds


