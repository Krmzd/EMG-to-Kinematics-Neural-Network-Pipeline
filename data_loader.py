import pandas as pd
import numpy as np
import torch
from typing import Tuple
from pathlib import path
from torch.utils.data import Dataset, DataLoader
import config
import logging

logger = logging.getLogger("Data_loder") 

class GaitDataset(Dataset):
    def __init__(self, data: pd.DataFrame, window_size: int = 1):

         # Separate X and y
        self.X = data.drop(columns=['Knee_Angle_X', 'Side']).values.astype(np.float32)
        self.y = data['Knee_Angle_X'].values.astype(np.float32)
        self.window_size = window_size

    def __len__(self) -> int:
        # We subtract window_size so we don't go out of bounds
        return len(self.X) - self.window_size

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.tensor]:
        # Create a window of 'history' data
        window_X = self.X[idx : idx + self.window_size]
        # The target is the angle at the last frame of that window
        target_y = self.y[idx + self.window_size]
        
        return torch.tensor(window_X), torch.tensor(target_y)

def get_mlr_data(participants_list: list):
    """
    Returns data in the 'Flat' format for Scikit-learn MLR.
    """
    all_dfs = []
    for sub in participants_list:
        df = pd.read_csv(config.Output_path / f"{sub}_ready.csv")
        if not path.exists():
            logger.error(f"ready_sub for {sub} not found at {path}")
            raise FileNotFoundError(f"Missing file: {path}")
        all_dfs.append(df)
    
    combined_df = pd.concat(all_dfs)
    
    # MLR only needs simple X and y arrays
    x = combined_df.drop(columns=['Knee_Angle_X', 'Side']).values
    y = combined_df['Knee_Angle_X'].values
    
    return x, y

def get_torch_loader(participant, window_size, batch_size, shuffle=True):
    """
    Returns a PyTorch DataLoader for CNN/LSTM models.
    """
    df = pd.read_csv(config.Output_path / f"{participant} ready_sub.csv")
    dataset = GaitDataset(df, window_size=window_size)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)