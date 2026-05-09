import pandas as pd
import numpy as np
import torch
from typing import Tuple
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
        path = config.Output_path / f"{sub}_combined_ready.csv"

        if not path.exists():
            raise FileNotFoundError(f"Missing file: {path}")
        
        combined_df = pd.concat(all_dfs, ignore_index=True)
        x = combined_df.drop(columns=['Knee_Angle_X', 'Side']).values
        y = combined_df['Knee_Angle_X'].values

        return x, y

    # MLR only needs simple X and y arrays
    x = combined_df.drop(columns=['Knee_Angle_X', 'Side']).values
    y = combined_df['Knee_Angle_X'].values
    
    return x, y

def get_torch_loader(subject_list: list, window_size: int, batch_size: int, shuffle=True):
    """
    Stacks multiple participants for training.
    Returns a PyTorch DataLoader for CNN/LSTM models.
    """
    all_dfs = []

    # Loop through the list of subjects
    for sub in subject_list:
        # Match exact filename from Processed_Results
        path = config.Output_path / f"{sub}_combined_ready.csv"
        all_dfs.append(pd.read_csv(path))

        if not path.exists():
            logger.error(f"Could not find {path}. Did you run the pipeline?")
            raise FileNotFoundError(f"Missing: {path}")
            
        all_dfs.append(pd.read_csv(path))

    # Stack them vertically into one training pool
    combined_df = pd.concat(all_dfs, ignore_index=True)
    # Iinitialize the Dataset with the provided window size
    dataset = GaitDataset(combined_df, window_size=window_size)
    
    # Return the DataLoader
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def get_intra_loader(subject: str, window_size: int, batch_size: int):
    """ Returns two DataLoaders (Train/Test) for a single-subject 80/20 split. """
    path = config.Output_path / f"{subject}_combined_ready.csv"

    if not path.exists():
        logger.error(f"File not found: {path}")
        raise FileNotFoundError(f"Missing file for subject {subject}")
    
    df = pd.read_csv(path)
    split = int(len(df) * 0.8)
    
    train_ds = GaitDataset(df.iloc[:split], window_size)
    test_ds = GaitDataset(df.iloc[split:], window_size)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader