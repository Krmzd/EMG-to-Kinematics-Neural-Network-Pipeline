import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import config

class GaitDataset(Dataset):
    def __init__(self, data: pd.DataFrame, window_size: int = 1):
        """
            Custom Dataset that can handle both 'Flat' (MLR) and 'Windowed' (AI) data.
        """
         # Separate X and y
        self.X = data.drop(columns=['Knee_Angle_X', 'Side']).values.astype(np.float32)
        self.y = data['Knee_Angle_X'].values.astype(np.float32)
        self.window_size = window_size