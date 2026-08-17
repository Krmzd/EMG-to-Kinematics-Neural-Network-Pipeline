import pandas as pd
import numpy as np
import torch
from typing import Tuple
from torch.utils.data import Dataset, DataLoader
import config
import logging

logger = logging.getLogger("Data_loader")

class GaitDataset(Dataset):
    # 1. Extract EMG features (X) and knee angle target (y).
    # 2. Normalize X and y using statistics calculated from the training data.
    # 3. Find valid starting positions for 25-sample windows without crossing trial/side boundaries.
    # 4. __getitem__() uses one valid starting position to create one window and uses the center of that window as the target knee angle.
    def __init__(self, data: pd.DataFrame, window_size: int, mu_y=None, sigma_y=None, mu_x=None, sigma_x=None):
        self.window_size = window_size
        self.side_labels = data['Side'].values
        #  drop every non_EMG column. the left -> EMG feature, stored as self.x and self.y
        drop_cols = [c for c in ['Knee_Angle_X', 'Side', 'trial_id', 'Frame'] if c in data.columns]
        emg_df = data.drop(columns=drop_cols)
        # Convert EMG data to a NumPy float32 array.
        # self.X contains all EMG samples/features.
        self.X = emg_df.values.astype(np.float32)
        # Store the knee angle as the prediction target.
        self.y = data['Knee_Angle_X'].values.astype(np.float32)
        # Normalize EMG input (X)
        # mu_x and sigma_x are calculated from the training data.
        # They can be:
        #   1. One mean/std for each EMG channel (global normalization), or
        #   2. Separate mean/std values for Left and Right sides.
        # The second option allows Left and Right EMG signals to be
        # normalized separately when their amplitudes are very different.
        if mu_x is not None and sigma_x is not None:
            # mu_x/sigma_x can be either:
            #   - a single Series/array -> one global scale applied to every row, or
            #   - a dict {side_name: Series/array} -> per-side normalization,
            #     since Left/Right EMG amplitude can differ a lot for some
            #     channels (e.g. RF/TA), and a single pooled scale would
            #     squash the smaller side toward zero after z-scoring.
            if isinstance(mu_x, dict):
                for side in np.unique(np.asarray(self.side_labels)):
                    # Create a Boolean mask selecting only rows from this side.
                    mask = self.side_labels == side
                    # Get the training mean for this side.
                    m_x = np.array(mu_x[side].values if hasattr(mu_x[side], 'values') else mu_x[side], dtype=np.float32)
                    # Get the training mean for this side.
                    s_x = np.array(sigma_x[side].values if hasattr(sigma_x[side], 'values') else sigma_x[side], dtype=np.float32)
                    # Z-score normalize only the rows belonging to this side.
                    # 1e-6 prevents division by zero if a channel has an extremely small/zero standard deviation.
                    self.X[mask] = (self.X[mask] - m_x) / (s_x + 1e-6)
            else:
                # If mu_x is not a dictionary, use one set of mean/std values for all rows.
                m_x = np.array(mu_x.values if hasattr(mu_x, 'values') else mu_x, dtype=np.float32)
                s_x = np.array(sigma_x.values if hasattr(sigma_x, 'values') else sigma_x, dtype=np.float32)
                self.X = (self.X - m_x) / (s_x + 1e-6)

        # Normalize target (knee angle)
        # mu_y and sigma_y are the mean and standard deviation of the
        # KNEE ANGLE calculated from the TRAINING data. We save them because they may later be needed to convert
        # normalized predictions back to degrees.
        self._mu_y = np.float32(mu_y) if mu_y is not None else None
        self._sigma_y = np.float32(sigma_y) if sigma_y is not None else None
        mu, sigma = self._mu_y, self._sigma_y
        self.y = np.asarray(data['Knee_Angle_X'].values, dtype=np.float32)
        # Normalize the knee angle using TRAINING mean/std.
        if mu is not None and sigma is not None:
            self.y = (self.y - mu) / sigma
        # Find valid window starting positions
        # A window must stay completely inside the same trial AND side.
        # For example, we do NOT want a 25-sample window like:
        # Trial 1: samples 90-99
        # Trial 2: samples 0-14
        # because those samples are not one continuous sequence.
        if 'trial_id' in data.columns:
            group_keys = list(zip(data['trial_id'].values, data['Side'].values))
        else:
            # If trial_id does not exist, treat the entire dataframe as one continuous sequence.
            group_keys = [0] * len(data)
        # This list will contain the starting row of every valid window.
        # Example: valid_starts = [0, 1, 2, 3, 10, 11, ...] Each number represents ONE possible window.
        self.valid_starts = []
        n = len(data)
        block_start = 0
        for i in range(1, n + 1):
            if i == n or group_keys[i] != group_keys[block_start]:
                block_len = i - block_start
                last_valid_local_start = block_len - window_size
                if last_valid_local_start >= 0:
                    self.valid_starts.extend(
                        block_start + s for s in range(last_valid_local_start + 1)
                    )
                block_start = i

    def __len__(self) -> int:
        return len(self.valid_starts)

    def __getitem__(self, i) -> Tuple[torch.Tensor, torch.Tensor]:
        idx = self.valid_starts[i]
        window_X = self.X[idx: idx + self.window_size].copy()
        center = idx + self.window_size // 2

        target_y = self.y[center]

        return torch.tensor(window_X), torch.tensor(np.float32(target_y))

def split_by_trial(df: pd.DataFrame, test_frac: float = 0.2, seed: int = 0):
    if "trial_id" not in df.columns:
        split = int(len(df) * (1 - test_frac))
        return df.iloc[:split], df.iloc[split:]

    trial_ids = sorted(df["trial_id"].unique())
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(trial_ids)
    n_test = max(1, int(round(len(shuffled) * test_frac)))
    test_trials = set(shuffled[:n_test])

    train_df = df[~df["trial_id"].isin(test_trials)]
    test_df = df[df["trial_id"].isin(test_trials)]
    return train_df, test_df

def get_mlr_data(participants_list: list):
    all_dfs = []
    for sub in participants_list:
        path = config.Output_path / f"{sub}_ready.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing file: {path}")
        df = pd.read_csv(path)
        all_dfs.append(df)

    combined_df = pd.concat(all_dfs, ignore_index=True)
    drop_cols = [c for c in ['Knee_Angle_X', 'Side', 'trial_id', 'Frame'] if c in combined_df.columns]
    x = combined_df.drop(columns=drop_cols).values
    y = combined_df['Knee_Angle_X'].values
    return x, y

