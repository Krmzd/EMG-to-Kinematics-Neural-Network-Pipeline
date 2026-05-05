import pandas as pd
import numpy as np
import os
from scipy.signal import butter, filtfilt
from typing import Tuple, Dict, List
from pathlib import Path

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Preprocess:
    """
    Handles preprocessing pipeline for EMG and Kinematic data.
    Includes filtering, syncing, and normalization.
    """

    def __init__(self, cut_off=6.0, fs_emg=1000, fs_kin=100):
        self.cut_off = cut_off
        self.fs_emg = fs_emg
        self.fs_kin = fs_kin
        self.ratio = fs_emg // fs_kin

        if fs_emg % fs_kin != 0:
            raise ValueError(
                f"EMG ({fs_emg}) must be a multiple of Kinematics ({fs_kin})"
            )

        self.ratio = fs_emg // fs_kin

    def butter_lowpass_filter(self, data: pd.Series) -> pd.Series:

        nyq = 0.5 * self.fs_emg
        filter_cutoff = self.cut_off / nyq
        b, a = butter(N=4, Wn=filter_cutoff, btype="low", analog=False)
        return filtfilt(b, a, data)

    def find_keyword(self, file_path: str, keyword: str) -> int:

        col0 = pd.read_csv(file_path, usecols=[0], header=None, low_memory=False)
        matches = col0.index[col0.iloc[:, 0] == keyword].tolist()

        if not matches:
            logger.error(f"Keyword '{keyword}' not found in {file_path.name}")
            raise ValueError(f"'{keyword}' missing in data file.")
        return matches[0]

    def handle_nans(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.interpolate(method="linear", limit_direction="both").ffill().bfill()

    def _z_score_data(self, df_list: List[pd.DataFrame]) -> Tuple[pd.DataFrame, Dict]:
        """Subject-level normalization logic."""
        combined = pd.concat(df_list, ignore_index=True)
        # Calculate stats
        mu = combined.mean()
        sigma = combined.std()
        stats = {col: {"mean": mu[col], "std": sigma[col]} for col in combined.columns}
        normalized_df = (combined - mu) / sigma
        return normalized_df, stats

    def process_emg(self, emg_df: pd.DataFrame) -> pd.DataFrame:
        """Full EMG chain: NaNs -> Rectify -> Filter -> Downsample."""

        emg_df = self._handle_nans(emg_df)  # Fix NaNs
        emg_rectified = emg_df.abs()  # Rectify (Absolute Value)
        emg_filtered = emg_rectified.apply(
            self._butter_lowpass_filter
        )  # Low-pass Filter (Linear Envelope)

        return (
            emg_filtered.groupby(np.arange(len(emg_filtered)) // self.ratio)
            .mean()
            .reset_index(drop=True)
        )  # Downsample (Take mean of every 10 rows)

    def extract_synced_trial(
        self, file_path: Path, feature_map: Dict, target_map: Dict
    ) -> pd.DataFrame:
        # Locate the anchors
        emg_idx = self._find_keyword(file_path, "Devices")
        joint_idx = self._find_keyword(file_path, "Joints")
        model_idx = self._find_keyword(file_path, "Model Outputs")

        # Extract EMG
        emg_header = (
            pd.read_csv(file_path, skiprows=emg_idx + 2, nrows=1, header=None)
            .iloc[0]
            .tolist()
        )

        emg_col_indices = []

        for muscle_name in feature_map.keys():
            try:
                idx = emg_header.index(muscle_name)
                emg_col_indices.append(idx)
            except ValueError:
                logger.error(f"Muscle '{muscle_name}' not found in {file_path.name}")
                raise

        num_emg_rows = joint_idx - emg_idx - 5
        emg_raw = pd.read_csv(
            file_path, usecols=emg_col_indices, skiprows=emg_idx + 4, nrows=num_emg_rows
        )
        emg_clean = self.process_emg(emg_raw).rename(
            columns=feature_map
        )  # Process: NaNs, -> Rectifu -> Filter -> Downsample

        # Read the header row for Model Outputs
        kin_header = (
            pd.read_csv(file_path, skiprows=model_idx + 2, nrows=1, header=None)
            .iloc[0]
            .tolist()
        )

        kin_col_indices = []
        for kin_name in target_map.keys():
            try:
                idx = kin_header.index(kin_name)
                kin_col_indices.append(idx)
            except ValueError:
                logger.error(f"Angle '{kin_name}' not found in {file_path.name}")
                raise

        kin_raw = pd.read_csv(
            file_path, usecols=target_map, skiprows=model_idx + 4
        )  # Extract the target
        kin_clean = (
            self._handle_nans(kin_raw).reset_index(drop=True).rename(columns=target_map)
        )  # clean kinematics

        # Final sync
        min_len = min(len(emg_clean), len(kin_clean))  # Sync lengths and stitch
        final_trial = pd.concat(
            [emg_clean.iloc[:min_len], kin_clean.iloc[:min_len]], axis=1
        )
        return final_trial

    def process_subject(
        self, folder_path: str, side_config: Dict
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Main coordinator:
         - Loops through trials to extract Right and Left data separately.
         - Calculates independent Z-scores for each side using global subject stats.
         - Stacks them into one Master Table.
        """
        folder = Path(folder_path)
        files = list(folder.glob("*.csv"))

        # Two separate lists to handle independent side Z-scoring
        right_trials = []
        left_trials = []

        for f in files:
            try:
                right_trials.append(
                    self.extract_synced_trial(
                        f,
                        side_config["Right"]["features"],
                        side_config["Right"]["target"],
                    )
                )
                left_trials.append(
                    self.extract_synced_trial(
                        f,
                        side_config["Left"]["features"],
                        side_config["Left"]["target"],
                    )
                )
            except Exception as e:
                logger.error(f"Skipping file {f.name} due to error: {e}")

            df_r_norm, stats_r = self._z_score_data(right_trials)
            df_r_norm["Side"] = "Right"

            df_l_norm, stats_l = self._z_score_data(left_trials)
            df_l_norm["Side"] = "Left"

            subject_df = pd.concat([df_r_norm, df_l_norm], ignore_index=True)
            return subject_df, {"Right": stats_r, "Left": stats_l}
