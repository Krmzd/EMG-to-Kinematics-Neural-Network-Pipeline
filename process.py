import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt
from typing import Dict
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Processor")

# Columns cleaning scripts use to flag rows that should be excluded
# (glitch frames, intra-trial dips). These should be 
# removed from trimmed files, but load_trimmed_angles double-checks
# and drops any that slipped through, logging a warning if it finds any.
EXCLUDE_FLAG_COLUMNS = ("excluded_glitch_frame", "intra_trial_dip_flag")

class Preprocess:
    """
    Handles preprocessing pipeline for EMG and Kinematic data.
    Includes filtering, syncing, and normalization.
    """

    def __init__(self, cut_off=6.0, fs_emg=1000, fs_kin=100,
                 bandpass_low=10.0, bandpass_high=400.0):
        self.cut_off = cut_off
        self.fs_emg = fs_emg
        self.fs_kin = fs_kin
        self.bandpass_low = bandpass_low
        self.bandpass_high = bandpass_high

        if fs_emg % fs_kin != 0:
            raise ValueError(
                f"EMG ({fs_emg}) must be a multiple of Kinematics ({fs_kin})"
            )
        self.ratio = fs_emg // fs_kin

    # Filters

    def butter_bandpass_filter(self, data) -> np.ndarray:
       
        nyq = 0.5 * self.fs_emg
        low = self.bandpass_low / nyq
        high = min(self.bandpass_high / nyq, 0.99)  # never exceed Nyquist
        b, a = butter(N=4, Wn=[low, high], btype="band", analog=False)
        return filtfilt(b, a, data)

    def butter_lowpass_filter(self, data) -> np.ndarray:
        """
        Low-pass filter. Used AFTER rectification to build the EMG linear
        envelope. cut_off defaults to 6 Hz, the same cutoff conventionally
        used to smooth Vicon-derived joint-angle signals -- if you reuse
        this method on the knee angle, that's intentional, not a mixup.
        """
        nyq = 0.5 * self.fs_emg
        filter_cutoff = self.cut_off / nyq
        b, a = butter(N=2, Wn=filter_cutoff, btype="low", analog=False)
        return filtfilt(b, a, data)

    # Utilities 
    def find_keyword(self, file_path: Path, keyword: str) -> int:
        col0 = pd.read_csv(file_path, usecols=[0], header=None, low_memory=False)
        matches = col0.index[col0.iloc[:, 0] == keyword].tolist()
        if not matches:
            logger.error(f"Keyword '{keyword}' not found in {file_path.name}")
            raise ValueError(f"'{keyword}' missing in {file_path.name}.")
        return matches[0]

    def handle_nans(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.apply(pd.to_numeric, errors="coerce")
        return df.interpolate(method="linear", limit_direction="both").ffill().bfill()

    # EMG pipeline
    def process_emg(self, emg_df: pd.DataFrame) -> pd.DataFrame:
        """
        1. Handle NaNs
        2. Bandpass filter (10-400 Hz)  
        3. Rectify (abs value)
        4. Lowpass filter (6 Hz)          --> linear envelope
        5. Downsample (mean pooling by ratio, 1000 Hz -> 100 Hz)
        """
        emg_df = self.handle_nans(emg_df)

        emg_bandpassed = emg_df.apply(self.butter_bandpass_filter) 
        emg_rectified = emg_bandpassed.abs()
        emg_envelope = emg_rectified.apply(lambda col: self.butter_lowpass_filter(col.values))

        emg_downsampled = (
            emg_envelope
            .groupby(np.arange(len(emg_envelope)) // self.ratio)
            .mean()
            .reset_index(drop=True)
        )
        return emg_downsampled

    # Trial extraction

    def extract_emg_for_trial(self, raw_path: Path, feature_map: Dict) -> pd.DataFrame:
        """
        Pulls one side's 4 EMG channels out of the raw Vicon export's
        Devices block, runs them through process_emg, and attaches the
        true Vicon Frame number to each downsampled (100 Hz) row so it can
        be merged against the cleaned kinematics afterward.
        """
        emg_idx = self.find_keyword(raw_path, "Devices")
        joint_idx = self.find_keyword(raw_path, "Joints")

        emg_header = (
            pd.read_csv(raw_path, skiprows=emg_idx + 3, nrows=1, header=None)
            .iloc[0].tolist()
        )
        emg_header = [str(n).strip() for n in emg_header]
        emg_indices = [emg_header.index(m) for m in feature_map.keys()]

        # FIXED: row layout is marker(0), rate(1), label(2), header(3),
        # units(4), data(5+) -- true data starts at emg_idx + 5, not + 7.
        # The blank separator row sits right before "Joints", so the true
        # last data row is joint_idx - 2.
        data_start = emg_idx + 5
        num_emg_rows = joint_idx - emg_idx - 6

        start_frame = int(
            pd.read_csv(raw_path, usecols=[0], skiprows=data_start,
                        nrows=1, header=None).iloc[0, 0]
        )

        emg_raw = pd.read_csv(
            raw_path, usecols=emg_indices, skiprows=data_start,
            nrows=num_emg_rows, header=None,
        )
        emg_raw.columns = list(feature_map.values())

        emg_clean = self.process_emg(emg_raw)
        emg_clean["Frame"] = start_frame + np.arange(len(emg_clean))
        return emg_clean

    def load_trimmed_angles(self, trimmed_path: Path, target_map: Dict) -> pd.DataFrame:
        """
        Loads already-cleaned/trimmed knee-angle CSV and keeps only
        the frame column plus the one target angle for this side. trim_marker_glitches.py
        / clean_MY.py pipeline - except for a final safety-net drop of
        any row still flagged True in EXCLUDE_FLAG_COLUMNS.
        Column-name handling: looks for the plain name first (e.g.
        'RKneeAngles') and falls back to the '_corrected' variant (e.g.
        'RKneeAngles_corrected') if the plain name isn't present -- never
        '_raw'.
        """
        kin_raw = pd.read_csv(trimmed_path)

        for flag_col in EXCLUDE_FLAG_COLUMNS:
            if flag_col in kin_raw.columns:
                flagged = kin_raw[flag_col].astype(bool)
                if flagged.any():
                    logger.warning(
                        f"{trimmed_path.name}: dropping {int(flagged.sum())} row(s) "
                        f"still flagged True in '{flag_col}' (should have been "
                        f"removed upstream)."
                    )
                    kin_raw = kin_raw[~flagged]

        frame_col = "frame" if "frame" in kin_raw.columns else "Frame"

        wanted_cols = [frame_col]
        rename_map = {frame_col: "Frame"}
        for target_col, out_name in target_map.items():
            if target_col in kin_raw.columns:
                source_col = target_col
            elif f"{target_col}_corrected" in kin_raw.columns:
                source_col = f"{target_col}_corrected"
            else:
                raise ValueError(
                    f"{trimmed_path.name}: expected column '{target_col}' "
                    f"(or '{target_col}_corrected') not found. "
                    f"Available columns: {list(kin_raw.columns)}"
                )
            wanted_cols.append(source_col)
            rename_map[source_col] = out_name

        kin = kin_raw[wanted_cols].rename(columns=rename_map)
        return kin

    def extract_synced_trial(
        self, raw_path: Path, trimmed_path: Path, feature_map: Dict, target_map: Dict) -> pd.DataFrame:
        emg_clean = self.extract_emg_for_trial(raw_path, feature_map)
        kin_clean = self.load_trimmed_angles(trimmed_path, target_map)
        final_trial = pd.merge(emg_clean, kin_clean, on="Frame", how="inner")
        return final_trial

    @staticmethod
    def _find_trimmed_matches(trimmed_folder: Path, trial_id: str, suffix_patterns) -> list:
        """
        Finds every cleaned angle file belonging to one raw trial. Usually
        that's exactly one file (t01_trimmed.csv / t01_knee_corrected.csv),
        but a trial that got split into salvaged segments (e.g. t18 ->
        t18_segA, t18_segB) will have more than one match -- both need to
        be returned, not just the first, or the second segment's data is
        silently dropped.
        """
        matches = set()
        for suffix in suffix_patterns:
            matches.update(trimmed_folder.rglob(f"{trial_id}{suffix}"))
            matches.update(trimmed_folder.rglob(f"{trial_id}_seg*{suffix}"))
        if not matches:
            matches.update(trimmed_folder.rglob(f"{trial_id}*.csv"))
        return sorted(matches)

    @staticmethod
    def _segment_id(trimmed_path: Path, suffix_patterns) -> str:
        """
        Derives the trial/segment identifier for a trimmed angle file.

        Prefers the file's own 'segment_id' column when present. That's a more reliable source of truth
        than the filename. Falls back to stripping the known suffix from
        the filename otherwise, e.g. 't18_segA_trimmed.csv' -> 't18_segA',
        't02_knee_corrected.csv' -> 't02'.
        """
        try:
            header_only = pd.read_csv(trimmed_path, nrows=0)
            if "segment_id" in header_only.columns:
                first_row = pd.read_csv(trimmed_path, usecols=["segment_id"], nrows=1)
                if len(first_row):
                    return str(first_row["segment_id"].iloc[0])
        except Exception:
            pass  # fall through to filename-based derivation

        stem = trimmed_path.stem
        for suffix in suffix_patterns:
            tail = suffix[:-4]  
            if stem.endswith(tail):
                return stem[: -len(tail)]
        return stem

    def process_subject(
        self,
        raw_folder: str,
        trimmed_folder: str,
        side_config: Dict,
        trimmed_suffix_patterns=("_trimmed.csv", "_knee_corrected.csv"),
    ) -> pd.DataFrame:
        raw_folder = Path(raw_folder)
        trimmed_folder = Path(trimmed_folder)
        raw_files = sorted(raw_folder.glob("*.csv"))

        all_trials = []
        for f in raw_files:
            trial_id = f.stem  # e.g. "t01"

            trimmed_matches = self._find_trimmed_matches(trimmed_folder, trial_id, trimmed_suffix_patterns)
            if not trimmed_matches:
                logger.error(f"No cleaned angle file found for {f.name} -- skipping")
                continue

            for side_name in ["Right", "Left"]:
                # Extract this raw trial's EMG once per side, then reuse it
                # for every matching segment  avoids re-filtering the same
                # raw Devices block twice when a trial has 2+ segments.
                try:
                    emg_clean = self.extract_emg_for_trial(f, side_config[side_name]["features"])
                except Exception as e:
                    logger.error(f"Skipping {f.name} ({side_name}) -- EMG extraction failed: {e}")
                    continue

                for trimmed_path in trimmed_matches:
                    segment_id = self._segment_id(trimmed_path, trimmed_suffix_patterns)
                    try:
                        kin_clean = self.load_trimmed_angles(
                            trimmed_path, side_config[side_name]["target"]
                        )
                        trial_data = pd.merge(emg_clean, kin_clean, on="Frame", how="inner")
                        trial_data["Side"] = side_name
                        trial_data["trial_id"] = segment_id  # each segment is its own block for windowing
                        all_trials.append(trial_data)
                    except Exception as e:
                        logger.error(
                            f"Skipping {f.name} / {trimmed_path.name} ({side_name}) due to error: {e}"
                        )

        return pd.concat(all_trials, ignore_index=True)
