import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

Base_dir = Path(__file__).resolve().parent
Raw_data_path = Path(os.getenv("RAW_DATA_PATH", Base_dir))
Trimmed_data_path = Path(os.getenv("TRIMMED_DATA_PATH", Base_dir))
Output_path = Path(os.getenv("OUTPUT_PATH", Base_dir / "Processed_Results"))

fs_EMG = 1000
fs_kin = 100
Lowpass_cutoff = 6.0
EMG_bandpass_low = 10.0
EMG_bandpass_high = 400.0

Batch_Size = 32
Learning_rate = 0.001
Weight_decay = 1e-4
Epochs = 100
Early_stop_patience = 30
Hidden_dim = 64
Dropout = 0.5  
Window_Size = 25
Split_seed = 1

Feature_cols = ["RF", "BF", "TA", "SOL"]
N_outputs = 1

Side_map = {
    "side_map": {
        "Right": {
            "features": {"RRF": "RF", "RBF": "BF", "RTA": "TA", "RSOL": "SOL"},
            "target": {"RKneeAngles": "Knee_Angle_X"},
        },
        "Left": {
            "features": {"LRF": "RF", "LBF": "BF", "LTA": "TA", "LSOL": "SOL"},
            "target": {"LKneeAngles": "Knee_Angle_X"},
        },
    },
}

Participants = ["MY", "JP"]
Trimmed_suffix_patterns = ("_trimmed.csv", "_knee_corrected.csv")
