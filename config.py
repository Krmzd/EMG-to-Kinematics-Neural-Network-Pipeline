from pathlib import Path
import os
from dotenv import load_dotenv


load_dotenv() 
# Directory setting
Base_dir = Path(__file__).resolve().parent
Raw_data_path = Path(os.getenv("Raw_data_path", "./Raw_Data"))
Output_path = Path(os.getenv("Output_path", "./Processed_Results"))

#  Signal processing setting
fs_EMG = 1000
fs_kin = 100
Lowpass_cutoff = 6.0

# Model hyperparameters
Window_Size = 50
Batch_Size = 32
Learning_rate = 0.001

# Model Architecture Settings
Hidden_dim = 64
N_features = 4  # 8 muscles (csv files has 4 columns)
N_outputs = 1   # 1 joint angle (Knee X)

Side_map = {
    "subjects": ["JP", "MY", "ZK"],
    "lowpass_cutoff": 6.0,
    # Column mappings for each side
    "side_map": {
        "Right": {
            "features": {
                "RRF": "RF",
                "RBF": "BF",
                "RTA": "TA",
                "RSOL": "SOL",
            },  # Right muscles
            "target": {"RKneeAngles": "Knee_Angle_X"},  # Right Knee Angle
        },
        "Left": {
            "features": {
                "LRF": "RF",
                "LBF": "BF",
                "LTA": "TA",
                "LSOL": "SOL",
            },  # Left muscles
            "target": {"LKneeAngles": "Knee_Angle_X"},
        },  # Left Knee Angle
    },
}

Participants = ["JP", "MY", "ZK"]
