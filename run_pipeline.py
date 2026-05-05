import os
import logging
from pathlib import Path
from typing import List, Dict
import joblib
from dotenv import load_dotenv
from process import Preprocess


# Setup logging
logging.basicConfig(
    level = logging.INFO, 
    format = "%(asctime)s - %(levelname)s - %(message)s", datefmt= "%H:%M:%S"
)
logger = logging.getLogger("PipelineManager")

def run_subject_pipeline(
    subjects: List[str],
    raw_dir: Path,
    output_dir: Path,
    side_map: Dict,
    cutoff: float

) -> None:
    """
    Run preprocessing for all subjects. 
    Processes both sides into a single master CSV per participant.
    """
    # Initialize with parameters 
    processor = Preprocess(cut_off=cutoff, fs_emg = 1000, fs_kin = 100)
    output_dir.mkdir(parents = True, exist_ok = True)

    success_count = 0
    for sub in subjects:
        logger.info(f"Processing subject: {sub}")
        subject_folder = raw_dir / sub

        if not subject_folder.exists():
            logger.error(f"Path not found: {subject_folder}. Skipping.")
            continue

        try:
            df_ready, stats = processor.process_subject(folder_path = str(subject_folder), side_config = side_map)

            # Define output paths
            data_out = output_dir / f"{sub}_ready.csv"
            stats_out = output_dir / f"{sub}_stats.pkl"
            # Save results
            df_ready.to_csv(data_out, index=False)
            joblib.dump(stats, stats_out)

            logger.info(f"Success: {sub} saved. Total Samples: {len(df_ready)}")
            success_count += 1

        except Exception as e:
            # In a real production environment, you might log the traceback
            logger.error(f"Critical error on {sub}: {str(e)}", exc_info = True)
    logger.info(f"--- Pipeline Finished. Successfully processed {success_count}/{len(subjects)} subjects. ---")


def main() -> None:
    """
    Main entry point: Loads environment variables and triggers the pipeline.
    """
    # Use Path objects
    load_dotenv()

    # Retrieve paths from .env 
    BASE_DIR = Path(__file__).resolve().parent
    RAW_DATA_ROOT = Path(os.getenv("RAW_DATA_PATH", "./Raw_Data"))
    OUTPUT_ROOT = Path(os.getenv("OUTPUT_PATH", "./Processed_Results"))
    
    CONFIG = {
        "subjects": ["JP", "MY", "ZK"],
        "lowpass_cutoff": 6.0,
        # Column mappings for each side
        "side_map": {
            "Right": {
                "features": {"RRF": "RF", "RBF": "BF", "RTA": "TA", "RSOL": "SOL"}, # Right muscles 
                "target": {"RKneeAngles": "Knee_Angle_X"} # Right Knee Angle
            },
            "Left": {
                "features": {"LRF": "RF", "LBF": "BF", "LTA": "TA", "LSOL": "SOL"}, # Left muscles
                "target": {"LKneeAngles": "Knee_Angle_X"}}# Left Knee Angle
        }
    }

    # Verify data exists 
    if not RAW_DATA_ROOT.exists():
        logger.critical(f"FATAL: Root data directory {RAW_DATA_ROOT} does not exist.")
        return
    
    run_subject_pipeline(
        subjects = CONFIG["JP", "MY", "ZK"],
        raw_dir = RAW_DATA_ROOT,
        output_dir = OUTPUT_ROOT,
        side_config=CONFIG["side_map"],
        cutoff=CONFIG["filter_cutoff"]
    )

if __name__ == "__main__":
    main()