import os
import logging
from pathlib import Path
from typing import List, Dict
import joblib
from dotenv import load_dotenv
from process import Preprocess
import config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("PipelineManager")


def run_subject_pipeline(
    subjects: List[str], raw_dir: Path, output_dir: Path, side_map: Dict, cutoff: float
) -> None:
    """
    Run preprocessing for all subjects.
    Processes both sides into a single master CSV per participant.
    """
    # Initialize with parameters
    processor = Preprocess(cut_off=cutoff, fs_emg=1000, fs_kin=100)
    output_dir.mkdir(parents=True, exist_ok=True)

    success_count = 0
    for sub in subjects:
        logger.info(f"Processing subject: {sub}")
        subject_folder = raw_dir / sub

        if not subject_folder.exists():
            logger.error(f"Path not found: {subject_folder}. Skipping.")
            continue

        try:
            df_ready, stats = processor.process_subject(
                folder_path=str(subject_folder), side_config=side_map
            )

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
            logger.error(f"Critical error on {sub}: {str(e)}", exc_info=True)
    logger.info(
        f"--- Pipeline Finished. Successfully processed {success_count}/{len(subjects)} subjects. ---"
    )


def main() -> None:
    """
    Main entry point: Pulls everything from config.py and runs.
    """
    # Variables in config.py
    run_subject_pipeline(
        subjects=config.Participants,
        raw_dir=config.Raw_data_path,
        output_dir=config.Output_path,
        side_config=config.Side_map["side_map"],  # Reaching into your Side_map dict
        cutoff=config.Lowpass_cutoff,
    )


if __name__ == "__main__":
    main()
