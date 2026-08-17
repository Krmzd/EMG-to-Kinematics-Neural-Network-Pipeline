import logging
from pathlib import Path
from typing import List, Dict
from process import Preprocess
import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("PipelineManager")


def run_subject_pipeline(
    subjects: List[str],
    raw_dir: Path,
    trimmed_dir: Path,
    output_dir: Path,
    side_map: Dict,
    lowpass_cutoff: float,
    bandpass_low: float,
    bandpass_high: float,
) -> None:
    """
    Run preprocessing for all subjects.
    Processes both sides into a single master CSV per participant

    """
    processor = Preprocess(
        cut_off=lowpass_cutoff, fs_emg=1000, fs_kin=100,
        bandpass_low=bandpass_low, bandpass_high=bandpass_high,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    success_count = 0
    for sub in subjects:
        logger.info(f"Processing subject: {sub}")
        raw_folder = raw_dir / sub
        trimmed_folder = trimmed_dir / sub

        if not raw_folder.exists():
            logger.error(f"Raw folder not found: {raw_folder}. Skipping.")
            continue
        if not trimmed_folder.exists():
            logger.error(f"Trimmed-angle folder not found: {trimmed_folder}. Skipping.")
            continue

        try:
            df_ready = processor.process_subject(
                raw_folder=str(raw_folder),
                trimmed_folder=str(trimmed_folder),
                side_config=side_map,
            )

            data_out = output_dir / f"{sub}_ready.csv"
            df_ready.to_csv(data_out, index=False)

            logger.info(f"Success: {sub} saved. Total Samples: {len(df_ready)}")
            success_count += 1

        except Exception as e:
            logger.error(f"Critical error on {sub}: {str(e)}", exc_info=True)
    logger.info(
        f" Pipeline Finished. Successfully processed {success_count}/{len(subjects)} subjects."
    )


def main() -> None:
    """
    Main entry point: Pulls everything from config.py and runs.
    """
    run_subject_pipeline(
        subjects=config.Participants,
        raw_dir=config.Raw_data_path,
        trimmed_dir=config.Trimmed_data_path,
        output_dir=config.Output_path,
        side_map=config.Side_map["side_map"],
        lowpass_cutoff=config.Lowpass_cutoff,
        bandpass_low=config.EMG_bandpass_low,
        bandpass_high=config.EMG_bandpass_high,
    )


if __name__ == "__main__":
    main()
