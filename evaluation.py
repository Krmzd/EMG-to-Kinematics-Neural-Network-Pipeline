import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
import joblib
import logging

import config

logger = logging.getLogger("Evaluator")

class Evaluator:
    def __init__(self, results_filename: str):
        self.results_path = config.Output_path / results_filename
        if not self.results_path.exists():
            raise FileNotFoundError(f"Cannot find results file: {self.results_path}")
            
        self.raw_results = joblib.load(self.results_path)
        self.processed_results = {}

    def _get_subject_stats(self, subject_id):
        """Loads the mu and sigma for a specific participant."""
        stats_path = config.Output_path / f"{subject_id}_stats.pkl"
        return joblib.load(stats_path)

    def run_evaluation(self):
            """
            Loops through every subject in the results, turns Z-scores 
            into Degrees, and calculates scores.
            """
            logger.info(f"Evaluating: {self.results_path.name}")      
            summary_metrics = []

            for sub, data in self.raw_results.items():
                # Get the scaling 'Key' for the participant
                mu = data['stats']['mu']
                sigma = data['stats']['sigma']

                # Extract arrays 
                y_true_z = data['true']
                y_pred_z = data['pred']

                min_len = min(len(y_true_z), len(y_pred_z))
                y_true_z = y_true_z[:min_len]
                y_pred_z = y_pred_z[:min_len]

                # Convert Z-scores back to real Degrees
                y_true_deg = (y_true_z * sigma) + mu
                y_pred_deg = (y_pred_z * sigma) + mu

                # Calculate metrics (In Degrees!)
                mae = mean_absolute_error(y_true_deg, y_pred_deg)
                rmse = root_mean_squared_error(y_true_deg, y_pred_deg)
                r2 = r2_score(y_true_deg, y_pred_deg)

                # 4. Store
                self.processed_results[sub] = {
                    "true_deg": y_true_deg,
                    "pred_deg": y_pred_deg,
                    "mae": mae,
                    "rmse": rmse,
                    "r2": r2,
                    "loss_history": data.get('loss_history', [])
                }

                summary_metrics.append({"Subject": sub, "MAE": mae, "RMSE": rmse, "R2": r2})
                logger.info(f"Subject {sub} | MAE: {mae:.2f}° | RMSE: {rmse:.2f}° | R2: {r2:.3f}")

            # 5. Save the final results for the Visualization script
            output_name = self.results_path.stem + "_Evaluated.pkl"
            joblib.dump(self.processed_results, config.Output_path / output_name)
            
            return pd.DataFrame(summary_metrics)

if __name__ == "__main__":
    # Choose which file to evaluae
    File_to_store = "INTRA_LSTM_results.pkl"

    critic = Evaluator(File_to_store)
    report = critic.run_evaluation()
    
    print("\n" + "="*40)
    print(report)  
    print("Final Global Avg")
    print(report.mean(numeric_only=True))
    print("="*40)