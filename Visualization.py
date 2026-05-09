import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import config
from pathlib import Path

# Set professional research style
plt.style.use('seaborn-v0_8-muted')
sns.set_context("talk")

class GaitVisualizer:
    def __init__(self, evaluated_filename: str):
        self.file_path = config.Output_path / evaluated_filename
        if not self.file_path.exists():
            raise FileNotFoundError(f"Run evaluation.py first! Missing: {self.file_path}")
            
        self.data = joblib.load(self.file_path)
        
        # Parse title info from filename
        parts = evaluated_filename.split('_')
        self.exp_name = f"{parts[0].upper()} | Model: {parts[1]}"

    def plot_time_series(self, subject_id, start=2000, length=500):
        """Plots Actual vs Predicted angles for a few seconds."""
        d = self.data[subject_id]
        plt.figure(figsize=(15, 5))
        plt.plot(d['true_deg'][start:start+length], label='Actual', color='black', lw=2)
        plt.plot(d['pred_deg'][start:start+length], label='Predicted', color='red', ls='--', alpha=0.8)
        
        plt.title(f"{self.exp_name}\nTracking Performance (Subject {subject_id})")
        plt.ylabel("Knee Angle (Degrees)")
        plt.xlabel("Frames (100Hz)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

    def plot_gait_cycle(self, subject_id, side="Right"):
        """
        Creates the 0-100% gait cycle plot with mean and std dev.
        Matches the style of the images you shared!
        """
        d = self.data[subject_id]
        # Filter for the specific side
        mask = (d['side'] == side)
        y_true = d['true_deg'][mask]
        y_pred = d['pred_deg'][mask]

        # Break continuous data into 1-second steps (100 frames)
        step_len = 100 
        num_steps = len(y_true) // step_len
        
        true_steps = y_true[:num_steps*step_len].reshape(-1, step_len)
        pred_steps = y_pred[:num_steps*step_len].reshape(-1, step_len)

        # Calculate averages
        x = np.linspace(0, 100, 100)
        m_true, s_true = np.mean(true_steps, axis=0), np.std(true_steps, axis=0)
        m_pred, s_pred = np.mean(pred_steps, axis=0), np.std(pred_steps, axis=0)

        plt.figure(figsize=(10, 6))
        # Plot True
        plt.plot(x, m_true, color='black', label='Ground Truth', lw=2)
        plt.fill_between(x, m_true-s_true, m_true+s_true, color='black', alpha=0.1)
        # Plot Pred
        plt.plot(x, m_pred, color='red', label='AI Prediction', ls='--')
        plt.fill_between(x, m_pred-s_pred, m_pred+s_pred, color='red', alpha=0.1)

        plt.title(f"{self.exp_name}\nAverage Gait Cycle ({side} Leg - {subject_id})")
        plt.xlabel("Gait Cycle (%)")
        plt.ylabel("Degrees")
        plt.legend()
        plt.show()

if __name__ == "__main__":
    # Choose your evaluated file
    My_results = "loso_MLR_results_evaluated.pkl"
    
    plotter = GaitVisualizer(My_results)
    plotter.plot_time_series("JP")
    plotter.plot_gait_cycle("JP", side="Right")