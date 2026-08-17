import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import config
 

plt.style.use('seaborn-v0_8-muted')
sns.set_context("talk")
 
 
class GaitVisualizer:
    def __init__(self, evaluated_filename: str):
        self.file_path = config.Output_path / evaluated_filename
        if not self.file_path.exists():
            raise FileNotFoundError(f"Run train.py first! Missing: {self.file_path}")
 
        self.data = joblib.load(self.file_path)
 
        parts = evaluated_filename.replace("_results.pkl", "").split("_")
  
    def _get_degrees(self, subject_id: str):
        """
        Returns true and predicted values in degrees.
        Supports both new format (true_deg/pred_deg keys)
        and old format (true/pred normalized + stats for inverse transform).
        """
        d = self.data[subject_id]
 
        if "true_deg" in d and "pred_deg" in d:
            return d["true_deg"], d["pred_deg"]
 
        elif "true" in d and "stats" in d:
            mu    = d["stats"]["mu"]
            sigma = d["stats"]["sigma"]
            true_deg = (d["true"] * sigma) + mu
            pred_deg = (d["pred"] * sigma) + mu
            return true_deg, pred_deg
 
        else:
            raise KeyError(
                f"Subject '{subject_id}' has no recognizable keys. "
                f"Available keys: {list(d.keys())}"
            )
 
    # Plots
 
    def plot_time_series(self, subject_id: str, start: int = 2000, length: int = 500):
        """Plots Actual vs Predicted knee angle over time."""
        true_deg, pred_deg = self._get_degrees(subject_id)
 
        # Clamp start/length to available data
        max_len = len(true_deg)
        start  = min(start, max_len)
        length = min(length, max_len - start)
 
        plt.figure(figsize=(15, 5))
        plt.plot(true_deg[start:start+length], label='Actual',    color='black', lw=2)
        plt.plot(pred_deg[start:start+length], label='Predicted', color='red', ls='--', alpha=0.8)
 
        plt.title(f"Tracking Performance")
        plt.ylabel("Knee Angle (Degrees)")
        plt.xlabel("Frames (100Hz)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
 
    def plot_gait_cycle(self, subject_id: str, side: str = "Right"):
        """Average gait cycle (0-100%) with mean +/- std shading."""
        d = self.data[subject_id]
        true_deg, pred_deg = self._get_degrees(subject_id)
 
        # Filter by side
        side_labels = d.get("side", None)
        if side_labels is not None:
            # Align lengths in case of window offset
            min_len = min(len(true_deg), len(side_labels))
            mask    = side_labels[:min_len] == side
            y_true  = true_deg[:min_len][mask]
            y_pred  = pred_deg[:min_len][mask]
        else:
            y_true = true_deg
            y_pred = pred_deg
 
        if len(y_true) == 0:
            print(f"No data found for side='{side}' in subject '{subject_id}'.")
            return
 
        # Break into 100-frame steps (approx 1 gait cycle at 100Hz)
        step_len  = 100
        num_steps = len(y_true) // step_len
 
        if num_steps == 0:
            print(f"Not enough data to plot gait cycle for subject '{subject_id}'.")
            return
 
        true_steps = y_true[:num_steps * step_len].reshape(-1, step_len)
        pred_steps = y_pred[:num_steps * step_len].reshape(-1, step_len)
 
        x      = np.linspace(0, 100, step_len)
        m_true = np.mean(true_steps, axis=0)
        s_true = np.std(true_steps,  axis=0)
        m_pred = np.mean(pred_steps, axis=0)
        s_pred = np.std(pred_steps,  axis=0)
 
        plt.figure(figsize=(10, 6))
        plt.plot(x, m_true, color='black', label='Ground Truth', lw=2)
        plt.fill_between(x, m_true - s_true, m_true + s_true, color='black', alpha=0.1)
        plt.plot(x, m_pred, color='red', label='AI Prediction', ls='--')
        plt.fill_between(x, m_pred - s_pred, m_pred + s_pred, color='red', alpha=0.1)
 
        plt.title(f"Average Gait Cycle ({side} Leg)")
        plt.xlabel("Gait Cycle (%)")
        plt.ylabel("Degrees")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    @staticmethod
    def _smooth(values, window: int = 3):
        """Simple moving-average smoothing for DISPLAY ONLY -- does not change
        the underlying loss values, just how the line is drawn."""
        values = np.asarray(values, dtype=float)
        if len(values) < window:
            return values
        kernel = np.ones(window) / window
        return np.convolve(values, kernel, mode="valid")
    def plot_learning_curve(self, subject_id: str, smooth_window: int = 3):
       
        d = self.data[subject_id]

        if "loss_history" not in d or not d["loss_history"]:
            print(f"No loss history found for '{subject_id}' (MLR has no loss curve).")
            return

        losses = d["loss_history"]

        plt.figure(figsize=(8, 5))

        def draw(values, color, label):
            plt.plot(values, color=color, lw=1, alpha=0.25)  # raw, faint
            smoothed = self._smooth(values, smooth_window)
            offset = (min(smooth_window, len(values)) - 1) / 2
            x = np.arange(len(smoothed)) + offset
            plt.plot(x, smoothed, color=color, lw=2, label=label)

        if isinstance(losses, dict):
            draw(losses["train"], "green", "Training Loss")
            if losses.get("val"):
                draw(losses["val"], "orange", "Validation Loss")
        else:
            draw(losses, "green", "Training Loss")

        plt.title(f"Learning Curve")
        plt.xlabel("Epochs")
        plt.ylabel("Loss (MSE)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

 
    def plot_all(self, subject_id: str, side: str = "Right"):
        """Convenience method: plots all three figures for a subject."""
        self.plot_time_series(subject_id)
        self.plot_gait_cycle(subject_id, side=side)
        self.plot_learning_curve(subject_id)
 
 
if __name__ == "__main__":
    My_results = "intra_CNN_results.pkl"

    plotter = GaitVisualizer(My_results)
    plotter.plot_time_series("MY", start=0, length=500)
    plotter.plot_gait_cycle("MY", side="Right")
    plotter.plot_learning_curve("MY")