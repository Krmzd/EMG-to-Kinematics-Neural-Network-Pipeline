import logging
import copy
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import config
import models
from data_loader import GaitDataset, split_by_trial

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")
logger = logging.getLogger("Trainer")

torch.manual_seed(0)
np.random.seed(0)

class EarlyStopping:
    def __init__(self, patience: int = 15, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = np.inf
        self.counter = 0
        self.best_weights = None

    def step(self, val_loss: float, model: nn.Module) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_weights = copy.deepcopy(model.state_dict())
        else:
            self.counter += 1
        return self.counter >= self.patience

    def restore_best(self, model: nn.Module) -> None:
        if self.best_weights is not None:
            model.load_state_dict(self.best_weights)
            logger.info(f"  Restored best weights (val_loss={self.best_loss:.4f})")

# Handles the machine-learning process for a single experiment:
# model initialization, training, validation, learning-rate scheduling,
# early stopping, and prediction on the test set.
class Engine:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Public training interface that runs the PyTorch training and prediction pipeline.
    def train_and_predict(self, train_input, test_input, val_input=None, n_features=None, dropout=None, weight_decay=None, hidden_dim=None, epochs=None, patience=None):
        return self.run_torch(train_input, test_input, val_input, n_features, dropout, weight_decay, hidden_dim=hidden_dim, epochs=epochs, patience=patience)
    # Run the full PyTorch training pipeline: initialize the model, train on the
    # training set, validate during training, apply early stopping, and predict
    # on the held-out test set.
    def run_torch(self, train_input, test_input, val_input=None, n_features=None, dropout=None, weight_decay=None, hidden_dim=None, epochs=None, patience=None):
        model_class = getattr(models, self.model_name)
        epochs = epochs if epochs is not None else config.Epochs
        dropout = dropout if dropout is not None else config.Dropout
        hidden_dim = hidden_dim if hidden_dim is not None else config.Hidden_dim
        weight_decay = weight_decay if weight_decay is not None else config.Weight_decay
        patience = patience if patience is not None else config.Early_stop_patience
        model = model_class(n_features=n_features, hidden_dim=hidden_dim, n_outputs=config.N_outputs, dropout=dropout).to(self.device)

        optimizer = optim.Adam(model.parameters(), lr=config.Learning_rate, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=5, factor=0.5)
        criterion = nn.MSELoss()
        early_stop = EarlyStopping(patience=patience)

        train_losses, val_losses = [], []

        for epoch in range(epochs):
            model.train()
            batch_losses = []
            for batch_x, batch_y in train_input:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                optimizer.zero_grad()
                pred = model(batch_x).squeeze()
                loss = criterion(pred, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                batch_losses.append(loss.item())

            avg_train_loss = np.mean(batch_losses)
            train_losses.append(avg_train_loss)

            if val_input is not None:
                avg_val_loss = self.evaluate_loss(model, val_input, criterion)
                val_losses.append(avg_val_loss)
                scheduler.step(avg_val_loss)
                if early_stop.step(avg_val_loss, model):
                    logger.info(f"Early stopping triggered at epoch {epoch+1}")
                    break

        early_stop.restore_best(model)

        model.eval()
        all_preds, all_true = [], []
        with torch.no_grad():
            for bx, by in test_input:
                bx = bx.to(self.device)
                all_preds.append(model(bx).squeeze().cpu().numpy())
                all_true.append(by.numpy())

        return (
            np.concatenate(all_true),
            np.concatenate(all_preds).flatten(),
            {"train": train_losses, "val": val_losses},
        )

    def evaluate_loss(self, model, input, criterion) -> float:
        model.eval()
        losses = []
        with torch.no_grad():
            for bx, by in input:
                bx = bx.to(self.device)
                by = by.to(self.device)
                pred = model(bx).squeeze()
                losses.append(criterion(pred, by).item())
        return float (np.mean(losses))
    
# Manages the complete intra-subject experiment for each participant:
# data splitting, normalization, dataset/DataLoader creation, model training,
# evaluation, and saving the results.
def run_experiment(model_type: str = "CNN_BiLSTM"):
    logger.info(f"Starting intra | Model: {model_type} | Feature_cols: {config.Feature_cols}")
    engine = Engine(model_type)
    results = {}

    for test_sub in config.Participants:
        logger.info(f"  Subject: {test_sub}")
        path = config.Output_path / f"{test_sub}_ready.csv"
        full_df = pd.read_csv(path)
        excluded = getattr(config, "Subject_exclude_trials", {}).get(test_sub, [])
        if excluded:
            full_df = full_df[~full_df["trial_id"].isin(excluded)]
            logger.info(f"  Excluded trials for {test_sub}: {excluded}")
        split_seed = getattr(config, "Split_seed", 1)

        df_train, df_test = split_by_trial(full_df, test_frac=0.2, seed=split_seed)

        # Normalization stats (from train only — prevents data leakage)
        mu_y = df_train["Knee_Angle_X"].mean()
        sigma_y = df_train["Knee_Angle_X"].std()

        sides_present = df_train["Side"].unique()

        # Only use the channels listed in config.Feature_cols - (e.g. drop RF/TA) just by editing config.py.
        emg_cols = [c for c in config.Feature_cols if c in df_train.columns]
    
        n_features = len(emg_cols)
        # Calculate EMG normalization statistics separately for each side.
        # `s` is the current side (e.g., "R" or "L") from "sides_present".
        # Select training rows where Side == s and calculate the mean/std
        mu_x    = {s: df_train.loc[df_train["Side"] == s, emg_cols].mean() for s in sides_present}
        sigma_x = {s: df_train.loc[df_train["Side"] == s, emg_cols].std()  for s in sides_present}

        # Build inputs 
        # split training data ino train and validation
        df_train_sub, df_val = split_by_trial(df_train, test_frac=0.15, seed=split_seed)
        # Restrict each split's dataframe selected emg_cols, so GaitDataset (which treats every
        # remaining column as a feature) only ever sees the channels we actually want.
        keep_cols = ["Knee_Angle_X", "Side", "trial_id", "Frame"] + emg_cols
        def restrict(df):
            return df[[c for c in keep_cols if c in df.columns]]

        # Create datasets for training, validation, and test sets
        Window_Size = config.Window_Size
        train_ds = GaitDataset(restrict(df_train_sub), Window_Size, mu_y, sigma_y, mu_x, sigma_x)
        val_ds   = GaitDataset(restrict(df_val),       Window_Size, mu_y, sigma_y, mu_x, sigma_x)
        test_ds  = GaitDataset(restrict(df_test),      Window_Size, mu_y, sigma_y, mu_x, sigma_x)
        # wrap them in DataLoader
        train_in = DataLoader(train_ds, batch_size=config.Batch_Size, shuffle=True)
        val_in   = DataLoader(val_ds,   batch_size=config.Batch_Size, shuffle=False)
        test_in  = DataLoader(test_ds,  batch_size=config.Batch_Size, shuffle=False)

        # Train & Predict 
        torch.manual_seed(0)
        np.random.seed(0)
        y_t, y_p, loss_h = engine.train_and_predict(train_in, test_in, val_in, n_features=n_features)

        results[test_sub] = {
            "true": y_t,
            "pred": y_p,
            "loss_history": loss_h,
            "stats": {"mu": mu_y, "sigma": sigma_y},
        }

    out_path = config.Output_path / f"intra_{model_type}_results.pkl"
    joblib.dump(results, out_path)
    logger.info(f"Saved → {out_path}")


if __name__ == "__main__":
    Selected_model = "CNN"
    run_experiment(model_type=Selected_model)
