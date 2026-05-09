import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import joblib
import logging

import config
import models
import data_loader

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")
logger = logging.getLogger("Trainer")

class Engine:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def train_and_predict(self, train_input, test_input):
        """
        This part decides which internal engine to run based on the model name
        """
        if self.model_name == "MLR":
            # Pass simple Arrays to the MLR worker
            return self.run_mlr_logic(train_input, test_input)
        else:
            # Pass DataLoaders to the PyTorch worker
            return self.run_torch_logic(train_input, test_input)
        
    # The MLR 
    def run_mlr(self, train_data, test_data):
        x_train, y_train = train_data
        x_test, y_test = test_data

        # Get model from models.py
        model = models.get_mlr_model()
        model.fit(x_train, y_train)

        y_pred = model.predict(x_test)
        return y_test, y_pred
    
    # Deep learning
    def run_torch(self, train_loader, test_loader):
        # Initialize the correct PyTorch model from model.py
        # 'getattr' to find the class name automatically
        model_class = getattr(models, self.model_name)

        # Initialize it using the settings from config file
        model = model_class(n_features=config.N_features, hidden_dim=config.Hidden_dim, n_outputs=config.N_outputs).to(self.device)

        optimizer = optim.Adam(model.parameters(), lr=config.Learning_rate)
        criterion = nn.MSELoss()

        # Training Loop 
        for epoch in range(20):
            model.train()
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                loss = criterion(model(batch_x).squeeze(), batch_y)
                loss.backward()
                optimizer.step()

        # Prediction phase
        model.eval()
        all_preds, all_true = [], []
        with torch.no_grad():
            for bx, by in test_loader:
                bx = bx.to(self.device)
                all_preds.append(model(bx).cpu().numpy())
                all_true.append(by.numpy())

        return np.concatenate(all_true), np.concatenate(all_preds).flatten()
    
def run_loso_exp(model_type="MLR"):
    logger.info(f"Starting LOSO experiment using {model_type}")
    engine = Engine(model_type)
    results = {}

    for test_sub in config.Participants:
        train_subs = [s for s in config.Participants if s != test_sub]

        if model_type == "MLR":

            # We need the 'Side' column before we drop it for the math
            path = config.Output_path / f"{test_sub}_combined_ready.csv"
            test_df = pd.read_csv(path)
            side_labels = test_df['Side'].values # Grab side
            # Get Arrays
            train_input = data_loader.get_mlr_data(train_subs)
            test_input = data_loader.get_mlr_data([test_sub])
        else:
            path = config.Output_path / f"{test_sub}_combined_ready.csv"
            side_labels = pd.read_csv(path)['Side'].values[config.Window_Size:] # Match window offset
            # Get DataLoaders
            train_input = data_loader.get_torch_loader(train_subs, config.Window_Size, config.Batch_Size, shuffle=True)
            test_input  = data_loader.get_torch_loader([test_sub], config.Window_Size, config.Batch_Size, shuffle=False)

        y_true, y_pred = engine.train_and_predict(train_input, test_input)
        results[test_sub] = {"true": y_true, "pred": y_pred, "side": side_labels}

    joblib.dump(results, config.Output_path / f"loso_{model_type}_results.pkl")
    logger.info(f"LOSO experiment for {model_type} finished and saved.")

# Within-Subject (Intra-Subject)
def run_intra_subject_exp(model_type="MLR"):
    logger.info(f"Starting Intra-Subject Experiment using {model_type}")
    engine = Engine(model_type)
    results = {}

    for sub in config.Participants:
        # Load the labels for the specific participant
        path = config.Output_path / f"{sub}_combined_ready.csv"
        df = pd.read_csv(path)
        split = int(len(df) * 0.8)

        if model_type == "MLR":
            x, y = data_loader.get_mlr_data([sub])
            train_input = (x[:split], y[:split])
            test_input = (x[split:], y[split:])
            side_labels = df['Side'].values[split:] # Grab side labels from index 'split' to the end
        else:
            # For Torch, we split the CSV first (simplest way)
            train_input, test_input = data_loader.get_intra_loader(sub, config.Window_Size, config.Batch_Size)
            side_labels = df['Side'].values[split + config.Window_Size:]# Match window offset (skip first 50 frames of the test set)
        y_true, y_pred = engine.train_and_predict(train_input, test_input)
        results[sub] = {"true": y_true, "pred": y_pred, "side": side_labels}

    joblib.dump(results, config.Output_path / f"intra_{model_type}_results.pkl")
    logger.info(f"Intra-Subject experiment for {model_type} finished and saved.")

if __name__ == "__main__":
    # CHOOSE YOUR MODEL: "MLR", "CNN_LSTM", "BiLSTM_CNN", "CNN_BiLSTM_Attention"
    My_model = "MLR" 

    # CHOOSE YOUR EXPERIMENT (Uncomment only one):
    run_loso_exp(model_type=My_model)
    # run_intra_subject_experiment(model_type=MY_MODEL)