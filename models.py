"""
models.py
=========
All model architectures for EMG -> Knee Angle prediction.

Available:
  - mlr_model             : Multiple Linear Regression (sklearn)
  - CNN                   : CNN feature extractor
  - LSTM                  : LSTM temporal model 
  - CNN_LSTM              : CNN feature extractor + LSTM temporal model 
  - BiLSTM_CNN            : BiLSTM temporal context + CNN local feature extraction
  - CNN_LSTM_Attention    : CNN + LSTM + Attention mechanism
  - CNN_BiLSTM_Attention  : CNN + Bidirectional LSTM + Attention 
"""

import torch
import torch.nn as nn
from sklearn.linear_model import LinearRegression

# Baseline: Multiple Linear Regression
def mlr_model():
    return LinearRegression()

def xgb_model():
    import xgboost as xgb
    return xgb.XGBRegressor(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=0,
        n_jobs=-1,
    )

# Attention Module

class AttentionLayer(nn.Module):
    """
    Soft attention over LSTM time steps.
    Learns which frames in the window are most informative.
    """
    def __init__(self, hidden_size: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1)
        )

    def forward(self, lstm_output: torch.Tensor):
        # lstm_output: (batch, seq_len, hidden)
        scores = self.attention(lstm_output)            # (batch, seq_len, 1)
        weights = torch.softmax(scores, dim=1)          # (batch, seq_len, 1)
        context = (lstm_output * weights).sum(dim=1)    # (batch, hidden)
        return context, weights


#  CNN_LSTM

class CNN_LSTM(nn.Module):
    """
    CNN extracts local patterns, LSTM captures temporal dynamics.
    """
    def __init__(self, n_features: int, hidden_dim: int, n_outputs: int, dropout: float = 0.3):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(n_features, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, n_outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, n_features)
        cnn_in = x.permute(0, 2, 1)            # (batch, n_features, seq_len)
        cnn_out = self.cnn(cnn_in)              # (batch, 64, seq_len)
        lstm_in = cnn_out.permute(0, 2, 1)     # (batch, seq_len, 64)
        lstm_out, _ = self.lstm(lstm_in)        # (batch, seq_len, hidden)
        out = self.dropout(lstm_out[:, -1, :])
        return self.fc(out)     # Last time step

#  CNN_BiLSTM (CNN-first variant of BiLSTM_CNN)

class CNN_BiLSTM(nn.Module):
    """
    same CNN-first ordering as CNN_LSTM / CNN_LSTM_Attention /
    CNN_BiLSTM_Attention (CNN extracts local EMG burst features first),
    paired with a bidirectional LSTM instead of a unidirectional one.
    """
    def __init__(self, n_features: int, hidden_dim: int, n_outputs: int, dropout: float = 0.3):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(n_features, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        self.bilstm = nn.LSTM(
            input_size=64,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, n_outputs)  # *2 because bidirectional

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, n_features)
        cnn_in = x.permute(0, 2, 1)            # (batch, n_features, seq_len)
        cnn_out = self.cnn(cnn_in)              # (batch, 64, seq_len)
        lstm_in = cnn_out.permute(0, 2, 1)     # (batch, seq_len, 64)
        lstm_out, _ = self.bilstm(lstm_in)      # (batch, seq_len, hidden*2)
        out = self.dropout(lstm_out[:, -1, :])  # last time step
        return self.fc(out)                     # (batch, n_outputs)

# CNN_LSTM_Attention

class CNN_LSTM_Attention(nn.Module):
    """
    CNN first extracts local EMG burst features across the window,
    LSTM then models their temporal sequence, and Attention highlights
    which time steps are most predictive of the knee angle.

    Good when: raw EMG has low-amplitude periods (stance phase) —
    attention learns to downweight those uninformative frames.
    """
    def __init__(self, n_features: int, hidden_dim: int, n_outputs: int, dropout: float = 0.3):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(n_features, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout
        )
        self.attention = AttentionLayer(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, n_outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, n_features)
        cnn_in = x.permute(0, 2, 1)            # (batch, n_features, seq_len)
        cnn_out = self.cnn(cnn_in)              # (batch, 64, seq_len)
        lstm_in = cnn_out.permute(0, 2, 1)     # (batch, seq_len, 64)
        lstm_out, _ = self.lstm(lstm_in)        # (batch, seq_len, hidden)
        context, _ = self.attention(lstm_out)   # (batch, hidden)
        out = self.dropout(context)
        return self.fc(out)                     # (batch, n_outputs)


# CNN_BiLSTM_Attention  

class CNN_BiLSTM_Attention(nn.Module):
    """
    - CNN: extracts local EMG envelope features
    - BiLSTM: models temporal dynamics in both directions
              (past muscle activation and future context)
    - Attention: learns which frames in the gait cycle matter most

    Especially strong for low-amplitude walking EMG because
    attention can learn to amplify the subtle stance-phase signals.
    """
    def __init__(self, n_features: int, hidden_dim: int, n_outputs: int, dropout: float = 0.3):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(n_features, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        self.bilstm = nn.LSTM(
            input_size=64,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )
        # hidden_dim * 2 because bidirectional
        self.attention = AttentionLayer(hidden_dim * 2)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, n_outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, n_features)
        cnn_in = x.permute(0, 2, 1)            # (batch, n_features, seq_len)
        cnn_out = self.cnn(cnn_in)              # (batch, 64, seq_len)
        lstm_in = cnn_out.permute(0, 2, 1)     # (batch, seq_len, 64)
        lstm_out, _ = self.bilstm(lstm_in)      # (batch, seq_len, hidden*2)
        context, _ = self.attention(lstm_out)   # (batch, hidden*2)
        out = self.dropout(context)
        return self.fc(out)                     # (batch, n_outputs)


# LSTM
class LSTM(nn.Module):
    """
    Plain LSTM directly on raw EMG windows, no CNN front-end.
    Same LSTM config as CNN_LSTM (num_layers=2, unidirectional).
    """
    def __init__(self, n_features: int, hidden_dim: int, n_outputs: int, dropout: float = 0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, n_outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, n_features) -- fed directly, no CNN step
        lstm_out, _ = self.lstm(x)              # (batch, seq_len, hidden)
        out = self.dropout(lstm_out[:, -1, :])  # last time step
        return self.fc(out)                     # (batch, n_outputs)


#  CNN
class CNN(nn.Module):
    """
    Plain CNN on raw EMG windows,  Same conv stack style as
    CNN_LSTM (2x Conv1d, kernel=3, padding=1), but since there's no
    recurrent layer to summarize the sequence, uses global average
    pooling across the window before the linear head, so the model
    still sees the whole window, just without any temporal recurrence.
    each individual position's own receptive field is only ~5
    frames (2 layers of kernel=3); the pooling step is what lets this
    version use the full window at all.
    """
    def __init__(self, n_features: int, hidden_dim: int, n_outputs: int, dropout: float = 0.3):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(n_features, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, n_outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, n_features)
        cnn_in = x.permute(0, 2, 1)            # (batch, n_features, seq_len)
        cnn_out = self.cnn(cnn_in)              # (batch, hidden_dim, seq_len)
        pooled = cnn_out.mean(dim=2)            # global average pool over time
        out = self.dropout(pooled)
        return self.fc(out)                     # (batch, n_outputs)