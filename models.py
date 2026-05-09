import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LinearRegression
import config 

# Multiple linear regression (Scikit-Learn)
def mlr_model():
    """
    Returns a Multiple Linear Regression model from Scikit-Learn.
    """
    return LinearRegression()

# Attention module
class Attention_module(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.attn = nn.Linear(input_dim, 1)

    def forward(self, lstm_output):
        # Calculate scores for all frames
        scores = self.attn(lstm_output) # [Batch, Time, 1]
        weights = F.softmax(scores, dim=1)
        # Sum the frames based on their 'importance' weights
        att_output = torch.sum(weights * lstm_output, dim=1)
        return att_output
    
# CNN-LSTM (PyTorch)
class CNN_LSTM(nn.Module):
    def __init__(self, n_features=config.N_features, hidden_dim=config.Hidden_dim, n_outputs=config.N_outputs):
        super().__init__()
        # CNN extracts spatial muscle patterns
        self.cnn = nn.Conv1d(in_channels=n_features, out_channels=32, kernel_size=3, padding=1)
        # LSTM handles temporal timing/delays
        self.lstm = nn.LSTM(input_size=32, hidden_size=hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, n_outputs)

    def forward(self, x):
        # x: [Batch, Time, Features] -> [Batch, Features, Time] for CNN
        x = x.permute(0, 2, 1)
        x = F.relu(self.cnn(x))
        # [Batch, Features, Time] -> [Batch, Time, Features] for LSTM
        x = x.permute(0, 2, 1)
        out, _ = self.lstm(x)
        # predict based on the very last moment of the sequence
        return self.fc(out[:, -1, :])
    
# Model Bi-directional LSTM (PyTorch)
class CNN_LSTM(nn.Module):
    def __init__(self, n_features=config.N_features, hidden_dim=config.Hidden_dim, n_outputs=config.N_outputs):
        super().__init__()
        self.cnn = nn.Conv1d(in_channels=n_features, out_channels=32, kernel_size=3, padding=1)
        self.lstm = nn.LSTM(input_size=32, hidden_size=hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, n_outputs)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = F.relu(self.cnn(x))
        x = x.permute(0, 2, 1)
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])
    
# CNN_LSTM + Attention 
class CNN_LSTM_attention(nn.Module):
    def __init__(self, n_features=config.N_features, hidden_dim=config.Hidden_dim, n_outputs=config.N_outputs):
        super().__init__()
        self.cnn = nn.Conv1d(in_channels=n_features, out_channels=32, kernel_size=3, padding=1)
        self.lstm = nn.LSTM(input_size=32, hidden_size=hidden_dim, batch_first=True)
        self.attention = Attention_module(hidden_dim) 
        self.fc = nn.Linear(in_features=hidden_dim, out_features=n_outputs)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = F.relu(self.cnn(x))
        x = x.permute(0, 2, 1)
        lstm_out, _ = self.lstm(x)
        CNN_LSTM_att_output = self.attention(lstm_out)
        return self.fc(CNN_LSTM_att_output)
    
# CNN_BiLSTM + Attention
class CNN_BiLSTM_Attention(nn.Module):
    def __init__(self, n_features=config.N_features, hidden_dim=config.Hidden_dim, n_outputs=config.N_outputs):
        super().__init__()
        self.cnn = nn.Conv1d(in_channels=n_features, out_channels=32, kernel_size=3, padding=1)
        self.lstm = nn.LSTM(input_size=32, hidden_size=hidden_dim, batch_first=True, bidirectional=True)
        self.attention = Attention_module(hidden_dim * 2) 
        self.fc = nn.Linear(in_features=hidden_dim * 2, out_features=n_outputs)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = F.relu(self.cnn(x))
        x = x.permute(0, 2, 1)
        lstm_out, _ = self.lstm(x)
        CNN_BiLSTM_att_output = self.attention(lstm_out)
        return self.fc(CNN_BiLSTM_att_output)