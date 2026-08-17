# EMG-to-Kinematics-Neural-Network-Pipeline

**Overview:** Predicting the actual joint angle enables continuous control and supports smoother, more natural real-time human-robot interaction. This project uses surface electromyography (sEMG) to predict continuous knee joint angle during walking. predicting the actual joint angle directly enables continuous, real-time results, supporting smoother human-robot interaction for applications like exoskeletons and prosthetics. This approach is grounded in the principle of myoelectric control: because a muscle generates an electrical signal before it physically contracts and produces movement,this lead time is significant for real-world robotic systems, which have their own inherent processing and actuation delays, predicting joint angle from EMG signals recorded ahead of the resulting movement can help compensate for those delays, supporting more responsive, naturally-timed device control.

**Project structure:** 
run_pipeline.py   - preprocesses raw EMG per subject
process.py        - Preprocess class: filtering, syncing
data_loader.py     - windowing, normalization, train/test splitting
train.py          - trains models, saves results
evaluation.py     - converts results to degrees, computes MAE/RMSE/R2
Visualization.py  - plots time series, gait cycle, learning curves
config.py         - all settings (paths, window size, hyperparameters)

**Setup / Installation:**
### Requirements
- Python 3.9
**Data:**
This dataset contains EMG and kinematic recordings from 3 healthy subjects (one excluded during preprocessing, leaving 2 subjects in the final analysis). Recordings were collected bilaterally (both Right and Left legs). The sampling rate for EMG is 1000 Hz, downsampled to 100 Hz during preprocessing to match the kinematic sampling rate.
EMG channels: Rectus Femoris (RF), Biceps Femoris (BF), Tibialis Anterior (TA), Soleus (SOL).
Target: Knee flexion/extension angle (`Knee_Angle_X`), recorded via motion capture at 100 Hz.

**Results**
Results below are from the INTRA-subject experiment (trial-level train/test split), reported in degrees.
| Model | MAE | RMSE | R² |
|---|---|---|---|
| CNN_BiLSTM | 3.600 | 5.941 | 0.892 |
| CNN_BiLSTM_Attention | 4.619 | 7.723 | 0.803 |
| CNN_LSTM_Attention | 4.776 | 7.664 | 0.807 |
| CNN_LSTM | 5.211 | 8.204 | 0.779 |
| LSTM  | 4.94 |  8.31  |  0.77  |
| CNN  | 5.56 |  8.49  |  0.76  |

