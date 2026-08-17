"""
cross_validation.py
Runs the real intra experiment multiple times, each with a different random
train/val/test split (via config.Split_seed), and averages the resulting
R2 for each subject. This gives a much more trustworthy number than a
single fixed split, especially with small trial counts where which trials
happen to land in the test set can swing R2 a lot on its own.

Requires train.py and evaluation.py to be importable (run from the same
folder as those files), and requires train.py to already support
config.Split_seed.
"""
import numpy as np
import config
import train
import evaluation

MODEL_TYPE = "CNN_BiLSTM"
EXP_TYPE = "INTRA"
N_FOLDS = 5
SEEDS = list(range(N_FOLDS))   # seeds 0,1,2,3,4 -- five different random splits


RESULTS_FILENAME = "intra_CNN_BiLSTM_results.pkl"

all_results = {sub: [] for sub in config.Participants}

for seed in SEEDS:
    print(f"\n Fold seed={seed}")
    config.Split_seed = seed
    train.run_experiment(exp_type=EXP_TYPE, model_type=MODEL_TYPE)
    critic = evaluation.Evaluator(RESULTS_FILENAME)
    report = critic.run_evaluation()
    for sub in config.Participants:
        row = report[report["Subject"] == sub].iloc[0]
        all_results[sub].append(row["R2"])
        print(f"  {sub}: R2={row['R2']:.4f}")

print("\n Cross-validation summary (5 folds)")
for sub in config.Participants:
    r2s = np.array(all_results[sub])
    print(f"{sub}: individual R2s = {[round(r, 4) for r in r2s]}")
    print(f"{sub}: mean R2 = {r2s.mean():.4f}  (std = {r2s.std():.4f})")

