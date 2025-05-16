# generate_offline.py
from pathlib import Path
import numpy as np, pickle, gzip

H = 500 # horizon
K = 50 # actions
TRIALS = 20 # trials
D = 200 # ambient dimension
SPARSITY = 0.1

action_sets = []
thetas       = []

rng = np.random.default_rng(seed=42)

for _ in range(TRIALS):
    A = rng.normal(size=(H, K, D))        # shape: (t, a, d)
    theta = rng.choice([0, 1], size=(D, 1), p=[1-SPARSITY, SPARSITY]) * rng.normal(size=(D, 1))
    action_sets.append(A)
    thetas.append(theta)

out = {"action_sets": action_sets, "thetas": thetas}

# ensure directory exists
output_dir = Path("data")
output_dir.mkdir(parents=True, exist_ok=True)

# write to gzip‐compressed pickle
file_path = output_dir / f"fixed_env_{H}x{K}x{D}.pkl.gz"
with gzip.open(file_path, "wb") as f:
    # protocol=pickle.HIGHEST_PROTOCOL is recommended but optional
    pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)

print(f"Wrote {len(action_sets)} trials to {file_path}")
