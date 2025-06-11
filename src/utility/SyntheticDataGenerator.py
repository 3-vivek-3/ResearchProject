from pathlib import Path
import numpy as np, pickle, gzip

H = 500 # horizon
N = 10 # actions
TRIALS = 30 # trials
D = 100 # ambient dimension
SPARSITY = 0.9

action_sets = []
thetas = []

rng = np.random.default_rng(seed=42)

for _ in range(TRIALS):
    A = rng.normal(size=(H, N, D))        # shape: (t, a, d)

    s = int(np.round((1 - SPARSITY) * D))
    nz_idx = rng.choice(D, size = s, replace=False)
    theta = np.zeros((D, 1))
    theta[nz_idx, 0] = rng.standard_normal(s)

    #theta = rng.choice([0, 1], size=(D, 1), p=[1-SPARSITY, SPARSITY]) * rng.normal(size=(D, 1))
    
    action_sets.append(A)
    thetas.append(theta)

out = {"action_sets": action_sets, "thetas": thetas}

# ensure directory exists
output_dir = Path("data")
output_dir.mkdir(parents=True, exist_ok=True)

# write to gzip‐compressed pickle
file_path = output_dir / f"env_{H}x{N}x{D}_{(int)(SPARSITY * 100)}%_sparsity_{TRIALS}_trials.pkl.gz"
with gzip.open(file_path, "wb") as f:
    # protocol=pickle.HIGHEST_PROTOCOL is recommended but optional
    pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)

print(f"Wrote {len(action_sets)} trials to {file_path}")
