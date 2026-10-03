import numpy as np
import pandas as pd
import json
from pathlib import Path

from gpit.pseudoflow import Pseudoflow


# -----------------------
# Dataset
# -----------------------

DATA = Path("data/BauxiteMed/bauxitemed.dat")

NX = 120
NY = 120
NZ = 26


# -----------------------
# Load EPV
# -----------------------

flat = np.loadtxt(DATA, dtype=np.float64)

print(f"Loaded blocks: {len(flat):,}")


df = pd.DataFrame({
    "EPV": flat
})


# -----------------------
# Block parameters
# -----------------------

bmparms = {
    "nx": NX,
    "ny": NY,
    "nz": NZ,
    "xorg": 0,
    "yorg": 0,
    "zorg": 0,
    "xs": 1,
    "ys": 1,
    "zs": 1,
    "volume": 1,
}


# -----------------------
# Solve UPL
# -----------------------

print("\nRunning Pseudoflow UPL...")

solver = Pseudoflow(df, bmparms)

solver.UPL(
    EPV_column="EPV",
    precedence="1x5",
    uplName="baseline"
)


pit = solver.resultUpl["baseline"]


# -----------------------
# Results
# -----------------------

blocks_inside = np.sum(pit)

objective = np.sum(flat[pit == 1])


print("\n========== BASELINE ==========")
print(f"Blocks selected : {blocks_inside:,.0f}")
print(f"Objective value : {objective:,.2f}")
print("==============================")



# -----------------------
# Save result
# -----------------------

Path("results").mkdir(exist_ok=True)


np.save(
    "results/bauxitemed_baseline_pit.npy",
    pit
)


with open(
    "results/bauxitemed_baseline.json",
    "w"
) as f:

    json.dump(
        {
            "dataset": "bauxitemed",
            "blocks": int(blocks_inside),
            "objective": float(objective)
        },
        f,
        indent=4
    )


print("\nSaved baseline results.")