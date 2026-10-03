"""Generate a small synthetic bauxite block model for pipeline testing.

Produces data/BauxiteMed/:
  - blocks.csv                  (block_id,x,y,z,grade,value,cost)
  - arcs.csv                    (from_block,to_block) precedence arcs
  - economic_parameters.json

The model is intentionally small (default 14 x 14 x 5 = 980 blocks) so the
whole pipeline runs in seconds.  Swap in real data by replacing the CSVs
with the same column layout.
"""

import csv
import json
from pathlib import Path

import numpy as np

NX, NY, NZ = 14, 14, 5
DX, DY, DZ = 10.0, 10.0, 8.0          # block dimensions (m)
DENSITY = 2.6                         # t/m^3
TONNAGE = DX * DY * DZ * DENSITY      # tonnes per block

ECON = {
    "commodity_price_per_t": 32.0,        # $/t of in-situ bauxite product
    "recovery": 0.88,
    "mining_cost_per_t": 2.4,
    "processing_cost_per_t": 3.1,
    "discount_rate": 0.10,
    "block_tonnage_t": TONNAGE,
    "slope_degrees": 45.0,
}


def bid(ix, iy, iz):
    return ix + NX * (iy + NY * iz)


def generate(data_dir: Path, seed: int = 42):
    rng = np.random.default_rng(seed)

    blocks = []
    for iz in range(NZ):
        for iy in range(NY):
            for ix in range(NX):
                # grade decays with depth, with a high-grade pod near the
                # centre of the deposit
                centre = np.exp(-(((ix - NX / 2) ** 2 + (iy - NY / 2) ** 2)
                                  / (2 * 4.0 ** 2)))
                grade = (0.18 + 0.45 * centre) * np.exp(-0.12 * iz)
                grade = float(np.clip(grade + rng.normal(0, 0.03), 0.05, 0.9))
                revenue = grade * ECON["recovery"] * ECON["commodity_price_per_t"] * TONNAGE
                cost = (ECON["mining_cost_per_t"] + ECON["processing_cost_per_t"]) * TONNAGE
                blocks.append((bid(ix, iy, iz), ix * DX, iy * DY, iz * DZ,
                               round(grade, 4), round(revenue - cost, 2),
                               round(cost, 2)))

    # Precedence arcs: a block at level z requires the 3x3 pad above it
    # (z + 1) to be mined first.  Arc direction: predecessor -> successor.
    arcs = []
    for iz in range(1, NZ):
        for iy in range(NY):
            for ix in range(NX):
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        jx, jy = ix + dx, iy + dy
                        if 0 <= jx < NX and 0 <= jy < NY:
                            arcs.append((bid(jx, jy, iz), bid(ix, iy, iz - 1)))

    data_dir.mkdir(parents=True, exist_ok=True)
    with open(data_dir / "blocks.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["block_id", "x", "y", "z", "grade", "value", "cost"])
        w.writerows(blocks)
    with open(data_dir / "arcs.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["from_block", "to_block"])
        w.writerows(arcs)
    with open(data_dir / "economic_parameters.json", "w") as f:
        json.dump(ECON, f, indent=2)

    print(f"wrote {len(blocks)} blocks, {len(arcs)} arcs -> {data_dir}")


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    generate(root / "data" / "BauxiteMed")
