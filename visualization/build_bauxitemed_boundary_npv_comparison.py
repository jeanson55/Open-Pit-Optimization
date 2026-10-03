import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
add_path = RESULTS / "partition_grid_bauxitemed_spatial_k4_add.json"
boundary_path = RESULTS / "partition_grid_bauxitemed_spatial_k4_boundary.json"
with add_path.open(encoding="utf-8") as f:
    add = json.load(f)
with boundary_path.open(encoding="utf-8") as f:
    boundary = json.load(f)

add_npv = float(add["final_npv"])
boundary_npv = float(boundary["final_npv"])
gain = boundary_npv - add_npv
pct = 100.0 * gain / abs(add_npv)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 14,
    "axes.labelsize": 10,
    "svg.fonttype": "none",
})
fig, ax = plt.subplots(figsize=(8.2, 4.8), layout="constrained")
colors = ["#547A9C", "#34866D"]
bars = ax.bar([0, 1], [add_npv, boundary_npv], width=0.56, color=colors,
              edgecolor="white", linewidth=0.8, zorder=3)
ax.set_xticks([0, 1], ["Add-only repair", "Boundary-refined"])
ax.set_ylabel("Final NPV")
ax.set_title("Spatial partitioning on BauxiteMed (k = 4)", loc="left", pad=14, weight="semibold")
ax.set_ylim(0, boundary_npv * 1.17)
ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x/1e6:.0f}M" if x else "0"))
ax.grid(axis="y", color="#DDE3E8", linewidth=0.8, zorder=0)
ax.spines[["top", "right", "left"]].set_visible(False)
ax.spines["bottom"].set_color("#AAB4BE")
ax.tick_params(axis="y", length=0, colors="#55616D")
ax.tick_params(axis="x", length=0, pad=9, colors="#26323D")
for bar, value in zip(bars, [add_npv, boundary_npv]):
    ax.text(bar.get_x() + bar.get_width()/2, value + boundary_npv*0.012,
            f"{value:,.0f}", ha="center", va="bottom", fontsize=10, weight="semibold", color="#202A33")
# Bracket and measured gain callout
bracket_y = boundary_npv * 1.105
ax.plot([0, 0, 1, 1], [bracket_y*0.985, bracket_y, bracket_y, bracket_y*0.985],
        color="#46515B", linewidth=1.1, clip_on=False)
ax.text(0.5, bracket_y + boundary_npv*0.018,
        f"Gain: +{gain:,.0f}  ({pct:.3f}%)", ha="center", va="bottom",
        fontsize=10, weight="semibold", color="#26323D")
fig.text(0.5, -0.015,
         f"Both runs finished with 0 precedence violations  •  Add-only NPV: {add_npv:,.0f}  •  Boundary NPV: {boundary_npv:,.0f}",
         ha="center", va="top", fontsize=8.5, color="#52606B")
out = ROOT / "visualization"
out.mkdir(exist_ok=True)
fig.savefig(out / "bauxitemed_spatial_add_vs_boundary_npv.png", dpi=300, bbox_inches="tight", facecolor="white")
fig.savefig(out / "bauxitemed_spatial_add_vs_boundary_npv.svg", bbox_inches="tight", facecolor="white")
print(f"Saved {out / 'bauxitemed_spatial_add_vs_boundary_npv.png'}")
print(f"Saved {out / 'bauxitemed_spatial_add_vs_boundary_npv.svg'}")
print(f"add={add_npv:.0f} boundary={boundary_npv:.0f} gain={gain:.0f} pct={pct:.6f}%")
