"""Create a clean ParaView-like full-block partition view from the saved trace."""
from pathlib import Path
import sys
PV = Path(r"C:\Program Files\ParaView 6.2.0")
SITE = PV / "bin" / "Lib" / "site-packages"
sys.path.insert(0, str(SITE))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[1]
d = np.load(ROOT / "results" / "bauxitemed_adaptive_trace.npz")
out = ROOT / "results" / "paraview_bauxitemed_trace" / "figures" / "bauxitemed_full_block_model_reference_style.png"
out.parent.mkdir(parents=True, exist_ok=True)

# Categorical swatch order and high-saturation colours follow the supplied reference.
colors = ["#ff8046", "#ff0000", "#00ef00", "#0000ff", "#ffff00",
          "#ff00ff", "#00eaea", "#9999ff", "#a77b5b"]
cmap = ListedColormap(colors)
norm = BoundaryNorm(np.arange(-0.5, 9.5, 1), cmap.N)
fig = plt.figure(figsize=(16, 9), dpi=180, facecolor="white")
ax = fig.add_axes([0.015, 0.02, 0.80, 0.96], projection="3d")
ax.scatter(d["x"], d["y"], d["z"], c=d["partition_id"], cmap=cmap, norm=norm,
           s=0.42, marker=".", linewidths=0, depthshade=False, antialiased=False)
ax.view_init(elev=19, azim=-56)
ax.set_xlim(0,120); ax.set_ylim(0,120); ax.set_zlim(0,25)
ax.set_box_aspect((120,120,40))
ax.set_axis_off()
ax.set_facecolor("white")
for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
    axis.pane.fill = False
    axis.pane.set_edgecolor("white")
    axis.line.set_color("white")

# Swatches only, like the on-screen ParaView scalar legend in the reference.
leg = fig.add_axes([0.958, 0.30, 0.012, 0.40])
leg.set_xlim(0,1); leg.set_ylim(0,9); leg.axis("off")
for i, color in enumerate(colors):
    leg.add_patch(Rectangle((0,8-i),1,0.88,facecolor=color,edgecolor="white",linewidth=0.6))

fig.savefig(out, facecolor="white", dpi=180)
plt.close(fig)
print(out)
print(f"blocks={len(d['x']):,}; partitions={int(d['partition_id'].max()+1)}")
