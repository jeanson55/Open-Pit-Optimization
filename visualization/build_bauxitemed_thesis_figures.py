"""Build thesis figures and a ParaView-ready VTP from the recorded adaptive trace.

This is a visualisation/export utility only; it does not rerun optimisation.
Run with the workspace Python runtime, which loads VTK/Matplotlib from ParaView.
"""
from pathlib import Path
import json
import os
import sys

PV = Path(r"C:\Program Files\ParaView 6.2.0")
os.add_dll_directory(str(PV / "bin"))
SITE = PV / "bin" / "Lib" / "site-packages"
sys.path.insert(0, str(SITE))
os.add_dll_directory(str(SITE / "vtkmodules"))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from vtkmodules.vtkCommonCore import vtkPoints
from vtkmodules.vtkCommonDataModel import vtkPolyData, vtkCellArray
from vtkmodules.util import numpy_support
from vtkmodules.vtkIOXML import vtkXMLPolyDataWriter
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
TRACE = ROOT / "results" / "bauxitemed_adaptive_trace.npz"
OUT = ROOT / "results" / "paraview_bauxitemed_trace"
FIG = OUT / "figures"


def export_vtp(d):
    n = len(d["x"])
    poly = vtkPolyData()
    points = vtkPoints()
    xyz = np.column_stack((d["x"], d["y"], d["z"])).astype(np.float32)
    points.SetData(numpy_support.numpy_to_vtk(xyz, deep=True))
    poly.SetPoints(points)
    # One vertex per real block.  The VTP remains point-based, not a filled grid.
    offsets = numpy_support.numpy_to_vtkIdTypeArray(np.arange(n + 1, dtype=np.int64), deep=True)
    connectivity = numpy_support.numpy_to_vtkIdTypeArray(np.arange(n, dtype=np.int64), deep=True)
    verts = vtkCellArray(); verts.SetData(offsets, connectivity); poly.SetVerts(verts)
    for key in ("node_id", "epv", "partition_id", "local_selected", "repaired_selected",
                "final_selected", "initial_violation", "repair_added", "repair_level",
                "boundary_member", "boundary_added", "boundary_removed"):
        arr = numpy_support.numpy_to_vtk(np.asarray(d[key]), deep=True)
        arr.SetName(key); poly.GetPointData().AddArray(arr)
    w = vtkXMLPolyDataWriter(); w.SetFileName(str(OUT / "bauxitemed_adaptive_trace.vtp"))
    w.SetInputData(poly); w.SetDataModeToAppended(); w.EncodeAppendedDataOn()
    if w.Write() != 1:
        raise RuntimeError("Failed to write ParaView VTP trace")


def render_isometric_blocks(d, mask, filename, title, subtitle):
    """Draw exposed unit-block faces with an isometric projection, without terrain inference."""
    nx, ny, nz = [int(v) for v in d["dimensions"]]
    occupied = np.zeros((nz, ny, nx), dtype=bool)
    partition = np.full((nz, ny, nx), -1, dtype=np.int8)
    sel = np.asarray(mask, dtype=bool)
    xs, ys, zs = d["x"][sel], d["y"][sel], d["z"][sel]
    occupied[zs, ys, xs] = True
    partition[zs, ys, xs] = d["partition_id"][sel]
    palette = [(39,119,171),(231,137,45),(54,139,90),(202,63,60),(139,105,180),
               (148,105,82),(214,116,174),(126,130,133),(187,181,54)]
    a0, b0, zh0 = 0.8660254, 0.5, 0.95
    # Fit the active pit itself into the canvas, avoiding a full-grid blank margin.
    corners=[]
    for xx in (xs.min(),xs.max()+1):
        for yy in (ys.min(),ys.max()+1):
            for zz in (zs.min(),zs.max()+1):
                corners.append((a0*(xx-yy), b0*(xx+yy)-zh0*zz))
    minx=min(p[0] for p in corners); maxx=max(p[0] for p in corners)
    miny=min(p[1] for p in corners); maxy=max(p[1] for p in corners)
    width,height=2200,1600
    scale=min(1450/max(maxx-minx,1),1180/max(maxy-miny,1))
    a=a0*scale; b=b0*scale; zh=zh0*scale
    origin_x=850-(minx+maxx)*scale/2; origin_y=820-(miny+maxy)*scale/2
    im = Image.new("RGB", (width,height), "white"); dr = ImageDraw.Draw(im)
    fontpath = r"C:\Windows\Fonts\arial.ttf"
    font = ImageFont.truetype(fontpath, 36); small = ImageFont.truetype(fontpath, 25)
    def project(x,y,z): return (origin_x + (x-y)*a, origin_y + (x+y)*b - z*zh)
    # Draw only visible faces; depth sorting keeps nearer voxels on top.
    faces=[]
    for x,y,z in zip(xs.tolist(),ys.tolist(),zs.tolist()):
        pid=int(partition[z,y,x]); col=palette[pid % len(palette)]
        if z+1>=nz or not occupied[z+1,y,x]:
            poly=[project(x,y,z+1),project(x+1,y,z+1),project(x+1,y+1,z+1),project(x,y+1,z+1)]
            faces.append((x+y+z,0,poly,col))
        if x+1>=nx or not occupied[z,y,x+1]:
            poly=[project(x+1,y,z),project(x+1,y+1,z),project(x+1,y+1,z+1),project(x+1,y,z+1)]
            faces.append((x+y+z,1,poly,tuple(int(c*.76) for c in col)))
        if y+1>=ny or not occupied[z,y+1,x]:
            poly=[project(x,y+1,z),project(x+1,y+1,z),project(x+1,y+1,z+1),project(x,y+1,z+1)]
            faces.append((x+y+z,2,poly,tuple(int(c*.58) for c in col)))
    faces.sort(key=lambda f:(f[0],f[1]))
    for _,_,poly,col in faces:
        points=[(round(px),round(py)) for px,py in poly]
        dr.polygon(points, fill=col, outline=(55,58,62), width=1)
    dr.text((100,70),title,font=font,fill=(20,26,34))
    dr.text((100,124),subtitle.format(count=int(sel.sum())),font=small,fill=(55,60,68))
    # Category legend
    lx,ly=1700,270
    dr.text((lx,ly-48),"Partition",font=small,fill=(35,40,45))
    for pid,col in enumerate(palette):
        y=ly+pid*43; dr.rectangle((lx,y,lx+26,y+26),fill=col,outline=(45,45,45)); dr.text((lx+38,y-1),str(pid),font=small,fill=(40,40,40))
    dr.text((90, height-75),"Source: bauxitemed_adaptive_trace.npz; bench index, not surveyed topography.",font=small,fill=(75,80,85))
    im.save(FIG/filename)


def render_partition_points(d, mask, filename, title):
    fig=plt.figure(figsize=(11,8.5),constrained_layout=True); ax=fig.add_subplot(111,projection='3d')
    mask=np.asarray(mask,dtype=bool)
    cmap=ListedColormap(plt.get_cmap('tab10').colors[:9])
    norm=BoundaryNorm(np.arange(-0.5,9.5,1),cmap.N)
    sc=ax.scatter(d['x'][mask],d['y'][mask],d['z'][mask],c=d['partition_id'][mask],cmap=cmap,norm=norm,
                  s=0.34 if mask.sum()>150000 else 0.9,marker='s',linewidths=0,depthshade=False)
    cb=fig.colorbar(sc,ax=ax,pad=0.10,shrink=0.68,ticks=range(9)); cb.set_label('Partition ID')
    axes3d(ax,f"{int(mask.sum()):,} blocks")
    fig.suptitle(title,fontsize=15,weight='bold')
    save(fig,filename)


def axes3d(ax, title):
    ax.set_title(title, fontsize=10, pad=5, weight="bold")
    ax.set_xlabel("X block", labelpad=3); ax.set_ylabel("Y block", labelpad=3); ax.set_zlabel("Bench index", labelpad=3)
    ax.view_init(elev=27, azim=-55)
    ax.set_xlim(0, 120); ax.set_ylim(0, 120); ax.set_zlim(0, 26)
    ax.set_box_aspect((120, 120, 34))
    ax.grid(False)


def scatter(ax, d, mask, color, size=1.4, alpha=0.85, **kwargs):
    return ax.scatter(d["x"][mask], d["y"][mask], d["z"][mask], c=color,
                      s=size, marker="s", alpha=alpha, depthshade=False,
                      linewidths=0, **kwargs)


def save(fig, name):
    fig.savefig(FIG / name, dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    if not TRACE.exists():
        raise FileNotFoundError(TRACE)
    OUT.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)
    d = np.load(TRACE)
    n = len(d["x"])
    # 1. Trace stages: independent panels, common coordinate extent.
    fig = plt.figure(figsize=(18, 7.2), constrained_layout=False)
    fig.subplots_adjust(left=0.015, right=0.99, bottom=0.05, top=0.82, wspace=0.08)
    stages = [("local_selected", "Local partition solution", "#d88925"),
              ("repaired_selected", "After precedence repair", "#3478a8"),
              ("final_selected", "After boundary refinement", "#36855b")]
    for j, (key, title, color) in enumerate(stages, 1):
        ax = fig.add_subplot(1, 3, j, projection="3d")
        mask = d[key].astype(bool)
        scatter(ax, d, mask, color, size=0.62, alpha=0.82)
        axes3d(ax, f"{title}\n{mask.sum():,} selected blocks")
    fig.suptitle("Bauxitemed adaptive partition–repair–refinement trace", fontsize=15, weight="bold", y=0.975)
    save(fig, "bauxitemed_trace_stages.png")

    # 2. Final pit with categorical partition identity. Values are fixed categories.
    fig = plt.figure(figsize=(10, 8), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")
    mask = d["final_selected"].astype(bool)
    cmap = ListedColormap(plt.get_cmap("tab10").colors[:9])
    norm = BoundaryNorm(np.arange(-0.5, 9.5, 1), cmap.N)
    sc = ax.scatter(d["x"][mask], d["y"][mask], d["z"][mask], c=d["partition_id"][mask],
                    cmap=cmap, norm=norm, s=0.9, marker="s", linewidths=0, depthshade=False)
    cb = fig.colorbar(sc, ax=ax, pad=0.10, shrink=0.68, ticks=range(9)); cb.set_label("Spatial partition ID")
    axes3d(ax, f"Final adaptive pit by partition\n{mask.sum():,} selected blocks")
    fig.suptitle("Bauxitemed adaptive trace — final pit, partition labels", fontsize=15, weight="bold")
    save(fig, "bauxitemed_final_pit_by_partition.png")

    # 3. Cascade points coloured by recorded level; context is a faint final-pit sample.
    fig = plt.figure(figsize=(10, 8), constrained_layout=True); ax = fig.add_subplot(111, projection="3d")
    final = d["final_selected"].astype(bool); context = np.flatnonzero(final)[::15]
    ax.scatter(d["x"][context], d["y"][context], d["z"][context], c="#c8cdd2", s=0.35, alpha=0.12, marker="s", linewidths=0, depthshade=False)
    add = d["repair_added"].astype(bool)
    sc = ax.scatter(d["x"][add], d["y"][add], d["z"][add], c=d["repair_level"][add], cmap="plasma",
                    s=1.3, marker="s", linewidths=0, depthshade=False)
    fig.colorbar(sc, ax=ax, pad=0.10, shrink=0.7, label="Repair cascade level")
    axes3d(ax, f"Precedence repair cascade\n{add.sum():,} additions; maximum level {int(d['repair_level'].max())}")
    fig.suptitle("Bauxitemed adaptive trace — repair cascade", fontsize=15, weight="bold")
    save(fig, "bauxitemed_repair_cascade.png")

    # 4. Initial violations highlighted against final selected membership.
    fig = plt.figure(figsize=(10, 8), constrained_layout=True); ax = fig.add_subplot(111, projection="3d")
    context = np.flatnonzero(final)[::10]
    ax.scatter(d["x"][context], d["y"][context], d["z"][context], c="#bfc5cb", s=0.34, alpha=0.16, marker="s", linewidths=0, depthshade=False)
    bad = d["initial_violation"].astype(bool)
    scatter(ax, d, bad, "#cf3d37", size=3.0, alpha=0.95)
    axes3d(ax, f"Initial precedence violations\n{bad.sum():,} violating blocks before repair")
    fig.suptitle("Bauxitemed adaptive trace — initial violations", fontsize=15, weight="bold")
    save(fig, "bauxitemed_initial_violations.png")

    # 5. Boundary edits distinguished without conflating membership with topography.
    fig = plt.figure(figsize=(14, 6.5), constrained_layout=True)
    addb = d["boundary_added"].astype(bool); rem = d["boundary_removed"].astype(bool)
    for j, (mask, title, color) in enumerate([(addb, "Added at boundary", "#27834a"), (rem, "Removed at boundary", "#bd3f3a")], 1):
        ax = fig.add_subplot(1, 2, j, projection="3d")
        bg = np.flatnonzero(final)[::18]
        ax.scatter(d["x"][bg], d["y"][bg], d["z"][bg], c="#c9ced3", s=0.35, alpha=0.15, marker="s", linewidths=0, depthshade=False)
        scatter(ax, d, mask, color, size=3.0, alpha=0.95)
        axes3d(ax, f"{title}\n{mask.sum():,} blocks")
    fig.suptitle("Bauxitemed adaptive trace — boundary membership changes", fontsize=15, weight="bold")
    save(fig, "bauxitemed_boundary_changes.png")

    export_vtp(d)
    all_blocks=np.ones(n,dtype=bool)
    render_partition_points(d,all_blocks,"bauxitemed_full_block_model_by_partition.png",
                            "Complete bauxitemed block model by partition")
    render_isometric_blocks(d,all_blocks,"bauxitemed_full_block_model_isometric.png",
                            "Bauxitemed full block model by partition",
                            "{count:,} real blocks | unit block faces | all partitions shown")
    render_isometric_blocks(d,d["final_selected"].astype(bool),"bauxitemed_final_pit_isometric_blocks.png",
                            "Bauxitemed adaptive trace: final pit by partition",
                            "{count:,} selected blocks | unit block faces | bench index, not surveyed topography")
    summary = {
        "dataset": "bauxitemed", "trace": str(TRACE.name), "blocks": n,
        "dimensions": [int(v) for v in d["dimensions"]],
        "partition_method": "METIS", "partition_count": int(d["partition_id"].max()+1),
        "local_selected": int(d["local_selected"].sum()),
        "repaired_selected": int(d["repaired_selected"].sum()),
        "final_selected": int(d["final_selected"].sum()),
        "initial_violations": int(d["initial_violation"].sum()),
        "repair_additions": int(d["repair_added"].sum()),
        "max_repair_level": int(d["repair_level"].max()),
        "boundary_members": int(d["boundary_member"].sum()),
        "boundary_added": int(d["boundary_added"].sum()),
        "boundary_removed": int(d["boundary_removed"].sum()),
        "initial_npv": float(d["initial_npv"][0]), "final_npv": float(d["final_npv"][0]),
        "warning": "This adaptive trace is a separate run from the nine-way Experiment E and the exact global pit. Do not combine its NPV with those experiments.",
        "figures": ["bauxitemed_full_block_model_by_partition.png", "bauxitemed_full_block_model_isometric.png", "bauxitemed_trace_stages.png", "bauxitemed_final_pit_by_partition.png", "bauxitemed_repair_cascade.png", "bauxitemed_initial_violations.png", "bauxitemed_boundary_changes.png", "bauxitemed_final_pit_isometric_blocks.png"]
    }
    (OUT / "figure_metadata.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"VTP: {OUT / 'bauxitemed_adaptive_trace.vtp'}")
    print(f"Figures: {FIG}")


if __name__ == "__main__":
    main()
