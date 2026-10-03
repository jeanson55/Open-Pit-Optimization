# ParaView Python script for the verified Bauxitemed adaptive LG/HPF trace.
# The trace contains only the 240,500 real blocks, not the full 120*120*26
# Cartesian cell volume. Therefore the export uses vtkPolyData points rather
# than vtkImageData cells, avoiding tuple/cell-count mismatch.
#
# Run:
#   pvpython visualization/paraview_bauxitemed.py \
#       --trace results/bauxitemed_adaptive_trace.npz

import argparse
import os
import numpy as np

from vtkmodules.vtkCommonCore import vtkPoints
from vtkmodules.vtkCommonDataModel import vtkPolyData, vtkCellArray
from vtkmodules.util import numpy_support
from vtkmodules.vtkIOXML import vtkXMLPolyDataWriter


def add_point_array(poly, name, arr):
    vtk_arr = numpy_support.numpy_to_vtk(
        np.asarray(arr), deep=True
    )
    vtk_arr.SetName(name)
    poly.GetPointData().AddArray(vtk_arr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True,
                    help="results/bauxitemed_adaptive_trace.npz")
    ap.add_argument("--outdir",
                    default="results/paraview_bauxitemed")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    d = np.load(args.trace)

    x = np.asarray(d["x"], dtype=np.float64)
    y = np.asarray(d["y"], dtype=np.float64)
    z = np.asarray(d["z"], dtype=np.float64)

    n = len(x)
    if not (len(y) == len(z) == n):
        raise ValueError("x, y and z arrays have different lengths")

    # Each real block is represented by one ParaView point.
    points = vtkPoints()
    xyz = np.column_stack((x, y, z))
    points.SetData(numpy_support.numpy_to_vtk(xyz, deep=True))

    # Create one vertex cell per block so every point is selectable/renderable.
    verts = vtkCellArray()
    for i in range(n):
        verts.InsertNextCell(1)
        verts.InsertCellPoint(i)

    poly = vtkPolyData()
    poly.SetPoints(points)
    poly.SetVerts(verts)

    arrays = [
        "node_id",
        "epv",
        "partition_id",
        "local_selected",
        "repaired_selected",
        "final_selected",
        "initial_violation",
        "repair_added",
        "repair_level",
        "boundary_member",
        "boundary_added",
        "boundary_removed",
    ]

    for name in arrays:
        if name in d:
            add_point_array(poly, name, d[name])

    # Convenience arrays for ParaView thresholding.
    add_point_array(
        poly, "boundary_changed",
        (d["boundary_added"].astype(np.uint8) |
         d["boundary_removed"].astype(np.uint8))
    )

    add_point_array(
        poly, "selected_transition",
        (d["final_selected"].astype(np.int8) -
         d["local_selected"].astype(np.int8))
    )

    writer = vtkXMLPolyDataWriter()
    output = os.path.join(
        args.outdir, "bauxitemed_adaptive_trace.vtp"
    )
    writer.SetFileName(output)
    writer.SetInputData(poly)
    writer.SetDataModeToAscii()

    if writer.Write() != 1:
        raise RuntimeError("ParaView VTP writer failed")

    # Export useful subsets as CSV.
    header = "node_id,x,y,z,epv,partition_id,repair_level"

    subsets = {
        "final_pit": d["final_selected"].astype(bool),
        "repair_cascade": d["repair_added"].astype(bool),
        "boundary_changes": (
            d["boundary_added"].astype(bool) |
            d["boundary_removed"].astype(bool)
        ),
        "initial_violations": d["initial_violation"].astype(bool),
    }

    for label, mask in subsets.items():
        idx = np.flatnonzero(mask)
        rows = np.column_stack([
            d["node_id"][idx],
            x[idx],
            y[idx],
            z[idx],
            d["epv"][idx],
            d["partition_id"][idx],
            d["repair_level"][idx],
        ])
        np.savetxt(
            os.path.join(args.outdir, label + ".csv"),
            rows,
            delimiter=",",
            header=header,
            comments=""
        )

    with open(os.path.join(args.outdir, "trace_summary.txt"), "w") as f:
        f.write("Bauxitemed Adaptive LG/HPF ParaView Trace\n")
        f.write("===========================================\n")
        f.write(f"Real blocks: {n:,}\n")
        f.write(
            f"Dimensions: "
            f"{int(d['dimensions'][0])} x "
            f"{int(d['dimensions'][1])} x "
            f"{int(d['dimensions'][2])}\n"
        )
        f.write(f"Initial NPV: {float(d['initial_npv'][0]):,.0f}\n")
        f.write(f"Final NPV: {float(d['final_npv'][0]):,.0f}\n")
        f.write(
            f"Local selected: "
            f"{np.count_nonzero(d['local_selected']):,}\n"
        )
        f.write(
            f"Repair additions: "
            f"{np.count_nonzero(d['repair_added']):,}\n"
        )
        f.write(
            f"Maximum repair level: "
            f"{int(d['repair_level'].max())}\n"
        )
        f.write(
            f"Boundary members: "
            f"{np.count_nonzero(d['boundary_member']):,}\n"
        )
        f.write(
            f"Boundary additions: "
            f"{np.count_nonzero(d['boundary_added']):,}\n"
        )
        f.write(
            f"Boundary removals: "
            f"{np.count_nonzero(d['boundary_removed']):,}\n"
        )

    print(f"Created: {output}")
    print(f"Real blocks exported: {n:,}")
    print(
        "Repair additions:",
        int(np.count_nonzero(d["repair_added"]))
    )
    print(
        "Maximum repair level:",
        int(d["repair_level"].max())
    )


if __name__ == "__main__":
    main()
