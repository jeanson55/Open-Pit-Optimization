"""Render the exact bauxitemed pit on the recorded 9-way spatial grid.

This is a block-index visualization, not surveyed topography. It reconstructs
the same row-major x/y/z indices used by Experiment E and spatial.py.
Run from the project root with Pillow installed:
    python experiments/build_bauxitemed_3d_block_model.py
"""
from __future__ import annotations

import ast
import json
import struct
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
DATA = ROOT / "data" / "BauxiteMed" / "bauxitemed.dat"
PIT = RESULTS / "bauxitemed_baseline_pit.npy"
OUTPUT = RESULTS / "bauxitemed_3d_block_model.png"

NX, NY, NZ = 120, 120, 26
SX, SY, SZ = 4.0, 1.75, 7.5
CX, CY = 610.0, 270.0
PALETTE = [
    "#377eb8", "#4daf4a", "#e6ab02",
    "#e7298a", "#7570b3", "#d95f02",
    "#1b9e77", "#66a61e", "#a6761d",
]


def projection(x: float, y: float, z: float) -> tuple[float, float]:
    return CX + (x - y) * SX, CY + (x + y) * SY - z * SZ


def load_pit() -> list[float]:
    raw = PIT.read_bytes()
    if raw[:6] != b"\x93NUMPY" or raw[6] != 1:
        raise ValueError("Expected a NumPy v1 pit membership file")
    header_len = struct.unpack_from("<H", raw, 8)[0]
    header = ast.literal_eval(raw[10:10 + header_len].decode("latin1").strip())
    if header != {"descr": "<f8", "fortran_order": False, "shape": (NX * NY * NZ,)}:
        raise ValueError(f"Unexpected pit array header: {header}")
    return [value[0] for value in struct.iter_unpack("<d", raw[10 + header_len:])]


def dashed(draw: ImageDraw.ImageDraw, a: tuple[float, float], b: tuple[float, float],
           fill: str, width: int = 3, dash: float = 11, gap: float = 7) -> None:
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    length = (dx * dx + dy * dy) ** 0.5
    if not length:
        return
    ux, uy = dx / length, dy / length
    position = 0.0
    while position < length:
        end = min(position + dash, length)
        draw.line((ax + ux * position, ay + uy * position,
                   ax + ux * end, ay + uy * end), fill=fill, width=width)
        position = end + gap


def get_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        r"C:\Windows\Fonts\arialbd.ttf" if bold else r"C:\Windows\Fonts\arial.ttf",
        r"C:\Windows\Fonts\segoeuib.ttf" if bold else r"C:\Windows\Fonts\segoeui.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            pass
    return ImageFont.load_default()


def main() -> None:
    values = [float(value) for value in DATA.read_text(encoding="ascii").split()]
    pit = load_pit()
    if len(values) != NX * NY * NZ:
        raise ValueError(f"Economic data has {len(values)} values, expected {NX * NY * NZ}")
    selected_ids = [i for i, selected in enumerate(pit) if selected != 0]
    selected_npv = sum(values[i] for i in selected_ids)
    exact = json.loads((RESULTS / "global_solver_benchmark_bauxitemed_normreuse.json").read_text(encoding="utf-8"))
    if len(selected_ids) != exact["hpf_blocks"] or selected_npv != exact["hpf_npv"]:
        raise ValueError("Pit membership or objective does not match the exact global HPF result")

    # Render at 2x and downsample for smoother small block faces and text.
    scale = 2
    image = Image.new("RGB", (1600 * scale, 1050 * scale), "#fbfcfe")
    draw = ImageDraw.Draw(image)
    def p(x: float, y: float, z: float) -> tuple[int, int]:
        px, py = projection(x, y, z)
        return round(px * scale), round(py * scale)

    # Draw the feasible exact-pit blocks back-to-front. Each selected block is
    # a small top-face diamond; its fill identifies the actual spatial region.
    ordered = sorted(selected_ids, key=lambda i: ((i // (NX * NY)) + (i // NX) % NY + i % NX))
    for i in ordered:
        z, rem = divmod(i, NX * NY)
        y, x = divmod(rem, NX)
        part = (y // 40) * 3 + (x // 40)
        px, py = p(x, y, z)
        dx, dy = round(SX * scale), round(SY * scale)
        pts = [(px + dx, py), (px, py - dy * 2), (px - dx, py), (px, py + dy * 2)]
        draw.polygon(pts, fill=PALETTE[part])

    # The four internal grid cuts are vertical partition planes extending
    # through the 26 benches. Draw their perimeter over the selected shell.
    boundary_color = "#172554"
    for axis in ("x", "y"):
        for cut in (40, 80):
            corners = (
                [(cut, 0, 0), (cut, NY, 0), (cut, NY, NZ - 1), (cut, 0, NZ - 1)]
                if axis == "x" else
                [(0, cut, 0), (NX, cut, 0), (NX, cut, NZ - 1), (0, cut, NZ - 1)]
            )
            projected = [p(*corner) for corner in corners]
            projected.append(projected[0])
            for a, b in zip(projected, projected[1:]):
                dashed(draw, a, b, boundary_color, width=3 * scale)

    # Coordinate frame and labels.
    origin = p(0, 0, 0)
    for end, label, offset in [
        (p(NX - 1, 0, 0), "x", (20, 8)),
        (p(0, NY - 1, 0), "y", (-34, 8)),
        (p(0, 0, NZ - 1), "bench index (z)", (82, 34)),
    ]:
        draw.line((origin, end), fill="#334155", width=3 * scale)
        draw.text((end[0] + offset[0] * scale, end[1] + offset[1] * scale), label,
                  fill="#0f172a", font=get_font(17 * scale, True))

    # Add title, subtitle, and a nine-part legend outside the projected model.
    draw.text((65 * scale, 34 * scale), "Bauxitemed: exact pit within the spatial partition grid",
              fill="#111827", font=get_font(27 * scale, True))
    draw.text((67 * scale, 77 * scale),
              "Selected blocks colored by 9-way partition · dashed lines mark the actual grid cuts",
              fill="#475569", font=get_font(15 * scale))
    draw.text((1120 * scale, 190 * scale), "Spatial regions", fill="#111827", font=get_font(17 * scale, True))
    for part, color in enumerate(PALETTE):
        col, row = part % 3, part // 3
        x0, y0 = (1120 + col * 126) * scale, (225 + row * 43) * scale
        draw.rectangle((x0, y0, x0 + 24 * scale, y0 + 24 * scale), fill=color)
        draw.text((x0 + 32 * scale, y0 + 2 * scale), f"P{part}", fill="#1f2937", font=get_font(14 * scale))
    draw.text((1120 * scale, 382 * scale), f"Exact pit: {len(selected_ids):,} blocks",
              fill="#1f2937", font=get_font(15 * scale, True))
    draw.text((1120 * scale, 412 * scale), f"NPV: {selected_npv:,.0f}",
              fill="#1f2937", font=get_font(15 * scale, True))
    draw.text((65 * scale, 986 * scale),
              "Block-index geometry reconstructed from the 120×120×26 input order. Vertical axis is bench index, not surveyed elevation.",
              fill="#475569", font=get_font(14 * scale))
    draw.text((65 * scale, 1012 * scale),
              "Membership and NPV verified against the full-model global HPF benchmark; this view is not a topographic surface.",
              fill="#475569", font=get_font(13 * scale))

    image = image.resize((1600, 1050), Image.Resampling.LANCZOS)
    image.save(OUTPUT, format="PNG", optimize=True)
    print(f"Wrote {OUTPUT}")
    print(f"Verified exact pit membership: {len(selected_ids):,} blocks; NPV={selected_npv:,.0f}")


if __name__ == "__main__":
    main()
