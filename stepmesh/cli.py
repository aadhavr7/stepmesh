"""Command line: stepmesh input.step -o output.obj [options]"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from .export_obj import triangulate_polys, write_obj
from .meshdata import stats
from .mesher import mesh_shape
from .occ import read_step
from .params import MeshParams


def build_parser():
    d = MeshParams()
    ap = argparse.ArgumentParser(prog="stepmesh", description="STEP to quad-dominant polygon mesh (OBJ).")
    ap.add_argument("input", help="STEP file")
    ap.add_argument("-o", "--output", help="output file: .obj .fbx .ply .stl .glb (default: input name with .obj)")
    ap.add_argument("-a", "--angle", type=float, default=d.angle_deg, help="max angle between surface normals (and curve tangents) across one polygon, degrees (default %(default)s)")
    ap.add_argument("--max-edge", type=float, default=d.max_edge, help="divide edges longer than this, model units, 0 = off")
    ap.add_argument("--max-edge-scope", choices=["curved", "planar", "all"], default=d.max_edge_scope, help="surfaces --max-edge applies to (default %(default)s)")
    ap.add_argument("--min-edge", type=float, default=d.min_edge, help="avoid edges shorter than this, model units, 0 = off")
    ap.add_argument("--aspect", type=float, default=d.aspect, help="max quad aspect ratio on all faces incl. planar (planar faces then get a grid), 0 = off")
    ap.add_argument("--trim-aspect", type=float, default=d.trim_aspect, help="aspect limit for trimmed curved faces only, 0 = off")
    ap.add_argument("-t", "--type", choices=["quads", "ngons", "tris"], default=d.output, help="output polygons: ngons (default), quads (quads+tris), tris")
    ap.add_argument("--band", choices=["cells", "cdt"], default=d.band, help="trims on curved faces: cells (clipped grid cells, ngons) or cdt (triangle band)")
    ap.add_argument("--band-margin", type=float, default=d.band_margin, help="min distance of grid nodes from trims, in cell sizes")
    ap.add_argument("--sharp", type=float, default=d.sharp_deg, help="normals split where B-rep faces meet sharper than this, degrees")
    ap.add_argument("--weld", choices=["all", "smooth", "none"], default="all",
                    help="all: one closed mesh; smooth: split vertices only at sharp edges; none: every B-rep face its own island (exact copies along edges)")
    ap.add_argument("--no-weld", action="store_true", help="same as --weld none")
    ap.add_argument("--no-heal", action="store_true", help="skip ShapeFix on import")
    ap.add_argument("--group", choices=["face", "body"], default="face", help="OBJ groups per B-rep face or per body")
    ap.add_argument("--stats", help="write stats JSON here")
    ap.add_argument("-q", "--quiet", action="store_true")
    return ap


def params_from_args(a) -> MeshParams:
    return MeshParams(angle_deg=a.angle, max_edge=a.max_edge, max_edge_scope=a.max_edge_scope, min_edge=a.min_edge, aspect=a.aspect,
                      trim_aspect=a.trim_aspect, output=a.type, band=a.band, band_margin=a.band_margin,
                      sharp_deg=a.sharp, heal=not a.no_heal, weld="none" if a.no_weld else a.weld)


def convert(inp, out, prm: MeshParams, group="face"):
    t0 = time.perf_counter()
    shape = read_step(inp, heal=prm.heal)
    mesh, mesher = mesh_shape(shape, prm)
    if prm.output == "tris":
        triangulate_polys(mesh)
    st = stats(mesh)
    st["seconds"] = round(time.perf_counter() - t0, 3)
    kinds = {}
    for fc in mesher.faces:
        if fc is not None:
            kinds[fc.kind] = kinds.get(fc.kind, 0) + 1
    st["face_kinds"] = kinds
    st["warnings"] = mesher.log
    header = f"source: {Path(inp).name}\nsettings: {prm}"
    if Path(out).suffix.lower() == ".obj":
        write_obj(mesh, out, sharp_deg=prm.sharp_deg, group_by=group, header=header, weld=prm.weld)
    else:
        from .exporters import FORMATS, export
        from .meshbuild import build_arrays
        if Path(out).suffix.lower() not in FORMATS:
            raise SystemExit(f"unknown output format {Path(out).suffix}; use one of {', '.join(FORMATS)}")
        export(out, mesh, build_arrays(mesh, prm.sharp_deg), weld=prm.weld, sharp_deg=prm.sharp_deg)
    return st


def main(argv=None):
    a = build_parser().parse_args(argv)
    inp = Path(a.input)
    out = Path(a.output) if a.output else inp.with_suffix(".obj")
    st = convert(inp, out, params_from_args(a), a.group)
    if a.stats:
        Path(a.stats).write_text(json.dumps(st, indent=2))
    if not a.quiet:
        print(f"{inp.name} -> {out}")
        for k, v in st.items():
            if k != "warnings":
                print(f"  {k}: {v}")
        for w in st["warnings"]:
            print(f"  warning: {w}")
    return 0 if st["watertight"] else 2


if __name__ == "__main__":
    sys.exit(main())
