"""Mesh every STEP in tests/parts in all output modes; fail on any open edge or flipped polygon.

Usage: python tests/run_tests.py
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from stepmesh.export_obj import triangulate_polys  # noqa: E402
from stepmesh.meshdata import stats  # noqa: E402
from stepmesh.mesher import mesh_shape  # noqa: E402
from stepmesh.occ import read_step  # noqa: E402
from stepmesh.params import MeshParams  # noqa: E402


def main():
    parts = sorted((ROOT / "tests" / "parts").glob("*.step"))
    if not parts:
        print("no test parts found; run: python tools/make_test_parts.py")
        return 1
    failures = 0
    for step in parts:
        shape = read_step(step)
        for mode in ("ngons", "quads", "tris"):
            t = time.perf_counter()
            mesh, _ = mesh_shape(shape, MeshParams(output=mode))
            if mode == "tris":
                triangulate_polys(mesh)
            st = stats(mesh)
            ok = st["watertight"] and st["orientation_errors"] == 0
            failures += not ok
            print(f"{'ok  ' if ok else 'FAIL'} {step.stem:28s} {mode:5s} polys {st['polys']:6d}  quads {st['quads']:6d}  "
                  f"tris {st['tris']:6d}  ngons {st['ngons']:5d}  open {st['open_edges']}  flipped {st['orientation_errors']}  "
                  f"{time.perf_counter() - t:.2f}s")
    print(f"\n{len(parts) * 3 - failures} passed, {failures} failed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
