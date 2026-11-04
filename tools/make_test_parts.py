"""Generate simple STEP test parts with build123d.

Usage: python tools/make_test_parts.py [out_dir]
"""
import sys
from pathlib import Path

from build123d import (
    Align, Axis, Box, Cone, Cylinder, Location, Pos, Rot, Sphere, Torus,
    export_step, fillet,
)


def filleted_box():
    b = Box(40, 30, 20)
    return fillet(b.edges(), radius=4)


def cylinder():
    return Cylinder(10, 40)


def sphere():
    return Sphere(15)


def cone():
    return Cone(12, 0, 25)


def torus():
    return Torus(20, 5)


def plate_with_holes():
    p = Box(60, 40, 5)
    for x in (-20, 20):
        for y in (-12, 12):
            p -= Pos(x, y, 0) * Cylinder(4, 10)
    p -= Cylinder(8, 10)
    return p


def boolean_cut_block():
    b = Box(40, 40, 30)
    # cross hole through the side: trimmed cylindrical faces
    b -= Rot(0, 90, 0) * Cylinder(6, 60)
    # spherical pocket on top
    b -= Pos(8, 8, 15) * Sphere(8)
    # cylinder boss cut into a corner: trimmed planar faces
    b -= Pos(20, -20, 0) * Cylinder(10, 40)
    return b


def cylinder_cross_hole():
    c = Cylinder(12, 50)
    c -= Rot(90, 0, 0) * Cylinder(4, 40)
    return c


def filleted_plate_with_holes():
    p = Box(60, 40, 8)
    p = fillet(p.edges().filter_by(Axis.Z), radius=6)
    for x in (-18, 18):
        p -= Pos(x, 0, 0) * Cylinder(5, 20)
    p = fillet(p.edges().group_by(Axis.Z)[-1], radius=1.5)
    return p


PARTS = {
    "filleted_box": filleted_box,
    "cylinder": cylinder,
    "sphere": sphere,
    "cone": cone,
    "torus": torus,
    "plate_with_holes": plate_with_holes,
    "boolean_cut_block": boolean_cut_block,
    "cylinder_cross_hole": cylinder_cross_hole,
    "filleted_plate_with_holes": filleted_plate_with_holes,
}


def main():
    out = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent / "tests" / "parts")
    out.mkdir(parents=True, exist_ok=True)
    for name, fn in PARTS.items():
        try:
            shape = fn()
            path = out / f"{name}.step"
            export_step(shape, str(path))
            print(f"wrote {path}  faces={len(shape.faces())} edges={len(shape.edges())}")
        except Exception as exc:  # keep going so one bad part does not block the rest
            print(f"FAILED {name}: {exc}")


if __name__ == "__main__":
    main()
