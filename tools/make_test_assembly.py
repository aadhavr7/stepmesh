"""Write tests/parts/assembly.step: a nested test assembly with one part reused four times (needs build123d)."""
import copy
from build123d import Box, Compound, Cylinder, Location, Pos, export_step, fillet, Axis
plate = fillet(Box(80, 50, 6).edges().filter_by(Axis.Z), radius=5); plate.label = "base_plate"
bolt = Cylinder(3, 14); bolt.label = "bolt"
bolts = []
for i, (x, y) in enumerate([(-32, -17), (32, -17), (32, 17), (-32, 17)]):
    b = copy.copy(bolt); b.label = f"bolt_{i + 1}"; b.location = Location((x, y, 7)); bolts.append(b)
bolt_group = Compound(children=bolts, label="bolts")
arm = Box(10, 10, 40); arm.label = "arm"; arm.location = Location((0, 0, 23))
cap = Cylinder(8, 6); cap.label = "cap"; cap.location = Location((0, 0, 46))
tower = Compound(children=[arm, cap], label="tower")
asm = Compound(children=[plate, bolt_group, tower], label="test_assembly")
export_step(asm, "tests/parts/assembly.step")
print("ok")
