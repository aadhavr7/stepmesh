"""STEP assembly structure (product tree) via OpenCASCADE XCAF.

read_assembly() returns the tree of assemblies and part instances with
names and local placements, plus the list of unique parts. Each unique part
is meshed once; repeated instances (eight identical screws) reuse it.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
from OCP.BRep import BRep_Builder
from OCP.IFSelect import IFSelect_RetDone
from OCP.ShapeFix import ShapeFix_Shape
from OCP.STEPCAFControl import STEPCAFControl_Reader
from OCP.TCollection import TCollection_ExtendedString
from OCP.TDataStd import TDataStd_Name
from OCP.TDF import TDF_Label

try:
    from OCP.TDF import TDF_LabelSequence
except ImportError:  # OCP 8 exposes NCollection templates under OCP.collections
    from OCP.collections import Sequence_TDF_Label as TDF_LabelSequence
from OCP.TDocStd import TDocStd_Document
from OCP.TopAbs import TopAbs_FACE
from OCP.TopExp import TopExp_Explorer
from OCP.TopoDS import TopoDS_Compound
from OCP.XCAFApp import XCAFApp_Application
from OCP.XCAFDoc import XCAFDoc_DocumentTool, XCAFDoc_ShapeTool


@dataclass
class Node:
    name: str
    matrix: np.ndarray                       # 4x4 local placement (relative to the parent), model units
    part: int | None = None                  # index into parts for a part instance, None for an assembly
    children: list = field(default_factory=list)


@dataclass
class Assembly:
    roots: list                              # top-level Nodes
    parts: list                              # unique part shapes (healed, in their own frame)
    part_names: list
    compound: object                         # all unique parts in one compound, for meshing


def _name(label):
    attr = TDataStd_Name()
    if label.FindAttribute(TDataStd_Name.GetID_s(), attr):
        try:
            return str(attr.Get().ToExtString()).strip()
        except Exception:
            pass
    return ""


_AUTO_NAME = re.compile(r"^(NAUO|SOLID|COMPOUND|PART)\d*$", re.I)


def _matrix(loc):
    t = loc.Transformation()
    m = np.eye(4)
    for r in range(3):
        for c in range(4):
            m[r, c] = t.Value(r + 1, c + 1)
    return m


def read_assembly(path, heal=True) -> Assembly:
    app = XCAFApp_Application.GetApplication_s()
    doc = TDocStd_Document(TCollection_ExtendedString("XmlXCAF"))
    app.InitDocument(doc)
    reader = STEPCAFControl_Reader()
    reader.SetNameMode(True)
    if reader.ReadFile(str(path)) != IFSelect_RetDone:
        raise RuntimeError(f"STEP read failed for {path}")
    reader.Transfer(doc)
    tool = XCAFDoc_DocumentTool.ShapeTool_s(doc.Main())

    parts, part_names, part_of_label = [], [], {}

    def nm(label):
        return _name(label)

    def part_index(label):
        entry = (label.Tag(), label.Depth(), _name(label))
        if entry not in part_of_label:
            shape = XCAFDoc_ShapeTool.GetShape_s(label)
            if heal:
                fixer = ShapeFix_Shape(shape)
                fixer.Perform()
                shape = fixer.Shape()
            part_of_label[entry] = len(parts)
            parts.append(shape)
            part_names.append(nm(label) or f"part_{len(parts)}")
        return part_of_label[entry]

    def build(label, name, matrix, depth=0):
        if depth > 64:
            raise RuntimeError("assembly nesting too deep")
        if XCAFDoc_ShapeTool.IsAssembly_s(label):
            node = Node(name=name or nm(label) or "assembly", matrix=matrix)
            comps = TDF_LabelSequence()
            XCAFDoc_ShapeTool.GetComponents_s(label, comps, False)
            for i in range(1, comps.Length() + 1):
                comp = comps.Value(i)
                ref = TDF_Label()
                if XCAFDoc_ShapeTool.IsReference_s(comp) and XCAFDoc_ShapeTool.GetReferredShape_s(comp, ref):
                    target = ref
                else:
                    target = comp
                child_name = nm(comp)
                if not child_name or _AUTO_NAME.match(child_name):   # SolidWorks "NAUO12" and the like
                    child_name = nm(target)
                node.children.append(build(target, child_name, _matrix(XCAFDoc_ShapeTool.GetLocation_s(comp)), depth + 1))
            return node
        return Node(name=name or nm(label) or "part", matrix=matrix, part=part_index(label))

    free = TDF_LabelSequence()
    tool.GetFreeShapes(free)
    roots = [build(free.Value(i), "", np.eye(4)) for i in range(1, free.Length() + 1)]

    builder = BRep_Builder()
    comp = TopoDS_Compound()
    builder.MakeCompound(comp)
    for s in parts:
        builder.Add(comp, s)
    return Assembly(roots=roots, parts=parts, part_names=part_names, compound=comp)


def face_parts(mesher, asm: Assembly):
    """B-rep face index (in mesher.topo) -> part index."""
    out = {}
    for pi, shape in enumerate(asm.parts):
        exp = TopExp_Explorer(shape, TopAbs_FACE)
        while exp.More():
            fi = mesher.topo.face_index(exp.Current())
            if fi >= 0:
                out[fi] = pi
            exp.Next()
    return out


def print_tree(asm: Assembly):
    def walk(n, d):
        tag = f"part #{n.part}" if n.part is not None else "assembly"
        t = n.matrix[:3, 3]
        print(f"{'  ' * d}{n.name}  [{tag}]  at ({t[0]:.1f}, {t[1]:.1f}, {t[2]:.1f})")
        for c in n.children:
            walk(c, d + 1)
    for r in asm.roots:
        walk(r, 0)
    print(f"{len(asm.parts)} unique parts: {asm.part_names}")
