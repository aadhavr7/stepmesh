"""Mesh file writers: OBJ (ngons, groups, split normals), PLY (ngons),
STL (triangles), GLB / glTF binary (triangles, split normals).

All writers take the same transform options:
  scale   multiply coordinates (0.001 turns millimetres into metres)
  up      "z" keeps CAD axes; "y" converts Z-up to Y-up (x, z, -y)
"""
from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np

from .export_obj import write_obj
from .meshbuild import MeshArrays
from .meshdata import Mesh

FORMATS = {
    ".obj": "Wavefront OBJ (quads + ngons, groups, normals)",
    ".ply": "Stanford PLY (quads + ngons)",
    ".stl": "STL binary (triangles)",
    ".glb": "glTF binary (triangles, normals)",
    ".fbx": "FBX binary (quads + ngons, normals, one object per body)",
}


def _xform(P, scale, up):
    P = np.asarray(P, dtype=np.float64) * scale
    if up == "y":
        P = np.stack([P[:, 0], P[:, 2], -P[:, 1]], axis=1)
    return P


def export(path, mesh: Mesh, arrays: MeshArrays, scale=1.0, up="z", weld="all", group="face", sharp_deg=20.0, header=""):
    path = Path(path)
    ext = path.suffix.lower()
    if ext == ".obj":
        if scale != 1.0 or up != "z":
            mesh = _transformed_copy(mesh, scale, up)
        return write_obj(mesh, path, sharp_deg=sharp_deg, group_by=group, header=header, weld=weld)
    if ext == ".ply":
        return write_ply(path, arrays, scale, up)
    if ext == ".stl":
        return write_stl(path, arrays, scale, up)
    if ext == ".glb":
        return write_glb(path, arrays, scale, up)
    if ext == ".fbx":
        from .fbx import write_fbx
        return write_fbx(path, mesh, scale=scale, up=up, weld=weld, sharp_deg=sharp_deg)
    raise ValueError(f"unsupported format {ext}")


def _transformed_copy(mesh, scale, up):
    from copy import copy
    m2 = copy(mesh)
    P = _xform(np.array(mesh.points), scale, up)
    m2.points = list(P)
    if up == "y":
        from .meshdata import Polygon
        m2.polys = [Polygon(p.verts, [np.array([n[0], n[2], -n[1]]) for n in p.normals], p.face, p.body)
                    for p in mesh.polys]
    return m2


def write_ply(path, A: MeshArrays, scale=1.0, up="z"):
    P = _xform(A.points, scale, up).astype("<f4")
    with open(path, "wb") as f:
        f.write(b"ply\nformat binary_little_endian 1.0\ncomment stepmesh\n")
        f.write(f"element vertex {len(P)}\nproperty float x\nproperty float y\nproperty float z\n".encode())
        f.write(f"element face {len(A.poly_v)}\nproperty list uchar int vertex_indices\nend_header\n".encode())
        f.write(P.tobytes())
        buf = bytearray()
        for vv in A.poly_v:
            if len(vv) > 255:
                raise ValueError("polygon with more than 255 corners: export PLY in triangle mode")
            buf += struct.pack("<B", len(vv)) + np.asarray(vv, dtype="<i4").tobytes()
        f.write(buf)
    return path


def write_stl(path, A: MeshArrays, scale=1.0, up="z"):
    P = _xform(A.points, scale, up)
    T = P[A.tri_v]
    n = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    n = np.where(ln > 0, n / np.maximum(ln, 1e-300), 0.0)
    rec = np.zeros(len(T), dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]))
    rec["n"] = n
    rec["v"] = T
    with open(path, "wb") as f:
        f.write(b"stepmesh binary STL".ljust(80, b" "))
        f.write(struct.pack("<I", len(T)))
        f.write(rec.tobytes())
    return path


def write_glb(path, A: MeshArrays, scale=1.0, up="z"):
    # glTF is Y-up by definition: always convert unless the caller already did
    P = _xform(A.points, scale, "y" if up == "z" else "z")
    N = A.normals.copy()
    if up == "z":
        N = np.stack([N[:, 0], N[:, 2], -N[:, 1]], axis=1)
    # one GPU vertex per (position, normal) pair
    pairs = np.stack([A.tri_v.ravel(), A.tri_n.ravel()], axis=1)
    uniq, inv = np.unique(pairs, axis=0, return_inverse=True)
    pos = P[uniq[:, 0]].astype("<f4")
    nrm = N[uniq[:, 1]].astype("<f4")
    idx = inv.reshape(-1).astype("<u4")
    blobs = [pos.tobytes(), nrm.tobytes(), idx.tobytes()]
    offsets, off = [], 0
    for b in blobs:
        offsets.append(off)
        off += (len(b) + 3) & ~3
    binbuf = b"".join(b + b"\0" * (((len(b) + 3) & ~3) - len(b)) for b in blobs)
    gltf = {
        "asset": {"version": "2.0", "generator": "stepmesh"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0, "name": Path(path).stem}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0, "NORMAL": 1}, "indices": 2}]}],
        "buffers": [{"byteLength": len(binbuf)}],
        "bufferViews": [
            {"buffer": 0, "byteOffset": offsets[0], "byteLength": len(blobs[0]), "target": 34962},
            {"buffer": 0, "byteOffset": offsets[1], "byteLength": len(blobs[1]), "target": 34962},
            {"buffer": 0, "byteOffset": offsets[2], "byteLength": len(blobs[2]), "target": 34963},
        ],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": len(pos), "type": "VEC3",
             "min": pos.min(axis=0).tolist(), "max": pos.max(axis=0).tolist()},
            {"bufferView": 1, "componentType": 5126, "count": len(nrm), "type": "VEC3"},
            {"bufferView": 2, "componentType": 5125, "count": len(idx), "type": "SCALAR"},
        ],
    }
    js = json.dumps(gltf, separators=(",", ":")).encode()
    js += b" " * ((4 - len(js) % 4) % 4)
    with open(path, "wb") as f:
        total = 12 + 8 + len(js) + 8 + len(binbuf)
        f.write(struct.pack("<III", 0x46546C67, 2, total))
        f.write(struct.pack("<II", len(js), 0x4E4F534A) + js)
        f.write(struct.pack("<II", len(binbuf), 0x004E4942) + binbuf)
    return path
