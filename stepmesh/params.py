"""Mesher settings. Every value is exposed on the CLI."""
from dataclasses import dataclass
import math


@dataclass
class MeshParams:
    # Max turning angle (degrees) allowed per segment, along edges and along
    # each isoparametric direction of a surface.
    angle_deg: float = 12.0
    # "Divide larger than": max edge length in model units (0 = off).
    max_edge: float = 0.0
    # Where max_edge applies: "curved" (default), "planar" or "all".
    max_edge_scope: str = "curved"
    # "Avoid smaller than": edges shorter than this are not subdivided further (0 = off).
    min_edge: float = 0.0
    # Max cell aspect ratio, all surfaces including planar ones (0 = off).
    # Long strips on cylinders and fillets are allowed when this is off.
    # When set (or when max_edge reaches planar faces) planar faces get a
    # grid instead of an ngon / minimal triangulation.
    aspect: float = 0.0
    # Vertex sharing in the written file: "all" (closed mesh), "smooth"
    # (split only where normals split at sharp edges) or "none" (each B-rep
    # face its own island, exact copies along edges). The mesh is always
    # built welded; this only changes how it is written.
    weld: str = "all"
    # Aspect limit applied only to trimmed curved faces so the grid around
    # holes stays usable. 0 disables.
    trim_aspect: float = 3.0
    # Output polygon type: "ngons" (default), "quads"
    # (quads + tris) or "tris".
    output: str = "ngons"
    # Trim handling on curved faces: "cells" clips grid cells against the
    # trims (ngons, no triangle band), "cdt" drops cells near trims and
    # fills a constrained Delaunay triangle band.
    band: str = "cells"
    # Keep grid cells whose free corners are at least this fraction of the
    # cell size away from a trim curve. Smaller = thinner triangle band.
    band_margin: float = 0.35
    # Pair triangles of planar faces into quads in "quads" mode.
    planar_quads: bool = True
    # Normals: smooth across B-rep edges whose dihedral angle is below this (degrees).
    sharp_deg: float = 20.0
    # Run ShapeFix on import.
    heal: bool = True

    @property
    def angle(self) -> float:
        return math.radians(max(self.angle_deg, 0.5))
