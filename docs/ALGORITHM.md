# How StepMesh works

## Sources

An independent implementation based on public literature and public documentation:

- Piegl and Tiller, The NURBS Book: surface evaluation, isoparametric curves, chordal and angular error.
- Gordon and Hall, transfinite interpolation (Coons patches): structured grids from four boundary curves.
- Shewchuk, Triangle: constrained Delaunay triangulation.
- General NURBS tessellation literature: UV grids for trimmed surfaces.
- Public user documentation of commercial NURBS modellers' mesh exporters, used only to match what the settings mean (angle between normals, divide larger than, avoid smaller than, aspect ratio limit, welding). No third-party binaries were inspected.

## Pipeline

1. **Read and heal.** STEP through OpenCASCADE `STEPControl_Reader`, then `ShapeFix_Shape`.
2. **Topology index.** Faces, edges and vertices get stable indices. Each B-rep vertex gets one mesh vertex.
3. **Per-face context.** Each face is taken in FORWARD orientation. Its wires are walked in order, and each edge use keeps its pcurve (seam edges get the right one from their orientation). UV scale factors make UV roughly metric for geometric tests.
4. **Sizing per direction.** A 25 x 25 sample grid of the surface measures, along u and along v separately, the turning of the surface normal and the length. Segment weight is `max(turn / angle, length / L_ref)`. The division count is the largest weight sum over rows, and points go at equal cumulative weight, which is a curvature-aware arc-length reparameterization. A cylinder gets many divisions around and one along. Poles and apexes are masked out.
5. **Classification.**
   - `planar`: one ngon, or a few ngons when the face has holes (constrained triangulation regrown into simple ngons). Quads and triangles in the other output modes. Planar faces get a grid instead when an aspect limit or a planar max edge length is set.
   - `quad`: one wire with four corners (turns over 40 degrees in scaled UV). Cylinders, cones, spheres and tori land here through their seam and pole edges.
   - `tri`: three corners. The sharpest corner becomes a collapsed fourth side, giving a lat-long style patch with a small triangle fan.
   - `trimmed`: everything else.
6. **Edge counts.** Each edge takes the largest of its own count (tangent turning and length), the requirement of each curved face where it is an isoparm, and the counts of edges opposite it on structured faces (propagated with union-find), so fillet chains stay consistent.
7. **Edge discretization, once.** Points are shared by every face that uses the edge, so the mesh is welded by construction. Degenerate edges map to their single vertex.
8. **Face meshing in UV.**
   - Structured faces: transfinite interpolation from the four sides. If opposite sides disagree or a cell inverts, the face uses the trimmed path.
   - Trimmed faces (default "clipped grid cells"): grid lines come from the sizing, aligned to the widest isoparametric boundary edge, with a line added through any hole no line crosses. Before any vertex exists, every crossing of a grid line with a trim becomes a required point of that edge. Each grid cell is then clipped against the face in scaled UV, and its corners map back to grid nodes or shared edge points, never to new points on a trim. Boundary points between grid lines become extra ngon corners, and slivers merge into a neighbour.
   - Optional "triangle band": cells near trims are dropped and the band between the grid and the trims is filled with a constrained Delaunay triangulation.
9. **Safety net.** A polygon side left without a partner, whose ends lie on its face boundary, gets the missing shared edge points inserted. Small closed gaps are filled. Free B-rep edges are never capped.
10. **Normals.** Every corner normal is evaluated on the exact surface at that corner's UV. At export, normals at a vertex merge when within the sharp edge angle.
11. **Export.** OBJ (groups per CAD surface), FBX 7.4 binary (one object per body, units in the file), PLY, STL and GLB.

## Vertex welding

Because faces only depend on the shared edge points, welding is an export choice:

- `all`: one vertex per position, a closed mesh.
- `smooth`: vertices split only where normals split (sharp edges).
- `none`: each CAD face is its own island, with exact copies along edges. Merge by Distance with a zero threshold restores the closed mesh.

## Known limits

- Large models are slow (pure Python): about 2 minutes for 2400 faces.
- Clip points next to structured faces, whose edges cannot take extra points, are snapped to the nearest edge point. That can bend a grid line by up to about one cell, and it is logged as a warning.
- Real-world STEP robustness beyond the tested parts is still being proven.
