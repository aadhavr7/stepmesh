# StepMesh

Free, open source STEP to polygon mesh converter that produces clean, quad-dominant topology: grids that follow the surfaces' own flow lines, ngons instead of triangle fans at holes and trims, and watertight output.

![StepMesh](docs/screenshot.png)

- Fillets and cylinders become tidy quad strips, spheres and tori become lat-long grids.
- Cut and trimmed surfaces keep their grid. Cells cut by a hole become ngons rather than slivers.
- Every shared edge is divided once, so the mesh is welded and watertight by construction.
- Normals come from the exact CAD surface.
- Exports OBJ, FBX, PLY, STL and GLB, with real units (mm to m for Blender) and Z-up or Y-up.

## Download (Windows)

Get `StepMesh-windows.zip` from the [Releases](../../releases) page, unzip it anywhere and run `StepMesh.exe`. No install or Python needed. Keep the `_internal` folder next to the exe.

Windows may warn that the app is from an unknown publisher (it is not code-signed). Click **More info**, then **Run anyway**.

## Run from source

Needs Python 3.11 ([python.org](https://www.python.org)).

```bash
git clone https://github.com/<your-account>/stepmesh.git
```

Then in the `stepmesh` folder double-click `setup.bat` once, and `run.bat` to start the app. On macOS or Linux:

```bash
python3.11 -m venv .venv && .venv/bin/pip install -r requirements.txt && .venv/bin/python StepMesh.pyw
```

## Using the app

- **Open** a `.step`, `.stp` or `.brep` with Ctrl+O, by dropping it on the window, or from Open recent. It meshes straight away.
- **Mesh settings:** density preset or angle, divide larger than, avoid smaller than, aspect ratio limit, output (N-gons, Quads & triangles, Triangles), trim handling and sharp edge angle. Click **Update mesh** or tick auto-update.
- **View:** colour by clay, by CAD surface, or by polygon type (quads blue, triangles orange, ngons green). Toggle polygon edges, flat shading, back-face highlight and open-edge highlight. Left drag orbits, middle or right drag pans, the wheel zooms, F frames.
- **Model info** shows polygon counts and a watertight check.
- **Export** with Ctrl+E.

## Command line

```bash
python -m stepmesh part.step -o part.obj
```

Common options: `-a 12` (angle in degrees), `-t ngons|quads|tris`, `--max-edge 5`, `--min-edge 0.5`, `--aspect 2`, `--weld all|smooth|none`. The output format follows the file extension (`.obj .fbx .ply .stl .glb`). Run `python -m stepmesh -h` for everything.

## Building the Windows app

After `setup.bat`, run `build.bat`. It runs the tests, builds `dist\StepMesh\StepMesh.exe` and zips it. Pushing a version tag (`git tag v0.1.0` then `git push --tags`) makes GitHub build the zip and attach it to a release automatically.

## Tests

```bash
python tests/run_tests.py
```

It meshes every part in `tests/parts` in all three output modes and fails on any open edge or flipped polygon. `tools/make_test_parts.py` regenerates the parts (needs `requirements-dev.txt`), and `tools/analyze_obj.py` reports duplicate vertices and open edges in any OBJ.

## How it works

See [docs/ALGORITHM.md](docs/ALGORITHM.md). It is an independent implementation built on OpenCASCADE and published geometry techniques.

## License

MIT, see `LICENSE`. Third-party libraries are listed in `THIRD_PARTY_NOTICES.md`. Note that the Triangle library used for some triangulations is free for non-commercial use only, so builds that include it may not be sold.
