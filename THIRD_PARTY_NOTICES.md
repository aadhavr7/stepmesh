# Third-party software

StepMesh's own code is MIT licensed (see `LICENSE`). It uses, and the packaged Windows app ships, these open source libraries, each under its own licence.

| Library | Used for | Licence | Source |
|---|---|---|---|
| Open CASCADE Technology (OCCT) | STEP reading, B-rep geometry | LGPL 2.1 with OCCT exception | https://github.com/Open-Cascade-SAS/OCCT |
| OCP (CadQuery bindings for OCCT) | Python access to OCCT | Apache 2.0 | https://github.com/CadQuery/OCP |
| Triangle (J. R. Shewchuk) and its Python wrapper | Constrained Delaunay triangulation | Triangle: free for non-commercial use (see below). Wrapper: LGPL 3.0 | https://www.cs.cmu.edu/~quake/triangle.html, https://github.com/drufat/triangle |
| Qt / PySide6 | Desktop app and OpenGL view | LGPL 3.0 | https://code.qt.io, https://www.qt.io/download-open-source |
| Shapely and GEOS | 2D polygon clipping | BSD 3-Clause (Shapely), LGPL 2.1 (GEOS) | https://github.com/shapely/shapely, https://github.com/libgeos/geos |
| mapbox_earcut (earcut.hpp) | Polygon triangulation for display | ISC | https://github.com/skogler/mapbox_earcut_python |
| NumPy | Arrays and maths | BSD 3-Clause | https://github.com/numpy/numpy |
| PyOpenGL | OpenGL calls | BSD-style | https://github.com/mcfletch/pyopengl |
| Python | Runtime | PSF License | https://www.python.org |
| PyInstaller bootloader | Packaging | GPL 2.0 with bootloader exception (no effect on the app's licence) | https://github.com/pyinstaller/pyinstaller |

## Important: Triangle is non-commercial

Triangle's author allows free use and redistribution, but it may not be sold or included in a commercial product without his permission. StepMesh itself is MIT, but **the packaged app and any build that includes Triangle may only be used and shared non-commercially.** For commercial use, obtain permission from Triangle's author or replace Triangle with a commercially licensed triangulator.

## LGPL libraries

OCCT, Qt and GEOS ship as separate, unmodified DLLs in the `_internal` folder of the app. You may replace them with your own compatible builds, and their full source code is available from the links above.
