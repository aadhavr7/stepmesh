# PyInstaller spec for the StepMesh desktop app (one-folder build).
# Build:  .venv/Scripts/pyinstaller StepMesh.spec --noconfirm
# Output: dist/StepMesh/StepMesh.exe
import os
import sysconfig

from PyInstaller.utils.hooks import collect_dynamic_libs, collect_submodules

site = sysconfig.get_paths()["purelib"]


def tree(src, dest, skip_ext=(".pyi", ".pyc")):
    out = []
    for root, _, files in os.walk(src):
        for f in files:
            if f.endswith(skip_ext):
                continue
            rel = os.path.relpath(root, src)
            out.append((os.path.join(root, f), os.path.normpath(os.path.join(dest, rel))))
    return out


datas = [
    ("stepmesh/gui/stepmesh.ico", "."),
    ("stepmesh/gui/stepmesh.png", "."),
]
binaries = []
# OCP: the extension module plus its delvewheel DLL folder, which OCP/__init__.py
# adds to the DLL search path relative to itself
datas += tree(os.path.join(site, "OCP"), "OCP")
binaries += [(os.path.join(site, "cadquery_ocp_novtk.libs", f), "cadquery_ocp_novtk.libs")
             for f in os.listdir(os.path.join(site, "cadquery_ocp_novtk.libs"))]
for libs in ("mapbox_earcut.libs", "shapely.libs"):
    p = os.path.join(site, libs)
    if os.path.isdir(p):
        binaries += [(os.path.join(p, f), libs) for f in os.listdir(p)]
binaries += collect_dynamic_libs("triangle")

hiddenimports = (["OCP"] + collect_submodules("stepmesh") + collect_submodules("shapely")
                 + ["triangle", "mapbox_earcut", "OpenGL.platform.win32", "OpenGL.arrays.numpymodule",
                    "OpenGL.arrays.ctypesarrays", "OpenGL.arrays.numbers", "OpenGL.arrays.strings"])

a = Analysis(
    ["StepMesh.pyw"],
    pathex=["."],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    excludes=["matplotlib", "scipy", "gmsh", "build123d", "tkinter", "IPython", "jupyter", "pandas",
              "PySide6.QtWebEngineCore", "PySide6.QtWebEngineWidgets", "PySide6.Qt3DCore", "PySide6.QtQuick",
              "PySide6.QtQml", "PySide6.QtMultimedia", "PySide6.QtPdf", "PySide6.QtCharts", "PySide6.QtDataVisualization"],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="StepMesh",
    icon="stepmesh/gui/stepmesh.ico",
    console=False,
    upx=False,
)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name="StepMesh")
