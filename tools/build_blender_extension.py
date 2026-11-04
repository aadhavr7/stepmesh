"""Build the StepMesh Blender extension zips (Blender 4.2 and newer, Windows x64).

Downloads the dependency wheels for Blender's Python versions (3.11 and
3.13), builds the stepmesh wheel from this repo, writes the manifest wheel
list and writes one zip per Python version into dist/.

Usage: python tools/build_blender_extension.py
Needs: pip (to download wheels) and setuptools; run with a normal Python.
"""
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXT = ROOT / "blender" / "stepmesh_importer"
WHEELS = EXT / "wheels"
# Blender 4.2 to 5.0 run Python 3.11 (numpy 1.24 to 1.26), Blender 5.2 runs 3.13 (numpy 2.3).
DEPS = {
    "3.11": ["cadquery-ocp-novtk==8.0.1.1.0", "shapely==2.0.7", "triangle==20250106", "mapbox-earcut==2.1.0"],
    "3.13": ["cadquery-ocp-novtk==8.0.1.1.0", "shapely==2.2.0", "triangle==20250106", "mapbox-earcut==2.1.0"],
}


def run(cmd):
    print(">", " ".join(cmd))
    subprocess.run(cmd, check=True)


# one zip per Blender Python: Blender 4.2's installer does not pick wheels by Python version
BUILDS = [
    ("3.11", "blender4.2-5.0", 'blender_version_min = "4.2.0"\nblender_version_max = "5.1.0"'),
    ("3.13", "blender5.1-plus", 'blender_version_min = "5.1.0"'),
]


def main():
    version = re.search(r'^version = "([^"]+)"', (ROOT / "pyproject.toml").read_text(), re.M).group(1)
    stage = ROOT / "build" / "blender_wheels"
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    run([sys.executable, "-m", "pip", "wheel", "--quiet", "--no-deps", "-w", str(stage), str(ROOT)])
    own = sorted(p.name for p in stage.glob("stepmesh-*.whl"))
    for py, label, versions in BUILDS:
        if WHEELS.exists():
            shutil.rmtree(WHEELS)
        WHEELS.mkdir(parents=True)
        run([sys.executable, "-m", "pip", "download", "--quiet", "--only-binary=:all:", "--no-deps",
             "--python-version", py, "--platform", "win_amd64", "-d", str(WHEELS), *DEPS[py]])
        for w in own:
            shutil.copy(stage / w, WHEELS / w)
        wheels = sorted(p.name for p in WHEELS.glob("*.whl"))
        manifest = (EXT / "blender_manifest.toml.in").read_text(encoding="utf-8")
        manifest = (manifest.replace("@VERSION@", version).replace("@BLENDER_VERSIONS@", versions)
                    .replace("@WHEELS@", ",\n".join(f'  "./wheels/{w}"' for w in wheels)))
        (EXT / "blender_manifest.toml").write_text(manifest, encoding="utf-8")
        out = ROOT / "dist" / f"stepmesh_blender-{version}-{label}-windows-x64.zip"
        out.parent.mkdir(exist_ok=True)
        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
            for f in ("__init__.py", "blender_manifest.toml"):
                z.write(EXT / f, f)
            for w in wheels:
                z.write(WHEELS / w, f"wheels/{w}")
            for f in ("LICENSE", "THIRD_PARTY_NOTICES.md"):
                z.write(ROOT / f, f)
        print(f"built {out} ({out.stat().st_size / 1e6:.0f} MB)")
    shutil.rmtree(stage, ignore_errors=True)


if __name__ == "__main__":
    main()
