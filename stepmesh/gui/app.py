"""StepMesh main window."""
from __future__ import annotations

import os
import sys
import time
import traceback
from pathlib import Path

from PySide6.QtCore import QObject, QSettings, Qt, QThread, Signal, Slot
from PySide6.QtGui import QAction, QIcon, QKeySequence
from PySide6.QtWidgets import (
    QApplication, QButtonGroup, QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDockWidget,
    QDoubleSpinBox, QFileDialog, QFormLayout, QGridLayout, QGroupBox, QHBoxLayout, QLabel,
    QMainWindow, QMessageBox, QPlainTextEdit, QProgressBar, QPushButton, QRadioButton,
    QScrollArea, QVBoxLayout, QWidget,
)

from .. import __version__
from ..params import MeshParams
from .viewer import Viewer, set_default_format

APP = "StepMesh"
STEP_FILTER = "STEP files (*.step *.stp *.STEP *.STP);;BREP files (*.brep);;All files (*)"
PRESETS = {"Draft": 24.0, "Medium": 16.0, "Fine": 12.0, "Extra fine": 6.0}


def resource(name):
    base = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
    p = base / name
    return p if p.exists() else Path(__file__).resolve().parent / name


# --------------------------------------------------------------------- workers

class LoadWorker(QObject):
    done = Signal(object, dict)
    failed = Signal(str)

    def __init__(self, path, heal):
        super().__init__()
        self.path, self.heal = path, heal

    @Slot()
    def run(self):
        try:
            t = time.perf_counter()
            from ..occ import Topology, read_step
            if str(self.path).lower().endswith(".brep"):
                from OCP.BRep import BRep_Builder
                from OCP.BRepTools import BRepTools
                from OCP.TopoDS import TopoDS_Shape
                shape = TopoDS_Shape()
                BRepTools.Read_s(shape, str(self.path), BRep_Builder())
            else:
                shape = read_step(self.path, heal=self.heal)
            topo = Topology(shape)
            info = {"faces": len(topo.faces), "edges": len(topo.edges), "bodies": len(topo.body_names),
                    "seconds": time.perf_counter() - t}
            self.done.emit(shape, info)
        except Exception as exc:
            self.failed.emit(f"{exc}\n\n{traceback.format_exc()}")


class MeshWorker(QObject):
    progress = Signal(float, str)
    done = Signal(object, object, object, dict)
    failed = Signal(str)
    cancelled = Signal()

    def __init__(self, shape, params):
        super().__init__()
        self.shape, self.params = shape, params
        self.cancel = False

    def _cb(self, frac, msg):
        self.progress.emit(frac, msg)
        return not self.cancel

    @Slot()
    def run(self):
        from ..meshbuild import build_arrays
        from ..meshdata import stats
        from ..mesher import Cancelled, mesh_shape
        try:
            t = time.perf_counter()
            mesh, mesher = mesh_shape(self.shape, self.params, progress=self._cb)
            if self.params.output == "tris":
                from ..export_obj import triangulate_polys
                triangulate_polys(mesh)
            self.progress.emit(0.98, "preparing display")
            arrays = build_arrays(mesh, self.params.sharp_deg)
            st = stats(mesh)
            st["seconds"] = round(time.perf_counter() - t, 2)
            kinds = {}
            for fc in mesher.faces:
                if fc is not None:
                    kinds[fc.kind] = kinds.get(fc.kind, 0) + 1
            st["face_kinds"] = kinds
            st["warnings"] = list(mesher.log)
            self.done.emit(mesh, arrays, mesher, st)
        except Cancelled:
            self.cancelled.emit()
        except Exception as exc:
            self.failed.emit(f"{exc}\n\n{traceback.format_exc()}")


class ExportWorker(QObject):
    done = Signal(str, float)
    failed = Signal(str)

    def __init__(self, path, mesh, arrays, opts, header):
        super().__init__()
        self.path, self.mesh, self.arrays, self.opts, self.header = path, mesh, arrays, opts, header

    @Slot()
    def run(self):
        try:
            from ..exporters import export
            t = time.perf_counter()
            export(self.path, self.mesh, self.arrays, header=self.header, **self.opts)
            self.done.emit(str(self.path), time.perf_counter() - t)
        except Exception as exc:
            self.failed.emit(f"{exc}\n\n{traceback.format_exc()}")


# --------------------------------------------------------------------- dialogs

class ExportOptions(QDialog):
    def __init__(self, parent, ext, settings: QSettings):
        super().__init__(parent)
        self.setWindowTitle("Export options")
        self.s = settings
        form = QFormLayout(self)
        self.units = QComboBox()
        self.units.addItems(["Keep model units (x1)", "Millimetres to metres (x0.001)",
                             "Millimetres to centimetres (x0.1)", "Inches to metres (x0.0254)", "Custom"])
        self.scale = QDoubleSpinBox()
        self.scale.setDecimals(6)
        self.scale.setRange(1e-6, 1e6)
        self.scale.setValue(float(settings.value("export/scale", 1.0)))
        self.units.setCurrentIndex(int(settings.value("export/units", 0)))
        self.units.currentIndexChanged.connect(self._units)
        self._units(self.units.currentIndex())
        form.addRow("Units", self.units)
        form.addRow("Scale", self.scale)
        self.up = QComboBox()
        self.up.addItems(["Z up (keep CAD axes)", "Y up (convert)"])
        self.up.setCurrentIndex(int(settings.value("export/up", 0)))
        if ext == ".glb":
            self.up.setToolTip("glTF is always written Y up; this only tells it the source is Z up or already Y up")
        form.addRow("Up axis", self.up)
        self.weld = QComboBox()
        self.weld.addItems(["Weld all (closed mesh)", "Split at sharp edges", "Each surface separate (stacked copies)"])
        self.weld.setCurrentIndex(int(settings.value("export/weld", 0)))
        self.group = QComboBox()
        self.group.addItems(["Group per CAD surface", "Group per body"])
        self.group.setCurrentIndex(int(settings.value("export/group", 0)))
        if ext in (".obj", ".fbx"):
            form.addRow("Vertices", self.weld)
        if ext == ".obj":
            form.addRow("Groups", self.group)
        if ext == ".fbx":
            note = QLabel("One object per body. Units are written into the file (CAD millimetres), "
                          "so Blender, Max, Maya and Houdini import at true size.")
            note.setWordWrap(True)
            note.setStyleSheet("color: #9aa0a6")
            form.addRow(note)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        form.addRow(bb)

    def _units(self, i):
        vals = [1.0, 0.001, 0.1, 0.0254]
        self.scale.setEnabled(i == 4)
        if i < 4:
            self.scale.setValue(vals[i])

    def options(self):
        self.s.setValue("export/units", self.units.currentIndex())
        self.s.setValue("export/scale", self.scale.value())
        self.s.setValue("export/up", self.up.currentIndex())
        self.s.setValue("export/weld", self.weld.currentIndex())
        self.s.setValue("export/group", self.group.currentIndex())
        return {"scale": self.scale.value(), "up": "zy"[self.up.currentIndex()],
                "weld": ["all", "smooth", "none"][self.weld.currentIndex()],
                "group": ["face", "body"][self.group.currentIndex()]}


# --------------------------------------------------------------------- main window

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.settings = QSettings(APP, APP)
        self.setWindowTitle(APP)
        ico = resource("stepmesh.ico")
        if ico.exists():
            self.setWindowIcon(QIcon(str(ico)))
        self.resize(1400, 860)
        self.shape = None
        self.src_path = None
        self.mesh = self.arrays = self.mesher = None
        self.stats = None
        self._thread = None
        self._worker = None

        self.viewer = Viewer(self)
        self.viewer.info.connect(lambda s: self.log(s))
        self.setCentralWidget(self.viewer)
        self.setAcceptDrops(True)
        self._build_actions()
        self._build_settings_dock()
        self._build_display_dock()
        self._build_info_dock()
        # panels can be moved or floated but not closed (closing hid them for good)
        self._docks = {d.objectName(): d for d in self.findChildren(QDockWidget)}
        for d in self._docks.values():
            d.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)
        self.view_menu.addSeparator()
        self.view_menu.addAction(QAction("Reset panel layout", self, triggered=self.reset_layout))
        self.statusBar().showMessage("Open a STEP file (Ctrl+O) or drop one on the window")
        self.progress = QProgressBar()
        self.progress.setMaximumWidth(260)
        self.progress.setRange(0, 1000)
        self.progress.hide()
        self.statusBar().addPermanentWidget(self.progress)
        geo = self.settings.value("window/geometry")
        if geo is not None:
            self.restoreGeometry(geo)
        state = self.settings.value("window/state")
        if state is not None:
            self.restoreState(state)
        # a saved layout from an older version may have panels hidden: always show them
        for d in self._docks.values():
            d.show()
        self.toolbar.show()

    def reset_layout(self):
        """Put every panel back in its default place (also recovers off-screen floating panels)."""
        areas = (("dock_settings", Qt.LeftDockWidgetArea), ("dock_display", Qt.RightDockWidgetArea),
                 ("dock_info", Qt.RightDockWidgetArea))
        for name, area in areas:
            d = self._docks[name]
            d.setFloating(False)
            self.addDockWidget(area, d)
            d.show()
        self.toolbar.show()
        self.settings.remove("window/state")

    # ------------------------------------------------------------ UI build
    def _build_actions(self):
        m = self.menuBar()
        fm = m.addMenu("&File")
        self.act_open = QAction("&Open STEP...", self, shortcut=QKeySequence.Open, triggered=self.open_dialog)
        self.act_export = QAction("&Export mesh...", self, shortcut=QKeySequence("Ctrl+E"), triggered=self.export_dialog)
        self.act_export.setEnabled(False)
        fm.addAction(self.act_open)
        self.recent_menu = fm.addMenu("Open &recent")
        self._refresh_recent()
        fm.addAction(self.act_export)
        fm.addSeparator()
        fm.addAction(QAction("E&xit", self, shortcut=QKeySequence.Quit, triggered=self.close))
        vm = m.addMenu("&View")
        for name, key in (("Frame all", "F"), ("Front", "1"), ("Right", "3"), ("Top", "7"), ("Iso", "0")):
            a = QAction(name, self, shortcut=QKeySequence(key))
            a.triggered.connect(self.viewer.frame if name == "Frame all" else (lambda _=False, n=name.lower(): self.viewer.view(n)))
            vm.addAction(a)
        self.view_menu = vm
        hm = m.addMenu("&Help")
        hm.addAction(QAction("Controls", self, triggered=self.show_controls))
        hm.addAction(QAction("About", self, triggered=self.show_about))
        tb = self.addToolBar("Main")
        tb.setObjectName("toolbar_main")
        tb.toggleViewAction().setVisible(False)
        self.toolbar = tb
        tb.addAction(self.act_open)
        tb.addAction(self.act_export)

    def _build_settings_dock(self):
        d = QDockWidget("Mesh settings", self)
        d.setObjectName("dock_settings")
        w = QWidget()
        lay = QVBoxLayout(w)
        dp = MeshParams()
        s = self.settings

        def dspin(lo, hi, val, step, dec=3, suffix="", special=None):
            sp = QDoubleSpinBox()
            sp.setRange(lo, hi)
            sp.setDecimals(dec)
            sp.setSingleStep(step)
            sp.setValue(float(val))
            if suffix:
                sp.setSuffix(suffix)
            if special:
                sp.setSpecialValueText(special)
            return sp

        g1 = QGroupBox("Density")
        f1 = QFormLayout(g1)
        self.preset = QComboBox()
        self.preset.addItems(list(PRESETS) + ["Custom"])
        f1.addRow("Preset", self.preset)
        self.angle = dspin(0.5, 90, s.value("mesh/angle", dp.angle_deg), 1, 1, " deg")
        self.angle.setToolTip("Max angle between surface normals across one polygon. Smaller = smoother, denser.")
        f1.addRow("Angle", self.angle)
        self.max_edge = dspin(0, 1e6, s.value("mesh/max_edge", 0), 1, 3, "", "off")
        self.max_edge.setToolTip("Divide larger than: polygons with edges longer than this get subdivided (model units).")
        f1.addRow("Divide larger than", self.max_edge)
        self.scope = QComboBox()
        self.scope.addItems(["curved", "planar", "all"])
        self.scope.setCurrentText(str(s.value("mesh/scope", dp.max_edge_scope)))
        f1.addRow("  applies to", self.scope)
        self.min_edge = dspin(0, 1e6, s.value("mesh/min_edge", 0), 0.1, 3, "", "off")
        self.min_edge.setToolTip("Avoid smaller than: stop subdividing below this edge length (model units).")
        f1.addRow("Avoid smaller than", self.min_edge)
        self.aspect = dspin(0, 100, s.value("mesh/aspect", 0), 0.5, 2, "", "off")
        self.aspect.setToolTip("Aspect ratio limit for all faces. When set, planar faces get a grid instead of one ngon.")
        f1.addRow("Aspect ratio limit", self.aspect)
        lay.addWidget(g1)

        g2 = QGroupBox("Topology")
        f2 = QFormLayout(g2)
        self.output = QComboBox()
        self.output.addItems(["N-gons", "Quads & triangles", "Triangles"])
        self.output.setCurrentIndex(int(s.value("mesh/output", 0)))
        f2.addRow("Output", self.output)
        self.band = QComboBox()
        self.band.addItems(["Clipped grid cells", "Triangle band"])
        self.band.setCurrentIndex(int(s.value("mesh/band", 0)))
        self.band.setToolTip("How curved surfaces meet trims and holes.")
        f2.addRow("At trims", self.band)
        self.trim_aspect = dspin(0, 100, s.value("mesh/trim_aspect", dp.trim_aspect), 0.5, 2, "", "off")
        f2.addRow("Trimmed face aspect", self.trim_aspect)
        self.sharp = dspin(0, 180, s.value("mesh/sharp", dp.sharp_deg), 1, 1, " deg")
        self.sharp.setToolTip("Normals stay split where surfaces meet sharper than this.")
        f2.addRow("Sharp edge angle", self.sharp)
        self.heal = QCheckBox("Heal STEP on import (ShapeFix)")
        self.heal.setChecked(s.value("mesh/heal", "true") in (True, "true"))
        f2.addRow(self.heal)
        lay.addWidget(g2)

        row = QHBoxLayout()
        self.btn_mesh = QPushButton("Update mesh")
        self.btn_mesh.setEnabled(False)
        self.btn_mesh.clicked.connect(self.remesh)
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.clicked.connect(self.cancel)
        row.addWidget(self.btn_mesh)
        row.addWidget(self.btn_cancel)
        lay.addLayout(row)
        self.auto = QCheckBox("Update automatically when settings change")
        self.auto.setChecked(s.value("mesh/auto", "false") in (True, "true"))
        lay.addWidget(self.auto)
        lay.addStretch(1)

        self._sync_preset_from_angle()
        self.preset.currentTextChanged.connect(self._preset_changed)
        for wdg in (self.angle, self.max_edge, self.min_edge, self.aspect, self.trim_aspect, self.sharp):
            wdg.valueChanged.connect(self._settings_changed)
        for wdg in (self.scope, self.output, self.band):
            wdg.currentIndexChanged.connect(self._settings_changed)
        self.angle.valueChanged.connect(self._sync_preset_from_angle)
        sc = QScrollArea()
        sc.setWidgetResizable(True)
        sc.setWidget(w)
        d.setWidget(sc)
        self.addDockWidget(Qt.LeftDockWidgetArea, d)

    def _build_display_dock(self):
        d = QDockWidget("Display", self)
        d.setObjectName("dock_display")
        w = QWidget()
        lay = QVBoxLayout(w)
        g = QGroupBox("Colour")
        gl = QVBoxLayout(g)
        self.color_group = QButtonGroup(self)
        for i, name in enumerate(("Clay", "By CAD surface", "By polygon type (quad / tri / ngon)")):
            rb = QRadioButton(name)
            self.color_group.addButton(rb, i)
            gl.addWidget(rb)
        self.color_group.button(0).setChecked(True)
        self.color_group.idClicked.connect(self._color_mode)
        lay.addWidget(g)
        self.cb_edges = QCheckBox("Show polygon edges (W)")
        self.cb_edges.setChecked(True)
        self.cb_flat = QCheckBox("Flat shading (S)")
        self.cb_back = QCheckBox("Highlight back faces (flipped normals)")
        self.cb_back.setChecked(True)
        self.cb_open = QCheckBox("Highlight open edges")
        self.cb_open.setChecked(True)
        self.cb_persp = QCheckBox("Perspective (P)")
        self.cb_persp.setChecked(True)
        for cb in (self.cb_edges, self.cb_flat, self.cb_back, self.cb_open, self.cb_persp):
            cb.toggled.connect(self._display_changed)
            lay.addWidget(cb)
        views = QGridLayout()
        for i, (name, key) in enumerate((("Front", "front"), ("Right", "right"), ("Top", "top"),
                                         ("Back", "back"), ("Left", "left"), ("Iso", "iso"))):
            b = QPushButton(name)
            b.clicked.connect(lambda _=False, k=key: self.viewer.view(k))
            views.addWidget(b, i // 3, i % 3)
        fb = QPushButton("Frame all (F)")
        fb.clicked.connect(self.viewer.frame)
        views.addWidget(fb, 2, 0, 1, 3)
        lay.addLayout(views)
        lay.addStretch(1)
        d.setWidget(w)
        self.addDockWidget(Qt.RightDockWidgetArea, d)

    def _build_info_dock(self):
        d = QDockWidget("Model info", self)
        d.setObjectName("dock_info")
        w = QWidget()
        lay = QVBoxLayout(w)
        self.info = QLabel("No model loaded")
        self.info.setTextFormat(Qt.RichText)
        self.info.setWordWrap(True)
        self.info.setTextInteractionFlags(Qt.TextSelectableByMouse)
        lay.addWidget(self.info)
        lay.addWidget(QLabel("Log"))
        self.logbox = QPlainTextEdit()
        self.logbox.setReadOnly(True)
        self.logbox.setMaximumBlockCount(5000)
        lay.addWidget(self.logbox, 1)
        d.setWidget(w)
        self.addDockWidget(Qt.RightDockWidgetArea, d)

    # ------------------------------------------------------------ helpers
    def log(self, msg):
        self.logbox.appendPlainText(f"[{time.strftime('%H:%M:%S')}] {msg}")

    def params(self) -> MeshParams:
        return MeshParams(
            angle_deg=self.angle.value(), max_edge=self.max_edge.value(), max_edge_scope=self.scope.currentText(),
            min_edge=self.min_edge.value(), aspect=self.aspect.value(), trim_aspect=self.trim_aspect.value(),
            output=["ngons", "quads", "tris"][self.output.currentIndex()],
            band=["cells", "cdt"][self.band.currentIndex()], sharp_deg=self.sharp.value(), heal=self.heal.isChecked())

    def _save_settings(self):
        s = self.settings
        s.setValue("mesh/angle", self.angle.value())
        s.setValue("mesh/max_edge", self.max_edge.value())
        s.setValue("mesh/scope", self.scope.currentText())
        s.setValue("mesh/min_edge", self.min_edge.value())
        s.setValue("mesh/aspect", self.aspect.value())
        s.setValue("mesh/trim_aspect", self.trim_aspect.value())
        s.setValue("mesh/output", self.output.currentIndex())
        s.setValue("mesh/band", self.band.currentIndex())
        s.setValue("mesh/sharp", self.sharp.value())
        s.setValue("mesh/heal", self.heal.isChecked())
        s.setValue("mesh/auto", self.auto.isChecked())

    def _sync_preset_from_angle(self, *_):
        a = self.angle.value()
        name = next((k for k, v in PRESETS.items() if abs(v - a) < 1e-6), "Custom")
        self.preset.blockSignals(True)
        self.preset.setCurrentText(name)
        self.preset.blockSignals(False)

    def _preset_changed(self, name):
        if name in PRESETS:
            self.angle.setValue(PRESETS[name])

    def _settings_changed(self, *_):
        if self.auto.isChecked() and self.shape is not None and self._thread is None:
            self.remesh()

    def _color_mode(self, i):
        self.viewer.mode = i
        self.viewer.update()

    def _display_changed(self, *_):
        v = self.viewer
        v.show_edges = self.cb_edges.isChecked()
        v.flat = self.cb_flat.isChecked()
        v.back_tint = self.cb_back.isChecked()
        v.show_open = self.cb_open.isChecked()
        v.persp = self.cb_persp.isChecked()
        v.update()

    def _busy(self, on, text=""):
        self.btn_mesh.setEnabled(not on and self.shape is not None)
        self.btn_cancel.setEnabled(on)
        self.act_open.setEnabled(not on)
        self.act_export.setEnabled(not on and self.mesh is not None)
        self.progress.setVisible(on)
        if on:
            self.progress.setValue(0)
            self.statusBar().showMessage(text)

    def _start(self, worker, *connections):
        th = QThread(self)
        worker.moveToThread(th)
        th.started.connect(worker.run)
        for sig, slot in connections:
            sig.connect(slot)
        self._thread, self._worker = th, worker
        th.start()

    def _finish_thread(self):
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait()
        self._thread = self._worker = None

    # ------------------------------------------------------------ open
    def _refresh_recent(self):
        self.recent_menu.clear()
        files = self.settings.value("recent", []) or []
        if isinstance(files, str):
            files = [files]
        for f in files[:10]:
            self.recent_menu.addAction(f, lambda p=f: self.open_path(p))
        self.recent_menu.setEnabled(bool(files))

    def _add_recent(self, path):
        files = self.settings.value("recent", []) or []
        if isinstance(files, str):
            files = [files]
        files = [path] + [f for f in files if f != path]
        self.settings.setValue("recent", files[:10])
        self._refresh_recent()

    def open_dialog(self):
        start = self.settings.value("last_dir", "")
        path, _ = QFileDialog.getOpenFileName(self, "Open STEP", start, STEP_FILTER)
        if path:
            self.open_path(path)

    def open_path(self, path):
        if self._thread is not None:
            return
        path = str(path)
        if not os.path.exists(path):
            QMessageBox.warning(self, APP, f"File not found:\n{path}")
            return
        self.settings.setValue("last_dir", str(Path(path).parent))
        self._add_recent(path)
        self.src_path = path
        self.shape = None
        self.mesh = self.arrays = None
        self.viewer.set_arrays(None)
        self.setWindowTitle(f"{Path(path).name} - {APP}")
        self.log(f"opening {path}")
        self._busy(True, f"Reading {Path(path).name}...")
        self.progress.setRange(0, 0)
        w = LoadWorker(path, self.heal.isChecked())
        self._start(w, (w.done, self._loaded), (w.failed, self._failed))

    def _loaded(self, shape, info):
        self._finish_thread()
        self.progress.setRange(0, 1000)
        self.shape = shape
        self.model_info = info
        self.log(f"read {info['faces']} faces, {info['edges']} edges, {info['bodies']} bodies in {info['seconds']:.1f}s")
        self._busy(False)
        self.remesh()

    # ------------------------------------------------------------ mesh
    def remesh(self):
        if self.shape is None or self._thread is not None:
            return
        self._save_settings()
        prm = self.params()
        self.log(f"meshing: angle {prm.angle_deg} deg, output {prm.output}, trims {prm.band}")
        self._busy(True, "Meshing...")
        w = MeshWorker(self.shape, prm)
        self._start(w, (w.progress, self._progress), (w.done, self._meshed), (w.failed, self._failed),
                    (w.cancelled, self._cancelled))

    def cancel(self):
        if isinstance(self._worker, MeshWorker):
            self._worker.cancel = True
            self.statusBar().showMessage("Cancelling...")

    def _progress(self, frac, msg):
        self.progress.setValue(int(frac * 1000))
        self.statusBar().showMessage(msg)

    def _meshed(self, mesh, arrays, mesher, st):
        self._finish_thread()
        self.mesh, self.arrays, self.mesher, self.stats = mesh, arrays, mesher, st
        first = self.viewer.ntri == 0
        self.viewer.set_arrays(arrays, frame=first)
        self._busy(False)
        self._show_info()
        for wmsg in st["warnings"]:
            self.log("warning: " + wmsg)
        self.statusBar().showMessage(
            f"{st['polys']:,} polygons ({st['quads']:,} quads, {st['tris']:,} tris, {st['ngons']:,} ngons), "
            f"{'watertight' if st['watertight'] else str(st['open_edges']) + ' open edges'}, {st['seconds']} s")
        snap = os.environ.get("STEPMESH_SNAPSHOT")
        if snap:
            # developer aid: render the window to a PNG, optionally in each colour mode, then quit
            from PySide6.QtCore import QTimer

            def grab():
                modes = os.environ.get("STEPMESH_SNAPSHOT_MODES", "0").split(",")
                for i, mode in enumerate(modes):
                    self.color_group.button(int(mode)).click()
                    QApplication.processEvents()
                    self.viewer.update()
                    QApplication.processEvents()
                    img = self.grab()
                    vimg = self.viewer.grabFramebuffer()
                    from PySide6.QtGui import QPainter
                    p = QPainter(img)
                    p.drawImage(self.viewer.mapTo(self, self.viewer.rect().topLeft()), vimg)
                    p.end()
                    img.save(snap.replace(".png", f"_{mode}.png") if len(modes) > 1 else snap)
                QApplication.quit()
            QTimer.singleShot(800, grab)

    def _cancelled(self):
        self._finish_thread()
        self._busy(False)
        self.log("meshing cancelled")
        self.statusBar().showMessage("Cancelled")

    def _failed(self, msg):
        self._finish_thread()
        self.progress.setRange(0, 1000)
        self._busy(False)
        self.log("error: " + msg)
        QMessageBox.critical(self, APP, msg.split("\n\n")[0])

    def _show_info(self):
        st, mi = self.stats, getattr(self, "model_info", {})
        ok = st["watertight"] and st["orientation_errors"] == 0
        badge = ("<span style='color:#3fb950'><b>Watertight</b></span>" if ok
                 else f"<span style='color:#f85149'><b>{st['open_edges']} open edges, {st['orientation_errors']} flipped</b></span>")
        kinds = ", ".join(f"{v} {k}" for k, v in sorted(st["face_kinds"].items()))
        self.info.setText(
            f"<b>{Path(self.src_path).name}</b><br>"
            f"CAD: {mi.get('faces', '?')} surfaces, {mi.get('edges', '?')} edges, {mi.get('bodies', '?')} bodies<br>"
            f"Surface types: {kinds}<br><br>"
            f"<table cellspacing=4>"
            f"<tr><td>Vertices</td><td align=right>{st['verts']:,}</td></tr>"
            f"<tr><td>Polygons</td><td align=right>{st['polys']:,}</td></tr>"
            f"<tr><td>&nbsp;&nbsp;Quads</td><td align=right>{st['quads']:,}</td></tr>"
            f"<tr><td>&nbsp;&nbsp;Triangles</td><td align=right>{st['tris']:,}</td></tr>"
            f"<tr><td>&nbsp;&nbsp;N-gons</td><td align=right>{st['ngons']:,}</td></tr>"
            f"<tr><td>Triangle equivalent</td><td align=right>{st['tri_equiv']:,}</td></tr>"
            f"<tr><td>Mesh time</td><td align=right>{st['seconds']} s</td></tr>"
            f"</table><br>{badge}")

    # ------------------------------------------------------------ export
    def export_dialog(self):
        if self.mesh is None or self._thread is not None:
            return
        from ..exporters import FORMATS
        filters = ";;".join(f"{desc} (*{ext})" for ext, desc in FORMATS.items())
        last_filter = self.settings.value("export/filter", "")
        start = str(Path(self.settings.value("export/dir", str(Path(self.src_path).parent))) / (Path(self.src_path).stem + ".obj"))
        path, flt = QFileDialog.getSaveFileName(self, "Export mesh", start, filters, last_filter)
        if not path:
            return
        ext = Path(path).suffix.lower()
        if ext not in FORMATS:
            ext = next((e for e, d in FORMATS.items() if d in flt), ".obj")
            path += ext
        self.settings.setValue("export/filter", flt)
        self.settings.setValue("export/dir", str(Path(path).parent))
        dlg = ExportOptions(self, ext, self.settings)
        if dlg.exec() != QDialog.Accepted:
            return
        opts = dlg.options()
        opts["sharp_deg"] = self.sharp.value()
        if ext != ".obj":
            opts.pop("group")
        if ext not in (".obj", ".fbx"):
            opts.pop("weld")
            opts.pop("sharp_deg")
        header = f"source: {Path(self.src_path).name}\nsettings: {self.params()}\nexported by {APP} {__version__}"
        self._busy(True, f"Exporting {Path(path).name}...")
        self.progress.setRange(0, 0)
        w = ExportWorker(path, self.mesh, self.arrays, opts, header)
        self._start(w, (w.done, self._exported), (w.failed, self._failed))

    def _exported(self, path, secs):
        self._finish_thread()
        self.progress.setRange(0, 1000)
        self._busy(False)
        size = os.path.getsize(path) / 1e6
        self.log(f"exported {path} ({size:.1f} MB, {secs:.1f}s)")
        self.statusBar().showMessage(f"Exported {Path(path).name}")

    # ------------------------------------------------------------ misc
    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):
        urls = e.mimeData().urls()
        if urls:
            self.open_path(urls[0].toLocalFile())

    def show_controls(self):
        QMessageBox.information(self, "Controls",
                                "Left drag: orbit\nMiddle or right drag, or Shift + left drag: pan\nWheel: zoom\n\n"
                                "F: frame all\n1 / 3 / 7 / 0: front / right / top / iso\n"
                                "W: polygon edges\nS: flat / smooth shading\nP: perspective / orthographic\n\n"
                                "Ctrl+O: open STEP, Ctrl+E: export")

    def show_about(self):
        QMessageBox.about(self, f"About {APP}",
                          f"<b>{APP} {__version__}</b><br>Clean-room STEP to quad-dominant mesh converter.<br>"
                          "Edge-first, isoparametric grids, clipped cells at trims, watertight output.<br><br>"
                          "Free and open source (MIT). Built on OpenCASCADE (OCP), Triangle, shapely, earcut and Qt (PySide6).")

    def closeEvent(self, e):
        self._save_settings()
        self.settings.setValue("window/geometry", self.saveGeometry())
        self.settings.setValue("window/state", self.saveState())
        if self._worker is not None and isinstance(self._worker, MeshWorker):
            self._worker.cancel = True
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait(3000)
        super().closeEvent(e)


def selftest(argv):
    """Headless check for packaged builds: StepMesh.exe --selftest in.step out.obj"""
    from ..exporters import export
    from ..meshbuild import build_arrays
    from ..meshdata import stats
    from ..mesher import mesh_shape
    from ..occ import read_step
    inp, out = argv[0], argv[1]
    mesh, _ = mesh_shape(read_step(inp), MeshParams())
    export(out, mesh, build_arrays(mesh))
    st = stats(mesh)
    msg = f"selftest: {st['polys']} polygons, watertight={st['watertight']}, wrote {out}"
    Path(out).with_suffix(".selftest.txt").write_text(msg)
    print(msg)
    return 0 if st["watertight"] else 2


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--selftest":
        sys.exit(selftest(sys.argv[2:]))
    set_default_format()
    QApplication.setAttribute(Qt.AA_ShareOpenGLContexts)
    app = QApplication(sys.argv)
    app.setApplicationName(APP)
    app.setStyle("Fusion")
    from PySide6.QtGui import QColor, QPalette
    pal = QPalette()
    for role, col in ((QPalette.Window, "#2b2d31"), (QPalette.WindowText, "#e6e6e6"), (QPalette.Base, "#1e1f22"),
                      (QPalette.AlternateBase, "#2b2d31"), (QPalette.Text, "#e6e6e6"), (QPalette.Button, "#3a3c42"),
                      (QPalette.ButtonText, "#e6e6e6"), (QPalette.Highlight, "#3d7bd9"), (QPalette.HighlightedText, "#ffffff"),
                      (QPalette.ToolTipBase, "#2b2d31"), (QPalette.ToolTipText, "#e6e6e6")):
        pal.setColor(role, QColor(col))
    pal.setColor(QPalette.Disabled, QPalette.Text, QColor("#777"))
    pal.setColor(QPalette.Disabled, QPalette.ButtonText, QColor("#777"))
    app.setPalette(pal)
    win = MainWindow()
    win.show()
    for a in sys.argv[1:]:
        if os.path.isfile(a):
            win.open_path(a)
            break
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
