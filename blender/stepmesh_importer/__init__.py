"""StepMesh importer for Blender: STEP files as clean, quad-dominant meshes.

File > Import > STEP (.step/.stp). Meshing runs in the background (Esc
cancels) using the bundled stepmesh library. Each CAD body becomes one
object with exact custom normals and a "cad_face" face attribute naming the
CAD surface every polygon came from. The StepMesh tab in the 3D View
sidebar re-meshes an imported model with new settings.
"""
import os
import sys
import threading
import time
import traceback
import uuid

import bpy
import numpy as np
from bpy.props import BoolProperty, EnumProperty, FloatProperty, IntProperty, PointerProperty, StringProperty
from bpy_extras.io_utils import ImportHelper

PRESETS = [("DRAFT", "Draft", "24 degrees"), ("MEDIUM", "Medium", "16 degrees"), ("FINE", "Fine", "12 degrees"),
           ("EXTRA", "Extra fine", "6 degrees"), ("CUSTOM", "Custom", "Set the angle yourself")]
PRESET_ANGLE = {"DRAFT": 24.0, "MEDIUM": 16.0, "FINE": 12.0, "EXTRA": 6.0}

# last STEP read, so re-meshing does not read the file again
_SHAPE_CACHE = {"key": None, "shape": None}


# ------------------------------------------------------------------ settings

def _preset_update(self, context):
    if self.preset in PRESET_ANGLE:
        self["angle"] = PRESET_ANGLE[self.preset]


class _Settings:
    """Mesh settings shared by the import operator and the stored per-object copy."""
    preset: EnumProperty(name="Density", items=PRESETS, default="FINE", update=_preset_update)
    angle: FloatProperty(name="Angle", default=12.0, min=0.5, max=90.0, subtype="NONE",
                         description="Max angle between surface normals across one polygon (degrees). Smaller is smoother and denser")
    max_edge: FloatProperty(name="Divide Larger Than", default=0.0, min=0.0,
                            description="Subdivide polygons with edges longer than this, in CAD units (usually mm). 0 = off")
    max_edge_scope: EnumProperty(name="Applies To", default="curved",
                                 items=[("curved", "Curved", "Curved surfaces only"), ("planar", "Planar", "Flat surfaces only"),
                                        ("all", "All", "Every surface")])
    min_edge: FloatProperty(name="Avoid Smaller Than", default=0.0, min=0.0,
                            description="Stop subdividing below this edge length, in CAD units. 0 = off")
    aspect: FloatProperty(name="Aspect Ratio Limit", default=0.0, min=0.0,
                          description="Max quad aspect ratio on all faces; flat faces then get a grid. 0 = off")
    output: EnumProperty(name="Polygons", default="ngons",
                         items=[("ngons", "N-gons", "Quads plus n-gons, cleanest"), ("quads", "Quads & Triangles", ""),
                                ("tris", "Triangles", "")])
    band: EnumProperty(name="At Trims", default="cells",
                       items=[("cells", "Clipped Grid Cells", "Grid cells cut by holes and trims become n-gons"),
                              ("cdt", "Triangle Band", "A band of triangles along holes and trims")])
    sharp: FloatProperty(name="Sharp Angle", default=20.0, min=0.0, max=180.0,
                         description="Vertices split where surfaces meet sharper than this (degrees)")
    weld: EnumProperty(name="Vertices", default="all",
                       items=[("all", "Weld All", "One closed mesh"),
                              ("smooth", "Split at Sharp Edges", "Separate vertices only where normals split"),
                              ("none", "Separate Surfaces", "Every CAD surface is its own island")])
    scale: FloatProperty(name="Scale", default=0.001, min=1e-9, precision=4,
                         description="CAD units to Blender units. 0.001 turns millimetres into metres")
    heal: BoolProperty(name="Heal STEP", default=True, description="Run OpenCASCADE ShapeFix on import")


SETTING_KEYS = ("preset", "angle", "max_edge", "max_edge_scope", "min_edge", "aspect", "output", "band",
                "sharp", "weld", "scale", "heal")


def _settings_dict(src):
    return {k: getattr(src, k) for k in SETTING_KEYS}


class StepMeshObjectSettings(bpy.types.PropertyGroup, _Settings):
    filepath: StringProperty(name="STEP File", subtype="FILE_PATH")
    import_id: StringProperty()
    body: IntProperty(default=-1)


def _draw_settings(layout, s):
    col = layout.column(align=False)
    col.use_property_split = True
    col.use_property_decorate = False
    col.prop(s, "preset")
    col.prop(s, "angle")
    col.prop(s, "max_edge")
    if s.max_edge > 0:
        col.prop(s, "max_edge_scope")
    col.prop(s, "min_edge")
    col.prop(s, "aspect")
    col.separator()
    col.prop(s, "output")
    col.prop(s, "band")
    col.prop(s, "weld")
    col.prop(s, "sharp")
    col.separator()
    col.prop(s, "scale")
    col.prop(s, "heal")


# ------------------------------------------------------------------ background job

class _Job:
    """Reads and meshes a STEP file in a background thread; mesh data is
    built into plain arrays so the main thread only has to hand them to Blender."""

    def __init__(self, filepath, settings):
        self.filepath, self.s = filepath, settings
        self.progress, self.message = 0.0, "starting"
        self.cancel = False
        self.done = False
        self.error = None
        self.bodies = None
        self.stats = None
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _cb(self, frac, msg):
        self.progress = 0.1 + 0.85 * frac
        self.message = msg
        return not self.cancel

    def _run(self):
        try:
            from stepmesh.meshdata import stats
            from stepmesh.mesher import Cancelled, mesh_shape
            from stepmesh.occ import read_step
            from stepmesh.params import MeshParams
            s = self.s
            key = (os.path.abspath(self.filepath), os.path.getmtime(self.filepath), s["heal"])
            if _SHAPE_CACHE["key"] == key:
                shape = _SHAPE_CACHE["shape"]
            else:
                self.message = "reading STEP"
                shape = read_step(self.filepath, heal=s["heal"])
                _SHAPE_CACHE.update(key=key, shape=shape)
            prm = MeshParams(angle_deg=s["angle"], max_edge=s["max_edge"], max_edge_scope=s["max_edge_scope"],
                             min_edge=s["min_edge"], aspect=s["aspect"], output=s["output"], band=s["band"],
                             sharp_deg=s["sharp"], weld=s["weld"])
            try:
                mesh, mesher = mesh_shape(shape, prm, progress=self._cb)
            except Cancelled:
                self.error = "cancelled"
                return
            if prm.output == "tris":
                from stepmesh.export_obj import triangulate_polys
                triangulate_polys(mesh)
            self.message = "building objects"
            self.stats = stats(mesh)
            self.bodies = _body_arrays(mesh, s)
        except Exception as exc:
            self.error = f"{exc}\n{traceback.format_exc()}"
        finally:
            self.done = True


def _body_arrays(mesh, s):
    """Per body: vertex positions, polygon loops, per-corner normals, CAD face ids."""
    weld, scale = s["weld"], s["scale"]
    poly_n = None
    if weld == "smooth":
        from stepmesh.export_obj import cluster_normals
        _, poly_n = cluster_normals(mesh, s["sharp"])
    P = mesh.points
    bodies = {}
    for pi, p in enumerate(mesh.polys):
        bodies.setdefault(p.body, []).append(pi)
    out = []
    for body, plist in sorted(bodies.items()):
        key_index, src, loops, starts, normals, faces = {}, [], [], [], [], []
        for pi in plist:
            p = mesh.polys[pi]
            starts.append(len(loops))
            for c, v in enumerate(p.verts):
                key = (p.face, v) if weld == "none" else (("n", poly_n[pi][c]) if weld == "smooth" else v)
                vi = key_index.get(key)
                if vi is None:
                    vi = key_index[key] = len(src)
                    src.append(v)
                loops.append(vi)
                normals.append(p.normals[c])
            faces.append(p.face)
        co = np.array([P[v] for v in src], dtype=np.float64).reshape(-1, 3) * scale
        out.append({
            "name": mesh.body_names.get(body, f"body_{body}"),
            "co": co.astype(np.float32).ravel(),
            "loops": np.asarray(loops, dtype=np.int32),
            "starts": np.asarray(starts, dtype=np.int32),
            "totals": np.diff(np.append(np.asarray(starts, dtype=np.int32), len(loops))).astype(np.int32),
            "normals": np.asarray(normals, dtype=np.float32).reshape(-1, 3),
            "faces": np.asarray(faces, dtype=np.int32),
        })
    return out


def _make_mesh(name, b):
    me = bpy.data.meshes.new(name)
    me.vertices.add(len(b["co"]) // 3)
    me.vertices.foreach_set("co", b["co"])
    me.loops.add(len(b["loops"]))
    me.loops.foreach_set("vertex_index", b["loops"])
    me.polygons.add(len(b["starts"]))
    me.polygons.foreach_set("loop_start", b["starts"])
    try:
        me.polygons.foreach_set("loop_total", b["totals"])   # read-only (derived) in newer Blender
    except (AttributeError, TypeError, RuntimeError):
        pass
    me.update(calc_edges=True)
    me.polygons.foreach_set("use_smooth", np.ones(len(b["starts"]), dtype=bool))
    if hasattr(me, "use_auto_smooth"):          # Blender 4.0 and older
        me.use_auto_smooth = True
    me.normals_split_custom_set(b["normals"])
    attr = me.attributes.new("cad_face", "INT", "FACE")
    attr.data.foreach_set("value", b["faces"])
    me.update()
    return me


class _JobOperator:
    """Modal runner shared by import and re-mesh."""
    _timer = None
    _job = None

    def _start(self, context, filepath, settings):
        self._job = _Job(filepath, settings)
        self._settings = settings
        self._filepath = filepath
        self._t0 = time.perf_counter()
        wm = context.window_manager
        self._timer = wm.event_timer_add(0.15, window=context.window)
        wm.progress_begin(0, 1000)
        wm.modal_handler_add(self)
        return {"RUNNING_MODAL"}

    def _stop(self, context):
        wm = context.window_manager
        wm.event_timer_remove(self._timer)
        wm.progress_end()
        if context.workspace:
            context.workspace.status_text_set(None)

    def modal(self, context, event):
        job = self._job
        if event.type == "ESC" and event.value == "PRESS":
            job.cancel = True
        if event.type != "TIMER":
            return {"PASS_THROUGH"}
        context.window_manager.progress_update(int(job.progress * 1000))
        if context.workspace:
            context.workspace.status_text_set(f"StepMesh: {job.message}  {int(job.progress * 100)}%   (Esc to cancel)")
        if not job.done:
            return {"PASS_THROUGH"}
        self._stop(context)
        if job.error == "cancelled":
            self.report({"WARNING"}, "StepMesh: cancelled")
            return {"CANCELLED"}
        if job.error:
            print(job.error)
            self.report({"ERROR"}, "StepMesh failed: " + job.error.splitlines()[0])
            return {"CANCELLED"}
        self._finish(context, job)
        st = job.stats
        self.report({"INFO"}, f"StepMesh: {st['polys']:,} polygons ({st['quads']:,} quads, {st['tris']:,} tris, "
                              f"{st['ngons']:,} ngons), {'watertight' if st['watertight'] else str(st['open_edges']) + ' open edges'}, "
                              f"{time.perf_counter() - self._t0:.1f} s")
        return {"FINISHED"}


def _store(obj, filepath, settings, import_id, body):
    s = obj.stepmesh
    for k, v in settings.items():
        setattr(s, k, v)
    s.filepath = filepath
    s.import_id = import_id
    s.body = body


class IMPORT_SCENE_OT_stepmesh(bpy.types.Operator, ImportHelper, _Settings, _JobOperator):
    """Import a STEP file as a clean quad-dominant mesh"""
    bl_idname = "import_scene.stepmesh"
    bl_label = "Import STEP (StepMesh)"
    bl_options = {"REGISTER", "UNDO"}

    filename_ext = ".step"
    filter_glob: StringProperty(default="*.step;*.stp;*.STEP;*.STP", options={"HIDDEN"})

    def draw(self, context):
        _draw_settings(self.layout, self)

    def execute(self, context):
        return self._start(context, self.filepath, _settings_dict(self))

    def _finish(self, context, job):
        import_id = uuid.uuid4().hex
        base = os.path.splitext(os.path.basename(self._filepath))[0]
        coll = bpy.data.collections.new(base)
        context.scene.collection.children.link(coll)
        for o in context.selected_objects:
            o.select_set(False)
        first = None
        for i, b in enumerate(job.bodies):
            name = base if len(job.bodies) == 1 else f"{base}_{b['name']}"
            obj = bpy.data.objects.new(name, _make_mesh(name, b))
            coll.objects.link(obj)
            _store(obj, self._filepath, self._settings, import_id, i)
            obj.select_set(True)
            first = first or obj
        if first:
            context.view_layer.objects.active = first


class OBJECT_OT_stepmesh_remesh(bpy.types.Operator, _JobOperator):
    """Re-mesh this imported STEP model with the settings shown in the StepMesh panel"""
    bl_idname = "object.stepmesh_remesh"
    bl_label = "Re-mesh"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        o = context.active_object
        return o is not None and o.type == "MESH" and bool(o.stepmesh.filepath)

    def execute(self, context):
        s = context.active_object.stepmesh
        path = bpy.path.abspath(s.filepath)
        if not os.path.isfile(path):
            self.report({"ERROR"}, f"STEP file not found: {path}")
            return {"CANCELLED"}
        self._import_id = s.import_id
        return self._start(context, path, _settings_dict(s))

    def _finish(self, context, job):
        targets = {o.stepmesh.body: o for o in bpy.data.objects
                   if o.type == "MESH" and o.stepmesh.import_id == self._import_id}
        for i, b in enumerate(job.bodies):
            obj = targets.get(i)
            if obj is None:
                continue
            old = obj.data
            new = _make_mesh(old.name, b)
            for mat in old.materials:
                new.materials.append(mat)
            obj.data = new
            if old.users == 0:
                bpy.data.meshes.remove(old)
            _store(obj, self._filepath, self._settings, self._import_id, i)


class VIEW3D_PT_stepmesh(bpy.types.Panel):
    bl_label = "StepMesh"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "StepMesh"

    def draw(self, context):
        layout = self.layout
        o = context.active_object
        if o is None or o.type != "MESH" or not o.stepmesh.filepath:
            layout.label(text="Select an imported STEP model,")
            layout.label(text="or import one:")
            layout.operator(IMPORT_SCENE_OT_stepmesh.bl_idname, icon="IMPORT")
            return
        s = o.stepmesh
        layout.label(text=os.path.basename(s.filepath), icon="FILE")
        me = o.data
        layout.label(text=f"{len(me.polygons):,} polygons, {len(me.vertices):,} vertices")
        _draw_settings(layout, s)
        layout.separator()
        layout.operator(OBJECT_OT_stepmesh_remesh.bl_idname, icon="FILE_REFRESH")


def _menu_import(self, context):
    self.layout.operator(IMPORT_SCENE_OT_stepmesh.bl_idname, text="STEP (.step/.stp)")


CLASSES = (StepMeshObjectSettings, IMPORT_SCENE_OT_stepmesh, OBJECT_OT_stepmesh_remesh, VIEW3D_PT_stepmesh)


def register():
    # worker processes for multi-core meshing need a plain Python, not Blender
    from stepmesh.parallel import set_embedded, set_worker_python
    set_embedded(True)
    exe = sys.executable or ""
    if os.path.basename(exe).lower().startswith("blender"):   # older Blender reported blender.exe
        cand = os.path.join(sys.prefix, "bin", "python.exe")
        if os.path.isfile(cand):
            set_worker_python(cand)
    for c in CLASSES:
        bpy.utils.register_class(c)
    bpy.types.Object.stepmesh = PointerProperty(type=StepMeshObjectSettings)
    bpy.types.TOPBAR_MT_file_import.append(_menu_import)


def unregister():
    bpy.types.TOPBAR_MT_file_import.remove(_menu_import)
    del bpy.types.Object.stepmesh
    for c in reversed(CLASSES):
        bpy.utils.unregister_class(c)
