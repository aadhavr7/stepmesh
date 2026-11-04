"""OpenGL 3.3 core viewport for polygon meshes (PySide6 QOpenGLWidget).

Draws filled triangles (from the polygon mesh), the real polygon edges
(quads and ngons as they are), open edges in red, and an axis gizmo.
Mouse: left drag orbit, middle / right drag or shift+left pan, wheel zoom.
Keys: F frame, 1 front, 3 right, 7 top, 0 iso, W edges, S flat/smooth, P perspective.
"""
from __future__ import annotations

import ctypes
import math

import numpy as np
from OpenGL import GL
from PySide6.QtCore import QPoint, Qt, Signal
from PySide6.QtGui import QSurfaceFormat
from PySide6.QtOpenGLWidgets import QOpenGLWidget

MESH_VS = """
#version 330 core
layout(location=0) in vec3 aPos;
layout(location=1) in vec3 aNrm;
layout(location=2) in vec2 aTag;   // x = B-rep face id, y = polygon side count
uniform mat4 uMVP;
uniform mat4 uMV;
out vec3 vN;
out vec3 vP;
flat out vec2 vTag;
void main() {
    vN = mat3(uMV) * aNrm;
    vP = (uMV * vec4(aPos, 1.0)).xyz;
    vTag = aTag;
    gl_Position = uMVP * vec4(aPos, 1.0);
}
"""

MESH_FS = """
#version 330 core
in vec3 vN;
in vec3 vP;
flat in vec2 vTag;
uniform int uMode;        // 0 clay, 1 by surface, 2 by polygon type
uniform int uFlat;
uniform int uBackTint;
out vec4 frag;
vec3 hsv(float h, float s, float v) {
    vec3 k = clamp(abs(mod(h * 6.0 + vec3(0.0, 4.0, 2.0), 6.0) - 3.0) - 1.0, 0.0, 1.0);
    return v * mix(vec3(1.0), k, s);
}
void main() {
    vec3 n = uFlat == 1 ? normalize(cross(dFdx(vP), dFdy(vP))) : normalize(vN);
    if (uFlat == 0 && !gl_FrontFacing) n = -n;
    vec3 base = vec3(0.78, 0.77, 0.74);
    if (uMode == 1) {
        float h = fract(vTag.x * 0.61803398875);
        base = hsv(h, 0.45, 0.88);
    } else if (uMode == 2) {
        base = vTag.y < 3.5 ? vec3(0.95, 0.55, 0.22) : (vTag.y < 4.5 ? vec3(0.62, 0.71, 0.85) : vec3(0.45, 0.78, 0.45));
    }
    if (!gl_FrontFacing && uBackTint == 1) base = vec3(0.9, 0.18, 0.2);
    vec3 V = normalize(-vP);
    vec3 L1 = normalize(vec3(0.35, 0.55, 1.0));
    vec3 L2 = normalize(vec3(-0.6, -0.2, 0.4));
    float d = max(dot(n, L1), 0.0) * 0.75 + max(dot(n, L2), 0.0) * 0.2;
    float s = pow(max(dot(reflect(-L1, n), V), 0.0), 48.0) * 0.22;
    float rim = pow(1.0 - max(dot(n, V), 0.0), 3.0) * 0.12;
    frag = vec4(base * (0.25 + d) + s + rim, 1.0);
}
"""

LINE_VS = """
#version 330 core
layout(location=0) in vec3 aPos;
uniform mat4 uMVP;
void main() { gl_Position = uMVP * vec4(aPos, 1.0); }
"""

LINE_FS = """
#version 330 core
uniform vec4 uColor;
out vec4 frag;
void main() { frag = uColor; }
"""

GIZMO_VS = """
#version 330 core
layout(location=0) in vec3 aPos;
layout(location=1) in vec3 aCol;
uniform mat4 uMVP;
out vec3 vC;
void main() { vC = aCol; gl_Position = uMVP * vec4(aPos, 1.0); }
"""

GIZMO_FS = """
#version 330 core
in vec3 vC;
out vec4 frag;
void main() { frag = vec4(vC, 1.0); }
"""


def set_default_format():
    fmt = QSurfaceFormat()
    fmt.setVersion(3, 3)
    fmt.setProfile(QSurfaceFormat.CoreProfile)
    fmt.setDepthBufferSize(24)
    fmt.setSamples(4)
    QSurfaceFormat.setDefaultFormat(fmt)


def _compile(vs, fs):
    def sh(src, kind):
        s = GL.glCreateShader(kind)
        GL.glShaderSource(s, src)
        GL.glCompileShader(s)
        if not GL.glGetShaderiv(s, GL.GL_COMPILE_STATUS):
            raise RuntimeError(GL.glGetShaderInfoLog(s).decode())
        return s
    p = GL.glCreateProgram()
    a, b = sh(vs, GL.GL_VERTEX_SHADER), sh(fs, GL.GL_FRAGMENT_SHADER)
    GL.glAttachShader(p, a)
    GL.glAttachShader(p, b)
    GL.glLinkProgram(p)
    if not GL.glGetProgramiv(p, GL.GL_LINK_STATUS):
        raise RuntimeError(GL.glGetProgramInfoLog(p).decode())
    GL.glDeleteShader(a)
    GL.glDeleteShader(b)
    return p


def _perspective(fovy, aspect, near, far):
    f = 1.0 / math.tan(math.radians(fovy) / 2)
    m = np.zeros((4, 4))
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = 2 * far * near / (near - far)
    m[3, 2] = -1
    return m


def _ortho(h, aspect, near, far):
    m = np.eye(4)
    m[0, 0] = 1 / (h * aspect)
    m[1, 1] = 1 / h
    m[2, 2] = -2 / (far - near)
    m[2, 3] = -(far + near) / (far - near)
    return m


def _look_at(eye, target, up):
    f = target - eye
    f /= np.linalg.norm(f)
    s = np.cross(f, up)
    if np.linalg.norm(s) < 1e-9:
        s = np.cross(f, [0, 1, 0])
    s /= np.linalg.norm(s)
    u = np.cross(s, f)
    m = np.eye(4)
    m[0, :3], m[1, :3], m[2, :3] = s, u, -f
    m[:3, 3] = -m[:3, :3] @ eye
    return m


class Viewer(QOpenGLWidget):
    info = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMinimumSize(400, 300)
        self.target = np.zeros(3)
        self.radius = 50.0
        self.dist = 150.0
        self.yaw = math.radians(-55)
        self.pitch = math.radians(28)
        self.persp = True
        self.mode = 0
        self.show_edges = True
        self.flat = False
        self.back_tint = True
        self.show_open = True
        self._last = QPoint()
        self._gl_ready = False
        self._pending = None
        self.ntri = self.nedge = self.nopen = 0
        self.edge_color = (0.08, 0.09, 0.12, 0.85)
        self.bg_top = (0.30, 0.32, 0.36)
        self.bg_bottom = (0.13, 0.14, 0.16)

    # ---------------------------------------------------------------- data
    def set_arrays(self, A, frame=True):
        """A: stepmesh.meshbuild.MeshArrays (or None to clear)."""
        if A is None:
            self._pending = (None, None, None, None)
        else:
            P = A.points.astype(np.float32)
            N = A.normals.astype(np.float32)
            tri = P[A.tri_v].reshape(-1, 3)
            nrm = N[A.tri_n].reshape(-1, 3)
            tag = np.repeat(np.stack([A.tri_face, A.tri_sides], axis=1).astype(np.float32), 3, axis=0)
            soup = np.ascontiguousarray(np.hstack([tri, nrm, tag]), dtype=np.float32)
            edges = np.ascontiguousarray(P[A.edges].reshape(-1, 3), dtype=np.float32)
            opn = np.ascontiguousarray(P[A.open_edges].reshape(-1, 3), dtype=np.float32)
            self._pending = (soup, edges, opn, None)
            self.bbox = (P.min(axis=0).astype(float), P.max(axis=0).astype(float))
        if self._gl_ready:
            self.makeCurrent()
            self._upload()
            self.doneCurrent()
        if frame and A is not None:
            self.frame()
        self.update()

    def _upload(self):
        if self._pending is None:
            return
        soup, edges, opn, bbox = self._pending
        self._pending = None
        if soup is None:
            self.ntri = self.nedge = self.nopen = 0
            return
        GL.glBindVertexArray(self.vao_mesh)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.vbo_mesh)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, soup.nbytes, soup, GL.GL_STATIC_DRAW)
        stride = 8 * 4
        for loc, size, off in ((0, 3, 0), (1, 3, 12), (2, 2, 24)):
            GL.glEnableVertexAttribArray(loc)
            GL.glVertexAttribPointer(loc, size, GL.GL_FLOAT, False, stride, ctypes.c_void_p(off))
        self.ntri = len(soup)
        for vao, vbo, data, attr in ((self.vao_edge, self.vbo_edge, edges, "nedge"), (self.vao_open, self.vbo_open, opn, "nopen")):
            GL.glBindVertexArray(vao)
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, vbo)
            if len(data):
                GL.glBufferData(GL.GL_ARRAY_BUFFER, data.nbytes, data, GL.GL_STATIC_DRAW)
            GL.glEnableVertexAttribArray(0)
            GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, False, 12, ctypes.c_void_p(0))
            setattr(self, attr, len(data))
        GL.glBindVertexArray(0)

    # ---------------------------------------------------------------- camera
    def frame(self):
        if getattr(self, "bbox", None) is None:
            return
        lo, hi = self.bbox
        self.target = (lo + hi) / 2
        self.radius = max(float(np.linalg.norm(hi - lo)) / 2, 1e-6)
        self.dist = self.radius / math.sin(math.radians(35) / 2) * 1.05
        self.update()

    def view(self, name):
        yaw, pitch = {
            "front": (-90, 0), "back": (90, 0), "right": (0, 0), "left": (180, 0),
            "top": (-90, 89.9), "bottom": (-90, -89.9), "iso": (-55, 28),
        }[name]
        self.yaw, self.pitch = math.radians(yaw), math.radians(pitch)
        self.update()

    def _eye(self):
        cp = math.cos(self.pitch)
        d = np.array([cp * math.cos(self.yaw), cp * math.sin(self.yaw), math.sin(self.pitch)])
        return self.target + self.dist * d

    def _matrices(self):
        w, h = max(self.width(), 1), max(self.height(), 1)
        aspect = w / h
        eye = self._eye()
        V = _look_at(eye, self.target.copy(), np.array([0.0, 0.0, 1.0]))
        near = max(self.dist - self.radius * 3, self.dist * 1e-3, 1e-6)
        far = self.dist + self.radius * 3
        if self.persp:
            Pm = _perspective(35, aspect, near, far)
        else:
            Pm = _ortho(self.dist * math.tan(math.radians(17.5)), aspect, near, far)
        return Pm, V

    # ---------------------------------------------------------------- GL
    def initializeGL(self):
        self.prog_mesh = _compile(MESH_VS, MESH_FS)
        self.prog_line = _compile(LINE_VS, LINE_FS)
        self.prog_giz = _compile(GIZMO_VS, GIZMO_FS)
        self.vao_mesh, self.vao_edge, self.vao_open, self.vao_giz = GL.glGenVertexArrays(4)
        self.vbo_mesh, self.vbo_edge, self.vbo_open, self.vbo_giz = GL.glGenBuffers(4)
        giz = np.array([[0, 0, 0, 1, .3, .3], [1, 0, 0, 1, .3, .3],
                        [0, 0, 0, .3, 1, .3], [0, 1, 0, .3, 1, .3],
                        [0, 0, 0, .35, .55, 1], [0, 0, 1, .35, .55, 1]], dtype=np.float32)
        GL.glBindVertexArray(self.vao_giz)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.vbo_giz)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, giz.nbytes, giz, GL.GL_STATIC_DRAW)
        GL.glEnableVertexAttribArray(0)
        GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, False, 24, ctypes.c_void_p(0))
        GL.glEnableVertexAttribArray(1)
        GL.glVertexAttribPointer(1, 3, GL.GL_FLOAT, False, 24, ctypes.c_void_p(12))
        GL.glBindVertexArray(0)
        self._gl_ready = True
        try:
            self.info.emit(f"OpenGL {GL.glGetString(GL.GL_VERSION).decode()} on {GL.glGetString(GL.GL_RENDERER).decode()}")
        except Exception:
            pass
        self._upload()

    def _clear_gradient(self):
        # cheap vertical gradient: scissored clears
        h = max(self.height(), 1)
        ratio = self.devicePixelRatioF()
        W, H = int(self.width() * ratio), int(h * ratio)
        steps = 24
        GL.glEnable(GL.GL_SCISSOR_TEST)
        for i in range(steps):
            t = i / (steps - 1)
            c = [self.bg_bottom[k] * (1 - t) + self.bg_top[k] * t for k in range(3)]
            y0 = int(H * i / steps)
            y1 = int(H * (i + 1) / steps)
            GL.glScissor(0, y0, W, y1 - y0)
            GL.glClearColor(c[0], c[1], c[2], 1.0)
            GL.glClear(GL.GL_COLOR_BUFFER_BIT)
        GL.glDisable(GL.GL_SCISSOR_TEST)
        GL.glClear(GL.GL_DEPTH_BUFFER_BIT)

    def paintGL(self):
        ratio = self.devicePixelRatioF()
        W, H = int(self.width() * ratio), int(self.height() * ratio)
        GL.glViewport(0, 0, W, H)
        self._clear_gradient()
        if self.ntri == 0:
            return
        Pm, V = self._matrices()
        mvp = (Pm @ V).astype(np.float32)
        mv = V.astype(np.float32)
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glDisable(GL.GL_CULL_FACE)
        GL.glEnable(GL.GL_POLYGON_OFFSET_FILL)
        GL.glPolygonOffset(1.0, 1.0)
        GL.glUseProgram(self.prog_mesh)
        GL.glUniformMatrix4fv(GL.glGetUniformLocation(self.prog_mesh, "uMVP"), 1, True, mvp)
        GL.glUniformMatrix4fv(GL.glGetUniformLocation(self.prog_mesh, "uMV"), 1, True, mv)
        GL.glUniform1i(GL.glGetUniformLocation(self.prog_mesh, "uMode"), self.mode)
        GL.glUniform1i(GL.glGetUniformLocation(self.prog_mesh, "uFlat"), int(self.flat))
        GL.glUniform1i(GL.glGetUniformLocation(self.prog_mesh, "uBackTint"), int(self.back_tint))
        GL.glBindVertexArray(self.vao_mesh)
        # front faces first; back faces in a second pass pushed slightly
        # further back, so they never bleed through along shared silhouettes
        GL.glEnable(GL.GL_CULL_FACE)
        GL.glCullFace(GL.GL_BACK)
        GL.glDrawArrays(GL.GL_TRIANGLES, 0, self.ntri)
        GL.glCullFace(GL.GL_FRONT)
        GL.glPolygonOffset(2.0, 8.0)
        GL.glDrawArrays(GL.GL_TRIANGLES, 0, self.ntri)
        GL.glDisable(GL.GL_CULL_FACE)
        GL.glDisable(GL.GL_POLYGON_OFFSET_FILL)
        GL.glUseProgram(self.prog_line)
        GL.glUniformMatrix4fv(GL.glGetUniformLocation(self.prog_line, "uMVP"), 1, True, mvp)
        GL.glEnable(GL.GL_BLEND)
        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
        if self.show_edges and self.nedge:
            GL.glUniform4f(GL.glGetUniformLocation(self.prog_line, "uColor"), *self.edge_color)
            GL.glBindVertexArray(self.vao_edge)
            GL.glDrawArrays(GL.GL_LINES, 0, self.nedge)
        if self.show_open and self.nopen:
            GL.glDisable(GL.GL_DEPTH_TEST)
            GL.glUniform4f(GL.glGetUniformLocation(self.prog_line, "uColor"), 1.0, 0.15, 0.15, 1.0)
            GL.glBindVertexArray(self.vao_open)
            GL.glDrawArrays(GL.GL_LINES, 0, self.nopen)
            GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glDisable(GL.GL_BLEND)
        # axis gizmo, bottom left
        s = int(90 * ratio)
        GL.glViewport(0, 0, s, s)
        GL.glClear(GL.GL_DEPTH_BUFFER_BIT)
        R = V.copy()
        R[:3, 3] = 0
        R[2, 3] = -3.0
        G = _ortho(1.25, 1.0, 0.1, 10) @ R
        GL.glUseProgram(self.prog_giz)
        GL.glUniformMatrix4fv(GL.glGetUniformLocation(self.prog_giz, "uMVP"), 1, True, G.astype(np.float32))
        GL.glBindVertexArray(self.vao_giz)
        GL.glDrawArrays(GL.GL_LINES, 0, 6)
        GL.glBindVertexArray(0)

    # ---------------------------------------------------------------- input
    def mousePressEvent(self, e):
        self._last = e.position().toPoint()

    def mouseMoveEvent(self, e):
        p = e.position().toPoint()
        dx, dy = p.x() - self._last.x(), p.y() - self._last.y()
        self._last = p
        btn = e.buttons()
        pan = btn & (Qt.MiddleButton | Qt.RightButton) or (btn & Qt.LeftButton and e.modifiers() & Qt.ShiftModifier)
        if pan:
            Pm, V = self._matrices()
            right = V[0, :3]
            up = V[1, :3]
            k = 2 * self.dist * math.tan(math.radians(17.5)) / max(self.height(), 1)
            self.target -= (dx * right - dy * up) * k
        elif btn & Qt.LeftButton:
            self.yaw -= dx * 0.008
            self.pitch = max(-1.55, min(1.55, self.pitch + dy * 0.008))
        else:
            return
        self.update()

    def wheelEvent(self, e):
        f = 0.85 ** (e.angleDelta().y() / 120)
        self.dist = max(self.dist * f, self.radius * 0.01)
        self.update()

    def keyPressEvent(self, e):
        k = e.key()
        if k == Qt.Key_F:
            self.frame()
        elif k == Qt.Key_1:
            self.view("front")
        elif k == Qt.Key_3:
            self.view("right")
        elif k == Qt.Key_7:
            self.view("top")
        elif k == Qt.Key_0:
            self.view("iso")
        elif k == Qt.Key_W:
            self.show_edges = not self.show_edges
        elif k == Qt.Key_S:
            self.flat = not self.flat
        elif k == Qt.Key_P:
            self.persp = not self.persp
        else:
            super().keyPressEvent(e)
            return
        self.update()
