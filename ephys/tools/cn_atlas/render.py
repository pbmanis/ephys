"""Shared rendering helpers: a bright shader, colormaps, scalar->RGBA."""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg

BRIGHT_SHADER = "cn_bright"


def register_shaders() -> None:
    """Register a brighter Lambertian shader for scalar/region surfaces.

    pyqtgraph's stock 'shaded' shader floors unlit faces at 0.2x the vertex
    color, which turns the dark end of a colormap nearly black. This variant
    lifts the floor to 0.55 so colors stay readable while keeping 3D form.
    """
    from pyqtgraph.opengl import shaders

    if any(s.name == BRIGHT_SHADER for s in shaders.Shaders):
        return
    shaders.Shaders.append(
        shaders.ShaderProgram(
            BRIGHT_SHADER,
            [
                shaders.VertexShader(
                    """
                    varying vec3 normal;
                    void main() {
                        normal = normalize(gl_NormalMatrix * gl_Normal);
                        gl_FrontColor = gl_Color;
                        gl_BackColor = gl_Color;
                        gl_Position = ftransform();
                    }
                    """
                ),
                shaders.FragmentShader(
                    """
                    varying vec3 normal;
                    void main() {
                        float p = dot(normal, normalize(vec3(1.0, -1.0, 1.0)));
                        p = p < 0. ? 0. : p * 0.45;
                        vec4 color = gl_Color;
                        color.x = color.x * (0.55 + p);
                        color.y = color.y * (0.55 + p);
                        color.z = color.z * (0.55 + p);
                        gl_FragColor = color;
                    }
                    """
                ),
            ],
        )
    )


def translucent_no_depth_write() -> dict:
    """GL options for a translucent hull that must NOT hide what is behind it.

    pyqtgraph's stock 'translucent' keeps depth-writing on, so a translucent
    brain outline writes depth and occludes the (opaque) structures inside it,
    which show up as patches that change with rotation. Disabling the depth
    *write* (depth test stays on) lets inner structures remain fully visible.
    Back-face culling keeps the hull to a single near-surface layer so it reads
    as a faint ghost instead of stacking many translucent layers. Pair with
    ``item.setDepthValue(1)`` so the hull is drawn after opaque items.
    """
    from OpenGL import GL

    return {
        GL.GL_DEPTH_TEST: True,
        GL.GL_BLEND: True,
        GL.GL_ALPHA_TEST: False,
        GL.GL_CULL_FACE: True,
        "glCullFace": (GL.GL_BACK,),
        "glBlendFunc": (GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA),
        "glDepthMask": (False,),
    }


def spectral_colormap() -> pg.ColorMap:
    """Tonotopic colormap following the physical frequency<->wavelength mapping:
    low frequency (long wavelength) = red, high frequency (short wavelength) =
    blue, through the visible spectrum in between.
    """
    pos = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    col = np.array(
        [
            [230, 30, 30, 255],   # red      (low frequency)
            [240, 130, 20, 255],  # orange
            [235, 215, 30, 255],  # yellow
            [40, 180, 50, 255],   # green
            [30, 160, 220, 255],  # cyan-blue
            [40, 40, 210, 255],   # blue     (high frequency)
        ],
        dtype=np.ubyte,
    )
    return pg.ColorMap(pos, col)


def scalar_to_rgba(
    scalar: np.ndarray, cmap: pg.ColorMap, lo: float, hi: float, alpha: float
) -> np.ndarray:
    """Map a scalar field to Nx4 float RGBA (0-1) with a constant alpha."""
    span = hi - lo if hi > lo else 1.0
    norm = np.clip((np.asarray(scalar, dtype=np.float64) - lo) / span, 0.0, 1.0)
    norm = np.nan_to_num(norm, nan=0.0)
    rgba = cmap.map(norm, mode="float").astype(np.float32)
    rgba[:, 3] = alpha
    return rgba
