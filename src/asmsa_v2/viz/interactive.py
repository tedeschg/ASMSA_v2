"""Interactive latent-space plots.

- :func:`interactive_latent` returns a Plotly figure with hover-info on
  individual frames (requires ``plotly``).
- :func:`latent_browser` returns an ``ipywidgets`` layout that pairs a Plotly
  scatter with a py3Dmol structure viewer; clicking (or using the slider)
  updates the rendered frame (requires ``plotly``, ``py3Dmol``, ``ipywidgets``).

All deps are optional; informative ImportError is raised if a user calls a
function whose backend isn't installed.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import torch


def _to_np(z):
    if isinstance(z, torch.Tensor):
        z = z.detach().cpu().numpy()
    return np.asarray(z)


try:
    import plotly.graph_objects as go
    _PLOTLY_AVAILABLE = True
except ImportError:
    go = None
    _PLOTLY_AVAILABLE = False


def _require_plotly():
    if not _PLOTLY_AVAILABLE:
        raise ImportError(
            "plotly is not installed. Install with `pip install plotly` to use "
            "ASMSA v2's interactive latent plots."
        )


def interactive_latent(
    z,
    labels: Sequence | None = None,
    hover_text: Sequence[str] | None = None,
    title: str = "Latent space",
    marker_size: int = 5,
    opacity: float = 0.7,
    width: int = 600,
    height: int = 600,
    colorbar_label: str | None = None,
):
    """Return a Plotly scatter of the latent, with hover tooltips.

    ``labels`` can be int cluster ids, floats (continuous color), or strings
    (categorical). ``hover_text`` overrides the default "frame N" tooltip.
    """
    _require_plotly()
    z = _to_np(z)
    n = z.shape[0]
    if hover_text is None:
        hover_text = [f"frame {i}" for i in range(n)]

    marker = dict(size=marker_size, opacity=opacity)
    if labels is not None:
        labels = np.asarray(labels)
        if labels.dtype.kind in ("U", "O"):
            fig = go.Figure()
            for lab in np.unique(labels):
                mask = labels == lab
                fig.add_trace(go.Scattergl(
                    x=z[mask, 0], y=z[mask, 1], mode="markers",
                    name=str(lab),
                    text=[hover_text[i] for i in np.where(mask)[0]],
                    marker=dict(size=marker_size, opacity=opacity),
                    hovertemplate="%{text}<br>z=(%{x:.2f}, %{y:.2f})<extra>%{fullData.name}</extra>",
                ))
        else:
            marker.update(dict(color=labels, colorscale="Viridis",
                               colorbar=dict(title=colorbar_label or "")))
            fig = go.Figure(go.Scattergl(
                x=z[:, 0], y=z[:, 1], mode="markers",
                text=hover_text, marker=marker,
                hovertemplate="%{text}<br>z=(%{x:.2f}, %{y:.2f})<extra></extra>",
            ))
    else:
        fig = go.Figure(go.Scattergl(
            x=z[:, 0], y=z[:, 1], mode="markers",
            text=hover_text, marker=marker,
            hovertemplate="%{text}<br>z=(%{x:.2f}, %{y:.2f})<extra></extra>",
        ))

    fig.update_layout(title=title, width=width, height=height,
                      xaxis_title="z[0]", yaxis_title="z[1]",
                      template="plotly_white",
                      yaxis=dict(scaleanchor="x", scaleratio=1))
    return fig


def latent_browser(
    z,
    traj,
    labels: Sequence | None = None,
    width: int = 500,
    height: int = 500,
    structure_style: str = "sticks",
    component_colors: dict[str, str] | None = None,
    component_atoms: dict[str, Sequence[int]] | None = None,
):
    """Interactive latent explorer: click a point to render that frame in 3D.

    Returns an ipywidgets ``HBox`` of a plotly ``FigureWidget`` and a
    py3Dmol structure pane. Use in a notebook. The plotly point's hovertext
    includes the frame index.

    All three backends (plotly, ipywidgets, py3Dmol) must be installed.
    """
    _require_plotly()
    try:
        import ipywidgets as widgets
        from IPython.display import display
    except ImportError:
        raise ImportError("ipywidgets is required for `latent_browser`. "
                          "Install with `pip install ipywidgets`.")
    from .structure import _require_p3d, view_frame
    _require_p3d()

    z = _to_np(z)
    n = z.shape[0]

    scatter_fig = interactive_latent(z, labels=labels, width=width, height=height,
                                     title="click a point → structure updates")
    fig = go.FigureWidget(scatter_fig)

    output = widgets.Output()

    def _render(frame_idx: int):
        output.clear_output(wait=True)
        with output:
            v = view_frame(
                traj, int(frame_idx),
                width=width, height=height,
                style=structure_style,
                component_colors=component_colors,
                component_atoms=component_atoms,
            )
            display(v.show())

    frame_slider = widgets.IntSlider(min=0, max=n - 1, value=0,
                                     description="frame", continuous_update=False,
                                     layout=widgets.Layout(width=f"{width}px"))
    frame_slider.observe(lambda change: _render(change["new"]) if change["name"] == "value" else None)

    def _on_click(trace, points, state):
        if not points.point_inds:
            return
        idx = int(points.point_inds[0])
        frame_slider.value = idx

    for trace in fig.data:
        trace.on_click(_on_click)

    _render(0)
    return widgets.HBox([widgets.VBox([fig, frame_slider]), output])
