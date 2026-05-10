"""Plotly visualizations for the interactive Streamlit dashboard."""

from __future__ import annotations

from typing import Iterable, List, Optional, Tuple

import numpy as np
import plotly.graph_objects as go

try:
    from .grid_model import GRID_SIZE, Coord, Grid, ZONES, ZONE_COLORS, ZONE_SYMBOLS, iter_cells
except ImportError:
    from grid_model import GRID_SIZE, Coord, Grid, ZONES, ZONE_COLORS, ZONE_SYMBOLS, iter_cells

ZONE_TO_NUM = {zone: idx for idx, zone in enumerate(ZONES)}


def _base_zone_arrays(grid: Grid):
    z = np.zeros((GRID_SIZE, GRID_SIZE))
    text = []
    custom = []
    for r in range(GRID_SIZE):
        row_text = []
        row_custom = []
        for c in range(GRID_SIZE):
            cell = grid[r][c]
            zone = str(cell["zone"])
            z[r, c] = ZONE_TO_NUM[zone]
            badges: List[str] = [ZONE_SYMBOLS[zone]]
            if cell["is_hub"]:
                badges.append("Hub")
            if cell["is_charging"]:
                badges.append("Charge")
            if cell["is_medical_pickup"]:
                badges.append("Med")
            if cell["no_fly"]:
                badges.append("No-Fly")
            row_text.append("<br>".join(badges))
            row_custom.append(
                f"Cell ({r}, {c})<br>Zone: {zone}<br>Density: {cell['density']}<br>Demand: {cell['demand']}<br>No-fly: {cell['no_fly']}"
            )
        text.append(row_text)
        custom.append(row_custom)
    return z, text, custom


def zone_map_figure(
    grid: Grid,
    route: Optional[List[Coord]] = None,
    explored: Optional[Iterable[Coord]] = None,
    title: str = "AeroNet Lite City Grid",
) -> go.Figure:
    z, text, custom = _base_zone_arrays(grid)
    colorscale = []
    n = max(1, len(ZONES) - 1)
    for idx, zone in enumerate(ZONES):
        colorscale.append([idx / n, ZONE_COLORS[zone]])

    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(
            z=z,
            text=text,
            customdata=custom,
            texttemplate="%{text}",
            hovertemplate="%{customdata}<extra></extra>",
            colorscale=colorscale,
            showscale=False,
            x=list(range(GRID_SIZE)),
            y=list(range(GRID_SIZE)),
        )
    )

    if explored:
        xs = [c for _, c in explored]
        ys = [r for r, _ in explored]
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="markers",
                marker=dict(size=8, symbol="square-open"),
                name="Explored",
                hovertemplate="Explored (%{y}, %{x})<extra></extra>",
            )
        )

    if route:
        xs = [c for r, c in route]
        ys = [r for r, c in route]
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines+markers",
                line=dict(width=4),
                marker=dict(size=9),
                name="Route",
                hovertemplate="Route cell (%{y}, %{x})<extra></extra>",
            )
        )

    # Make grid lines visible.
    fig.update_xaxes(
        tickmode="linear",
        dtick=1,
        range=[-0.5, GRID_SIZE - 0.5],
        showgrid=True,
        gridwidth=1,
        side="top",
        title="Column",
    )
    fig.update_yaxes(
        tickmode="linear",
        dtick=1,
        range=[GRID_SIZE - 0.5, -0.5],
        showgrid=True,
        gridwidth=1,
        title="Row",
    )
    fig.update_layout(
        title=title,
        height=650,
        margin=dict(l=20, r=20, t=60, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def demand_heatmap_figure(grid: Grid, title: str = "Demand Heatmap") -> go.Figure:
    demand = np.array([[float(grid[r][c]["demand"]) for c in range(GRID_SIZE)] for r in range(GRID_SIZE)])
    hover = [
        [f"Cell ({r}, {c})<br>Demand: {demand[r, c]:.2f}<br>Zone: {grid[r][c]['zone']}" for c in range(GRID_SIZE)]
        for r in range(GRID_SIZE)
    ]
    fig = go.Figure(
        data=go.Heatmap(
            z=demand,
            customdata=hover,
            hovertemplate="%{customdata}<extra></extra>",
            colorbar=dict(title="Demand"),
        )
    )
    fig.update_xaxes(tickmode="linear", dtick=1, side="top", title="Column")
    fig.update_yaxes(tickmode="linear", dtick=1, autorange="reversed", title="Row")
    fig.update_layout(title=title, height=600, margin=dict(l=20, r=20, t=60, b=20))
    return fig


def confusion_matrix_figure(cm, labels, title: str = "Confusion Matrix") -> go.Figure:
    fig = go.Figure(
        data=go.Heatmap(
            z=cm,
            x=labels,
            y=labels,
            text=cm,
            texttemplate="%{text}",
            hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>",
            colorbar=dict(title="Count"),
        )
    )
    fig.update_layout(title=title, height=450, xaxis_title="Predicted", yaxis_title="Actual")
    return fig


def fleet_score_figure(options) -> go.Figure:
    top = options[:12]
    labels = [f"L{opt.light_count}-H{opt.heavy_count}" for opt in top]
    scores = [opt.score for opt in top]
    fig = go.Figure(data=go.Bar(x=labels, y=scores, hovertext=[str(opt.as_dict()) for opt in top]))
    fig.update_layout(title="Top Fleet Options", xaxis_title="Fleet mix", yaxis_title="Fitness score", height=430)
    return fig


def ga_history_figure(history) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[h["generation"] for h in history],
            y=[h["best_score"] for h in history],
            mode="lines+markers",
            name="Best score",
        )
    )
    fig.update_layout(title="Genetic Algorithm Progress", xaxis_title="Generation", yaxis_title="Best score", height=430)
    return fig
