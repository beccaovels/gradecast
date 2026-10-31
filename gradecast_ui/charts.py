"""Plotly figure builders for the story steps and the playground.

All charts share one look: transparent background, almost no gridlines, colours
from the palette, and labels placed directly on the data instead of in legends.
Each function takes plain data and returns a figure; none of them touches
Streamlit, so they are easy to read and reuse.
"""

import numpy as np
import plotly.graph_objects as go

from gradecast.grading import LETTERS
from gradecast_ui.styles import GREY_TEXT, INK, LIGHT_SKY, MINT, MUTED, SKY, WARM

# Hide Plotly's toolbar: it adds clutter for a beginner-friendly page.
PLOT_CONFIG = {"displayModeBar": False}
FONT = "Plus Jakarta Sans, sans-serif"


def style_figure(fig: go.Figure, height: int) -> go.Figure:
    """Apply the shared chart look.

    Args:
        fig: A Plotly figure.
        height: Height in pixels (width follows the page, so it works on phones).

    Returns:
        The same figure, restyled.
    """
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=30, b=10),
        font=dict(family=FONT, color=INK),
        showlegend=False,
    )
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(showgrid=False, zeroline=False)
    return fig


def grade_bar_chart(probabilities: dict[str, float], height: int = 300) -> go.Figure:
    """Bars for the chance of each letter grade, A on the left.

    Args:
        probabilities: Letter -> probability.
        height: Chart height in pixels.

    Returns:
        A figure with the A bar in blue, the others muted, percentages as labels.
    """
    from gradecast.explain import format_probability

    values = [probabilities[letter] for letter in LETTERS]
    fig = go.Figure(
        go.Bar(
            x=LETTERS,
            y=values,
            marker=dict(color=[SKY if letter == "A" else MUTED for letter in LETTERS], line_width=0),
            text=[format_probability(value) if value > 0 else "" for value in values],
            textposition="outside",
            cliponaxis=False,
        )
    )
    fig.update_yaxes(visible=False, range=[0, max(max(values) * 1.2, 0.05)])
    return style_figure(fig, height)


def belief_figure(curves: dict, y_max: float, n_used: int) -> go.Figure:
    """Two bell curves: a typical student (grey) and you after N scores (blue).

    Args:
        curves: Output of ``explain.belief_curves``.
        y_max: Fixed top of the y axis, so the blue curve visibly grows taller.
        n_used: How many of the student's scores were used (for the label).

    Returns:
        A Plotly figure.
    """
    x = curves["x"]
    prior, posterior = curves["prior_belief"], curves["posterior_belief"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=curves["prior"], mode="lines", line=dict(color=MUTED, width=3),
        fill="tozeroy", fillcolor="rgba(184,196,214,0.35)", hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=x, y=curves["posterior"], mode="lines", line=dict(color=SKY, width=3),
        fill="tozeroy", fillcolor="rgba(47,128,237,0.25)", hoverinfo="skip",
    ))

    def density_at(belief, at: float) -> float:
        """Height of a belief's curve at one score (for placing labels)."""
        z = (at - belief.mean) / belief.sd
        return float(np.exp(-0.5 * z**2) / (belief.sd * np.sqrt(2 * np.pi)))

    # Direct labels sit on opposite shoulders of the curves so they rarely collide.
    right_x = min(prior.mean + 1.3 * prior.sd, 99)
    left_x = max(posterior.mean - 1.3 * posterior.sd, 1)
    fig.add_annotation(
        x=right_x, y=density_at(prior, right_x), text="a typical student<br>(our starting guess)",
        showarrow=False, xanchor="left", yanchor="bottom", font=dict(color=GREY_TEXT, size=12),
    )
    label = "you, before any scores" if n_used == 0 else f"you, after {n_used} score{'s' if n_used != 1 else ''}"
    fig.add_annotation(
        x=left_x, y=density_at(posterior, left_x), text=label,
        showarrow=False, xanchor="right", yanchor="bottom", font=dict(color=SKY, size=12),
    )
    fig.update_xaxes(range=[0, 100], title="Score on your next quiz-style item")
    fig.update_yaxes(visible=False, range=[0, y_max])
    return style_figure(fig, 300)


def forecast_figure(forecasts: list[dict]) -> go.Figure:
    """Range bars for each remaining exam: low, likely and high, plus the score needed for an A.

    Args:
        forecasts: Output of ``explain.exam_forecasts``.

    Returns:
        A horizontal range chart (0-100) with one row per exam.
    """
    fig = go.Figure()
    for row, forecast in enumerate(forecasts):
        name, low, median, high = forecast["name"], forecast["low"], forecast["median"], forecast["high"]
        needed = forecast["needed_for_a"]
        # A soft mint zone shows every score on this exam that would be enough for an A.
        if needed <= 100:
            fig.add_shape(
                type="rect", x0=max(needed, 0), x1=100, y0=row - 0.42, y1=row + 0.42,
                fillcolor="rgba(126,224,195,0.30)", line_width=0, layer="below",
            )
        # The model's likely range, with the low and high ends labelled.
        fig.add_trace(go.Scatter(
            x=[low, high], y=[name, name], mode="lines+text",
            line=dict(color=LIGHT_SKY, width=16),
            text=[f"{low:.0f}", f"{high:.0f}"], textposition=["middle left", "middle right"],
            hoverinfo="skip",
        ))
        # The middle (median) forecast.
        fig.add_trace(go.Scatter(
            x=[median], y=[name], mode="markers+text",
            marker=dict(symbol="diamond", size=16, color=INK),
            text=[f"likely {median:.0f}"], textposition="bottom center", hoverinfo="skip",
        ))
        # The score you need, only drawn if it lies on the 0-100 scale.
        if 0 <= needed <= 100:
            fig.add_trace(go.Scatter(
                x=[needed], y=[name], mode="markers+text",
                marker=dict(symbol="line-ns-open", size=34, line=dict(color=WARM, width=4)),
                text=[f"needed for A: {needed:.0f}"], textposition="top center",
                textfont=dict(color=INK), hoverinfo="skip",
            ))
    fig.update_xaxes(range=[0, 100], title="Final exam score")
    fig.update_yaxes(type="category", autorange="reversed")
    return style_figure(fig, 150 + 90 * len(forecasts))
