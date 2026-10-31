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


def _short(name: str) -> str:
    """Shorten a long item name so it fits under an axis tick."""
    return name if len(name) <= 10 else name[:9] + "…"


def ghost_figure(paths: dict, highlight: dict | None = None) -> go.Figure:
    """'Ghost semesters': faint running-total lines from many simulated semesters.

    Args:
        paths: Output of ``explain.sample_paths`` with many semesters.
        highlight: Optional ``sample_paths`` output with one semester to draw boldly.

    Returns:
        A line chart where lines ending at an A (85+) are mint and the rest are muted.
    """
    steps = list(range(len(paths["names"]) + 1))
    fig = go.Figure()
    for running, final in zip(paths["cumulative"], paths["finals"]):
        fig.add_trace(go.Scatter(
            x=steps, y=running, mode="lines", opacity=0.85, hoverinfo="skip",
            line=dict(color=MINT if final >= 85 else MUTED, width=2.5),
        ))
    if highlight is not None:
        running, final, grade = highlight["cumulative"][0], highlight["finals"][0], highlight["grades"][0]
        # Only the last point gets a dot, so the line ends in a clear marker.
        fig.add_trace(go.Scatter(
            x=steps, y=running, mode="lines+markers", hoverinfo="skip",
            line=dict(color=INK, width=4), marker=dict(size=[0] * (len(steps) - 1) + [11], color=INK),
        ))
        fig.add_annotation(
            x=steps[-1], y=final, text=f"<b>{final:.1f} → {grade}</b>", showarrow=False,
            xanchor="left", xshift=10, font=dict(color=INK, size=13),
        )
    fig.add_hline(
        y=85, line_dash="dash", line_color=INK, line_width=1.5,
        annotation_text="A line", annotation_position="top left",
    )
    fig.update_xaxes(
        tickmode="array", tickvals=steps, ticktext=["Start"] + [_short(n) for n in paths["names"]],
        tickangle=-35, range=[0, len(steps) - 1 + 0.9],
    )
    fig.update_yaxes(range=[0, 100], title="Running total", showgrid=True, gridcolor="rgba(184,196,214,0.3)")
    return style_figure(fig, 380)


def convergence_figure(snapshots: list[np.ndarray], all_totals: np.ndarray) -> go.Figure:
    """An animated histogram that settles as more simulated runs are added.

    Args:
        snapshots: Output of ``explain.convergence_snapshots`` (first N totals per checkpoint).
        all_totals: Every simulated total, used to fix the bin edges for all frames.

    Returns:
        A figure with one animation frame per snapshot, a play button and a slider.
    """
    from gradecast.explain import format_probability

    low, high = float(all_totals.min()), float(all_totals.max())
    if high - low < 1e-9:
        low, high = low - 1, high + 1
    # Same 40 bins in every frame, counted with NumPy so frames stay small.
    edges = np.linspace(low, high, 41)
    centers = (edges[:-1] + edges[1:]) / 2
    colors = [MINT if center >= 85 else LIGHT_SKY for center in centers]

    def frame_parts(snapshot: np.ndarray) -> tuple[go.Bar, float, str]:
        """Bars, y-axis top, and title for one snapshot."""
        share = np.histogram(snapshot, bins=edges)[0] / len(snapshot) * 100
        bars = go.Bar(
            x=centers, y=share, width=(edges[1] - edges[0]) * 0.92,
            marker=dict(color=colors, line_width=0), hoverinfo="skip",
        )
        chance = format_probability(float(np.mean(snapshot >= 85)))
        title = f"After {len(snapshot):,} runs: chance of an A ≈ {chance}"
        return bars, float(share.max()) * 1.2, title

    parts = [frame_parts(snapshot) for snapshot in snapshots]
    frames = [
        go.Frame(
            name=str(len(snapshot)), data=[bars],
            layout=go.Layout(yaxis=dict(range=[0, top]), title=dict(text=title)),
        )
        for snapshot, (bars, top, title) in zip(snapshots, parts)
    ]
    fig = go.Figure(data=[parts[0][0]], frames=frames)
    fig.add_vline(x=85, line_dash="dash", line_color=INK, line_width=1.5, annotation_text="A line")
    play = dict(frame=dict(duration=1100, redraw=True), transition=dict(duration=400), fromcurrent=True)
    stop = dict(frame=dict(duration=0, redraw=False), mode="immediate", transition=dict(duration=0))
    fig.update_layout(
        title=dict(text=parts[0][2], x=0, font=dict(size=15)),
        xaxis_title="Final course total", yaxis_title="Share of runs",
        updatemenus=[dict(
            type="buttons", direction="left", x=0, y=-0.32, xanchor="left", showactive=False,
            buttons=[
                dict(label="Play", method="animate", args=[None, play]),
                dict(label="Pause", method="animate", args=[[None], stop]),
            ],
        )],
        sliders=[dict(
            active=0, x=0.18, len=0.8, y=-0.18, pad=dict(t=0, b=0),
            currentvalue=dict(prefix="Runs: ", font=dict(size=13)),
            steps=[
                dict(method="animate", label=f"{len(snapshot):,}",
                     args=[[str(len(snapshot))], dict(mode="immediate", frame=dict(duration=300, redraw=True),
                                                      transition=dict(duration=300))])
                for snapshot in snapshots
            ],
        )],
    )
    fig.update_yaxes(ticksuffix="%")
    styled = style_figure(fig, 430)
    # Leave room under the chart for the play button and slider.
    styled.update_layout(margin=dict(l=10, r=10, t=50, b=70))
    return styled
