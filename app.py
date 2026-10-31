"""GradeCast: Streamlit interface for grade-probability predictions.

This is the only file that imports Streamlit. It lets a student edit their
course scores, then calls the plain-Python modules in ``gradecast/`` to compute
the deterministic numbers, run the Monte Carlo simulation, and compare what-if
scenarios. All state lives in the Streamlit session; there is no database.
"""

import copy
import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from gradecast import model
from gradecast.course import DEMO_SCORES, PRESETS, Course, Item
from gradecast.grading import LETTERS
from gradecast.simulator import SCENARIOS, grade_probabilities, safe_zone, simulate

# ---- Palette (defined once, reused by charts and custom HTML) ----
SKY = "#2F80ED"  # primary colour and the "A" bar
LIGHT_SKY = "#5EA8FF"  # top of the hero gradient
MINT = "#7EE0C3"  # positive / "secured" accent
SOFT_GREEN = "#E6F6EA"  # soft green card
SOFT_BLUE = "#E3EEFF"  # soft blue card
SOFT_PEACH = "#FDEBDD"  # soft peach card
INK = "#111827"  # text and dark pills
MUTED = "#B8C4D6"  # non-A bars

A_LINE = 85  # course total that earns an A
MODEL_PATH = Path("models/gb_quantile.joblib")
METRICS_PATH = Path("models/metrics.json")
TABLE_COLUMNS = ["Item", "Kind", "Weight (%)", "Score"]
KINDS = ["quiz", "exam", "other"]

st.set_page_config(page_title="GradeCast", page_icon="🎯", layout="wide")

# Small stylesheet: white rounded cards, pill buttons and the hero banner.
st.markdown(
    f"""
    <style>
    /* Metric cards: white, rounded, very soft shadow */
    [data-testid="stMetric"] {{
        background: white; border-radius: 16px; padding: 14px 18px;
        box-shadow: 0 2px 10px rgba(47, 128, 237, 0.08);
    }}
    /* Bordered containers act as white section cards */
    [data-testid="stVerticalBlockBorderWrapper"] {{
        background: white; border-radius: 16px;
        box-shadow: 0 2px 10px rgba(47, 128, 237, 0.08);
    }}
    /* Dark pill buttons */
    .stButton > button {{
        border-radius: 999px; background: {INK}; color: white; border: none;
    }}
    .stButton > button:hover {{ background: {SKY}; color: white; }}
    /* Hero banner with the sky-blue gradient */
    .hero {{
        background: linear-gradient(135deg, {LIGHT_SKY}, {SKY});
        color: white; border-radius: 20px; padding: 28px 32px; margin-bottom: 16px;
    }}
    .hero h1 {{ color: white; margin: 0; padding: 0; }}
    .hero p {{ margin: 4px 0 0 0; opacity: 0.95; }}
    .hero .big {{ font-size: 3rem; font-weight: 800; margin-top: 16px; }}
    .pill {{
        display: inline-block; background: {MINT}; color: {INK};
        border-radius: 999px; padding: 4px 14px; font-weight: 700;
    }}
    /* Soft coloured call-out boxes */
    .callout {{ border-radius: 16px; padding: 14px 18px; color: {INK}; }}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_models() -> model.Models | None:
    """Load the trained exam models once and reuse them across reruns.

    Returns:
        The models, or None if the model file has not been trained yet.
    """
    if not MODEL_PATH.exists():
        return None
    return model.load(MODEL_PATH)


def course_to_table(course: Course) -> pd.DataFrame:
    """Convert a Course into the table shown in the editor.

    Args:
        course: Course to display.

    Returns:
        DataFrame with Item / Kind / Weight (%) / Score columns (Score is NaN
        when the item has not been taken).
    """
    rows = [
        {
            "Item": item.name,
            "Kind": item.kind,
            "Weight (%)": item.weight,
            "Score": float("nan") if item.score is None else item.score,
        }
        for item in course.items
    ]
    return pd.DataFrame(rows, columns=TABLE_COLUMNS)


def table_to_course(table: pd.DataFrame, name: str) -> Course:
    """Convert the edited table back into a Course.

    Rows without a name or weight (half-typed new rows) are ignored.

    Args:
        table: The table returned by ``st.data_editor``.
        name: Course name.

    Returns:
        A Course ready for the maths and simulation.
    """
    items: list[Item] = []
    for row in table.to_dict("records"):
        # Skip rows the user has only partly filled in.
        if pd.isna(row["Item"]) or not str(row["Item"]).strip() or pd.isna(row["Weight (%)"]):
            continue
        kind = row["Kind"] if row["Kind"] in KINDS else "other"
        score = None if pd.isna(row["Score"]) else float(row["Score"])
        items.append(Item(str(row["Item"]).strip(), float(row["Weight (%)"]), kind, score))
    return Course(name, items)


def reset_table(preset_name: str, with_demo_scores: bool) -> None:
    """Replace the editor's table with a fresh copy of a preset.

    Args:
        preset_name: Key in PRESETS.
        with_demo_scores: True to fill in the example scores, False to clear.
    """
    course = copy.deepcopy(PRESETS[preset_name])
    if with_demo_scores:
        for item in course.items:
            item.score = DEMO_SCORES.get(item.name)
    st.session_state.table = course_to_table(course)
    st.session_state.preset = preset_name
    # A new editor key forces the editor to forget old edits and show the new table.
    st.session_state.editor_version = st.session_state.get("editor_version", 0) + 1


def style_figure(fig: go.Figure) -> go.Figure:
    """Apply the shared chart look: transparent background, light gridlines.

    Args:
        fig: A Plotly figure.

    Returns:
        The same figure, restyled.
    """
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=30, b=10),
        font=dict(color=INK),
        showlegend=False,
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False, showticklabels=False)
    return fig


def grade_bar_chart(probabilities: dict[str, float]) -> go.Figure:
    """Bar chart of the probability of each letter grade (A to E).

    Args:
        probabilities: Letter -> probability.

    Returns:
        A Plotly figure with the A bar highlighted.
    """
    colors = [SKY if letter == "A" else MUTED for letter in LETTERS]
    values = [probabilities[letter] for letter in LETTERS]
    fig = go.Figure(
        go.Bar(
            x=LETTERS,
            y=values,
            marker=dict(color=colors, line_width=0),
            text=[f"{value:.0%}" for value in values],
            textposition="outside",
            cliponaxis=False,
        )
    )
    fig.update_layout(height=320)
    return style_figure(fig)


def total_histogram(totals) -> go.Figure:
    """Histogram of simulated final totals with a dashed line at the A line.

    Args:
        totals: Array of simulated totals.

    Returns:
        A Plotly figure.
    """
    fig = go.Figure(
        go.Histogram(x=totals, nbinsx=40, marker=dict(color=LIGHT_SKY, line_width=0))
    )
    fig.add_vline(
        x=A_LINE, line_dash="dash", line_color=INK,
        annotation_text="A line", annotation_position="top",
    )
    fig.update_layout(height=320, xaxis_title="Final course total")
    return style_figure(fig)


def most_likely(probabilities: dict[str, float]) -> str:
    """Letter grade with the highest probability."""
    return max(probabilities, key=probabilities.get)


# ---- Session state: the course table survives reruns ----
if "table" not in st.session_state:
    reset_table("Statistics & Probability", with_demo_scores=False)

exam_models = get_models()

# ---- Sidebar: course editor ----
with st.sidebar:
    st.header("Your course")
    preset_name = st.selectbox("Preset", list(PRESETS), key="preset_choice")
    # Switching preset replaces the table with that preset's blank layout.
    if preset_name != st.session_state.preset:
        reset_table(preset_name, with_demo_scores=False)
    load_col, clear_col = st.columns(2)
    if load_col.button("Load example scores", width="stretch"):
        reset_table(preset_name, with_demo_scores=True)
    if clear_col.button("Clear scores", width="stretch"):
        reset_table(preset_name, with_demo_scores=False)

    edited = st.data_editor(
        st.session_state.table,
        key=f"editor_{st.session_state.editor_version}",
        num_rows="dynamic",
        hide_index=True,
        width="stretch",
        column_config={
            "Item": st.column_config.TextColumn("Item", required=True),
            "Kind": st.column_config.SelectboxColumn("Kind", options=KINDS, required=True),
            "Weight (%)": st.column_config.NumberColumn("Weight (%)", step=0.5, required=True),
            "Score": st.column_config.NumberColumn("Score", step=0.5),
        },
    )
    course = table_to_course(edited, preset_name)
    for warning in course.validate():
        st.warning(warning)

# ---- Missing model: stop early with a clear instruction ----
if exam_models is None:
    st.error("Trained model not found. Run `python scripts/train_model.py` first, then reload.")
    st.stop()

if not course.items:
    st.info("Add some items to the table in the sidebar to get started.")
    st.stop()

# ---- Run the main simulation ----
totals = simulate(course, exam_models)
probabilities = grade_probabilities(totals)
chance_of_a = probabilities["A"]

# ---- Hero ----
st.markdown(
    f"""
    <div class="hero">
      <h1>GradeCast</h1>
      <p>Your chance of an A, not just the score you need.</p>
      <div class="big">Chance of an A: {chance_of_a:.0%}</div>
      <div style="margin-top:8px">Most likely grade: <span class="pill">{most_likely(probabilities)}</span></div>
    </div>
    """,
    unsafe_allow_html=True,
)

if not course.taken():
    st.markdown(
        f'<div class="callout" style="background:{SOFT_BLUE}">'
        "Enter a few scores or click <b>Load example scores</b>.</div>",
        unsafe_allow_html=True,
    )

# ---- Metric cards ----
required = course.required_average(A_LINE)
if required is None:
    needed_text = "Nothing left"
elif required == 0:
    needed_text = "Secured ✅"
elif required > 100:
    needed_text = "Out of reach"
else:
    needed_text = f"{required:.1f}"

metric_cols = st.columns(4)
metric_cols[0].metric("Current total", f"{course.current_total():.1f}")
metric_cols[1].metric("Remaining weight", f"{course.remaining_weight():g}%")
metric_cols[2].metric("Needed average for A", needed_text)
metric_cols[3].metric("Best possible", f"{course.best_case():.1f}")

# ---- Charts ----
chart_left, chart_right = st.columns(2)
with chart_left, st.container(border=True):
    st.subheader("Grade probabilities")
    st.plotly_chart(grade_bar_chart(probabilities), width="stretch")
with chart_right, st.container(border=True):
    st.subheader("Where your total could land")
    st.plotly_chart(total_histogram(totals), width="stretch")

# ---- Scenarios ----
with st.container(border=True):
    st.subheader("What-if scenarios")
    rows = []
    for scenario in SCENARIOS:
        scenario_pins, scenario_shift = scenario.build(course)
        scenario_probs = grade_probabilities(
            simulate(course, exam_models, pins=scenario_pins, shift=scenario_shift)
        )
        rows.append(
            {
                "Scenario": scenario.name,
                "P(A)": f"{scenario_probs['A']:.0%}",
                "Most likely grade": most_likely(scenario_probs),
            }
        )
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

    with st.expander("Custom scenario"):
        custom_pins: dict[str, float] = {}
        remaining_items = course.remaining()
        if not remaining_items:
            st.write("Nothing left to pin: every item already has a score.")
        for item in remaining_items:
            check_col, slider_col = st.columns([1, 3])
            pinned = check_col.checkbox(f"Pin {item.name}", key=f"pin_on_{item.name}")
            value = slider_col.slider(
                item.name, 0, 100, 80, key=f"pin_value_{item.name}", disabled=not pinned
            )
            if pinned:
                custom_pins[item.name] = float(value)
        if custom_pins:
            custom_probs = grade_probabilities(simulate(course, exam_models, pins=custom_pins))
            st.metric("Chance of an A with these pins", f"{custom_probs['A']:.0%}")
            st.plotly_chart(grade_bar_chart(custom_probs), width="stretch")
        else:
            st.caption("Tick \"Pin\" on an item to fix its score and see the effect.")

# ---- Safe zone ----
if "Final Exam" in {item.name for item in course.remaining()}:
    safe_score = safe_zone(course, exam_models, "Final Exam")
    if safe_score is None:
        message = "Even a perfect Final Exam gives you less than a 50% chance of an A."
    elif safe_score == 0:
        message = "You are already at least 50% likely to get an A, whatever the Final Exam brings."
    else:
        message = f"Keep your Final Exam above {safe_score:.0f} to stay at least 50% likely to get an A."
    st.markdown(
        f'<div class="callout" style="background:{SOFT_GREEN}"><b>Safe zone:</b> {message}</div>',
        unsafe_allow_html=True,
    )

# ---- How it works ----
with st.expander("How this works"):
    st.markdown(
        """
**1. The math.** Your current total, the weight still to come, the average you need
on the rest to reach 85 (an A), and your best possible total.

**2. Two predictors.**
- *Gradient boosting* predicts the **final exam** as a range (a low, middle and high
  guess), trained on real student data.
- *Bayesian updating* predicts **quizzes and other items**: it starts from a typical
  score and shifts toward your own scores as you add them.

**3. Simulation.** We play out the rest of the semester 10,000 times using those
predictions and count how often each letter grade happens.
        """
    )
    if METRICS_PATH.exists():
        metrics = json.loads(METRICS_PATH.read_text())
        metric_rows = [
            {
                "Model": label,
                "MAE (model)": round(metrics[key]["mae"], 2),
                "MAE (baseline)": round(metrics[key]["baseline_mae"], 2),
                "Coverage (ideal 0.80)": round(metrics[key]["coverage"], 2),
            }
            for key, label in [("with_midterm", "With midterm"), ("without_midterm", "Without midterm")]
        ]
        st.markdown(f"**Model test results** ({metrics['n_test']} held-out students)")
        st.dataframe(pd.DataFrame(metric_rows), hide_index=True, width="stretch")
    st.markdown(
        f'<div class="callout" style="background:{SOFT_PEACH}"><b>Limitations:</b> the exam '
        "model is trained on Portuguese secondary-school data, so it is a starting estimate, "
        "not a guarantee. With only a few scores the uncertainty is wide.</div>",
        unsafe_allow_html=True,
    )
