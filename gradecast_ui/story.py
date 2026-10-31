"""'How GradeCast thinks': a four-step interactive walkthrough of the model.

Each step shows one picture, says what is happening in plain English, and why
it matters. Technical names appear only as small grey captions. The four steps
are: your semester so far, learning how you score (Bayesian updating),
forecasting the final exam (gradient boosting), and playing out 10,000
semesters (Monte Carlo).
"""

import pandas as pd
import streamlit as st

from gradecast.course import Course
from gradecast.explain import belief_curves, convergence_snapshots
from gradecast_ui.charts import (
    PLOT_CONFIG,
    belief_figure,
    convergence_figure,
    forecast_figure,
    ghost_figure,
    grade_bar_chart,
)
from gradecast_ui.compute import (
    course_key,
    forecasts_for,
    get_metrics,
    paths_for,
    probabilities_for,
    totals_for,
)
from gradecast_ui.styles import MUTED, SKY, SOFT_BLUE, esc, page_html

A_LINE = 85
# Run counts for the "watch it settle" animation (7 frames keeps the chart light).
CHECKPOINTS = (10, 30, 100, 300, 1000, 3000, 10_000)
STEPS = ["1 · Your semester", "2 · Learning you", "3 · The final exam", "4 · 10,000 semesters"]


def _intro(headline: str, technique: str) -> None:
    """Show a step's headline and its small grey technique caption."""
    page_html(f'<div class="step-title">{esc(headline)}</div><div class="tech">technique: {esc(technique)}</div>')


def _explain(what: str, why: str) -> None:
    """Show the 'What's happening' and 'Why this matters' boxes."""
    page_html(
        f'<div class="explain"><div class="what"><b>What\'s happening</b>{esc(what)}</div>'
        f'<div class="why"><b>Why this matters</b>{esc(why)}</div></div>'
    )


def _step_semester(course: Course) -> None:
    """Step 1: every item as a chip sized by weight, plus the points-vs-A-line bar."""
    _intro("Here is your semester so far", "weighted scoring")
    chips = []
    for item in course.items:
        # Weight decides how wide each chip is; hovering shows it as a tooltip.
        state = "done" if item.score is not None else "todo"
        score = f"{item.score:g}" if item.score is not None else "to come"
        chips.append(
            f'<div class="chip {state}" style="--w:{item.weight:g}" '
            f'title="{esc(item.name)}: {item.weight:g}% of the course">'
            f'<div class="name">{esc(item.name)}</div><div class="score">{score}</div></div>'
        )
    page_html(f'<div class="timeline">{"".join(chips)}</div>')

    earned = course.current_total()
    available = course.remaining_weight()
    out_of_reach = max(100 - earned - available, 0)
    # Everything in the bar is measured in course points out of 100.
    page_html(
        f"""
        <div class="pbar" role="img" aria-label="{earned:.1f} points earned, {available:g} still available, A line at {A_LINE}">
        <div class="earned" style="width:{min(earned, 100):.2f}%"></div>
        <div class="avail" style="left:{min(earned, 100):.2f}%;width:{min(available, 100 - earned):.2f}%"></div>
        <div class="aline" style="left:{A_LINE}%"><span>A line</span></div>
        </div>
        <div class="legend">
        <span><i class="dot" style="background:{SKY}"></i>Earned {earned:.1f}</span>
        <span><i class="dot" style="background:{SOFT_BLUE};border:1px dashed {SKY}"></i>Still available {available:g}</span>
        <span><i class="dot" style="background:{MUTED}"></i>Already out of reach {out_of_reach:.1f}</span>
        </div>
        """
    )
    _explain(
        "Solid blue boxes are items you have scored; dashed boxes are still to come. The bar adds "
        "up the points you have earned and the points still on the table, and marks the 85 you need for an A.",
        "Every prediction starts from these facts. Boxes are wider when they are worth more, "
        "so you can see where the points still are.",
    )


def _step_learning(course: Course) -> None:
    """Step 2: the Bayesian belief curve moving toward the student as scores are counted."""
    _intro("The model learns how you score", "Bayesian updating")
    # Quiz-style and participation scores are what the Bayesian update learns from.
    scores = [i.score for i in course.items if i.score is not None and i.kind in ("quiz", "other")]
    if not scores:
        page_html(
            '<div class="callout blue">Add a few quiz or participation scores in the sidebar and '
            "watch the model learn about you. For now it only knows a typical student.</div>"
        )
    if len(scores) >= 2:
        n_used = st.slider("Count my first N scores", 0, len(scores), len(scores), key="bayes_n")
    else:
        n_used = len(scores)

    curves = belief_curves(scores, n_used)
    # Fix the y axis using the fullest curve so you can see the blue curve grow taller.
    fullest = belief_curves(scores, len(scores))
    y_max = max(fullest["posterior"].max(), fullest["prior"].max()) * 1.25
    st.plotly_chart(belief_figure(curves, y_max, n_used), width="stretch", config=PLOT_CONFIG)

    posterior, prior = curves["posterior_belief"], curves["prior_belief"]
    used = " · ".join(f"{s:g}" for s in scores[:n_used]) or "none yet"
    st.caption(
        f"Grey: where a typical student's next score would land ({prior.mean:.0f} ± {prior.sd:.0f}). "
        f"Blue: where yours would land after counting these scores ({used}): "
        f"{posterior.mean:.0f} ± {posterior.sd:.0f}."
    )
    _explain(
        "The model starts with a guess about a typical student, then lets each of your own scores "
        "pull that guess toward you. Drag the slider and watch the blue curve move and narrow.",
        "More scores → the model gets more sure about you. This is how it predicts your remaining quizzes.",
    )


def _verdict(forecast: dict) -> str:
    """One sentence comparing the score needed for an A with the model's range."""
    name, needed = forecast["name"], forecast["needed_for_a"]
    low, high = forecast["low"], forecast["high"]
    if needed <= 0:
        return f"If the rest goes as expected, you need almost nothing on the {name} to reach an A."
    if needed > 100:
        return f"If the rest goes as expected, an A would need more than a perfect {name}."
    if needed > high:
        return (f"The score you need ({needed:.0f}) is above the model's likely range "
                f"({low:.0f}-{high:.0f}), so an A would be a stretch.")
    if needed < low:
        return (f"The score you need ({needed:.0f}) is below the model's likely range "
                f"({low:.0f}-{high:.0f}), so an A looks likely if the rest goes as expected.")
    return (f"The score you need ({needed:.0f}) falls inside the model's likely range "
            f"({low:.0f}-{high:.0f}), so an A is a real possibility.")


def _step_forecast(course: Course) -> None:
    """Step 3: the exam forecast as a range, what the model looked at, and its report card."""
    _intro("The model forecasts your final exam", "gradient boosting")
    forecasts = forecasts_for(course_key(course))
    if not forecasts:
        page_html('<div class="callout blue">There is no exam left to forecast.</div>')
        return
    st.plotly_chart(forecast_figure(forecasts), width="stretch", config=PLOT_CONFIG)
    st.caption(
        "Each bar is the model's likely range (low to high) with a diamond at its middle forecast. "
        "The orange tick is the score you would need for an A, and the green zone is every score that is enough."
    )
    for forecast in forecasts:
        st.markdown(f"**{forecast['name']}.** {_verdict(forecast)}")

    # What the model looked at: the same two numbers for every exam.
    first = forecasts[0]
    coursework = f"{first['coursework']:.0f}" + ("" if first["coursework_known"] else " (assumed, no scores yet)")
    midterm = f"{first['midterm']:g}" if first["midterm"] is not None else "not taken yet"
    page_html(
        f"""
        <div class="tech">What the model looked at</div>
        <div class="inputs">
        <span class="badge">Coursework average: <b>{esc(coursework)}</b></span>
        <span class="badge">Mid term exam: <b>{esc(midterm)}</b></span>
        </div>
        """
    )

    # The model's honest report card, from the held-out test students.
    metrics = get_metrics()
    if metrics:
        report = metrics["with_midterm" if first["midterm"] is not None else "without_midterm"]
        gain = report["baseline_mae"] - report["mae"]
        page_html(
            f"""
            <div class="badges">
            <span class="badge">Typical error <b>±{report['mae']:.1f} points</b></span>
            <span class="badge">Baseline (just averaging your scores) <b>±{report['baseline_mae']:.1f} points</b></span>
            </div>
            """
        )
        if gain > 0.2:
            st.caption(f"On students it had never seen, the model's guess was about {gain:.1f} points closer than simply averaging their scores.")
        else:
            st.caption("Here, simply averaging your scores is about as good as the model, so treat the range as the real answer, not the middle.")
    _explain(
        "The model compares you with hundreds of real students who had similar early scores, "
        "and reports a range for the final: a low guess, a likely guess and a high guess.",
        "One exam score is never certain, so a range is more honest than a single number. "
        "It trained on 600+ real students and predicts a range, not one number.",
    )


def _roll_again() -> None:
    """Button callback: ask for a fresh simulated semester."""
    st.session_state.roll = st.session_state.get("roll", 0) + 1


def _step_simulation(course: Course) -> None:
    """Step 4: ghost semesters, one rollable semester, the settling histogram, and the grade bars."""
    _intro("Playing out 10,000 semesters", "Monte Carlo simulation")
    if not course.remaining():
        page_html('<div class="callout blue">Every item already has a score, so there is nothing left to simulate.</div>')
        return
    key = course_key(course)

    # Visual A: ~25 faint "ghost" semesters plus one highlighted, re-rollable semester.
    roll = st.session_state.get("roll", 0)
    rolled = paths_for(key, 1, 1000 + roll)
    st.plotly_chart(ghost_figure(paths_for(key, 25, 42), rolled), width="stretch", config=PLOT_CONFIG)
    st.caption(
        "Each thin line is one imagined semester: it climbs as items are graded and fans out after your last "
        "scored item. Mint lines end with an A (85 or more), grey lines do not. The dark line is the semester "
        "highlighted below."
    )
    st.button("Roll another semester", type="primary", key="roll_btn", on_click=_roll_again)
    drawn = [
        (name, score) for name, fixed, score in zip(rolled["names"], rolled["fixed"], rolled["scores"][0]) if not fixed
    ]
    st.dataframe(
        pd.DataFrame({"Item": [n for n, _ in drawn], "This semester it scored": [round(s) for _, s in drawn]}),
        hide_index=True, width="stretch",
    )
    summary = " · ".join(f"{name}: {score:.0f}" for name, score in drawn)
    page_html(
        f'<div class="callout green">{esc(summary)} → total <b>{rolled["finals"][0]:.1f}</b> → '
        f'<b>{esc(rolled["grades"][0])}</b></div>'
    )

    # Visual B: the histogram of final totals settling as runs accumulate.
    totals = totals_for(key)
    st.plotly_chart(
        convergence_figure(convergence_snapshots(totals, CHECKPOINTS), totals), width="stretch", config=PLOT_CONFIG
    )
    st.caption("Why 10,000? Run it only 10 times and the answer jumps around. Run it 10,000 times and it settles.")

    st.markdown("**How the 10,000 semesters ended**")
    st.plotly_chart(grade_bar_chart(probabilities_for(key)), width="stretch", config=PLOT_CONFIG)
    st.caption("Share of the simulated semesters that ended in each letter grade; the A bar is blue.")
    _explain(
        "For every item still to come, the computer draws a plausible score (from the Bayesian curve for quizzes, "
        "from the exam range for exams), adds everything up, and repeats that 10,000 times.",
        "Nobody can know the future, but counting how often each grade shows up turns uncertainty into a percentage you can use.",
    )


@st.fragment
def render_story(course: Course) -> None:
    """Draw the four-step walkthrough.

    This is a fragment, so clicking inside a step only redraws the story and
    not the whole page.

    Args:
        course: The student's course.
    """
    with st.container(key="card_story"):
        st.subheader("How GradeCast thinks 🧠")
        choice = st.segmented_control(
            "Step", STEPS, default=STEPS[0], key="story_step", label_visibility="collapsed"
        )
        # Clicking the selected step again would clear it; fall back to step 1.
        index = STEPS.index(choice) if choice in STEPS else 0
        st.progress((index + 1) / len(STEPS), text=f"Step {index + 1} of {len(STEPS)}")

        if index == 0:
            _step_semester(course)
        elif index == 1:
            _step_learning(course)
        elif index == 2:
            _step_forecast(course)
        else:
            _step_simulation(course)
