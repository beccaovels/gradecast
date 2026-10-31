"""'How GradeCast thinks': a four-step interactive walkthrough of the model.

Each step shows one picture, says what is happening in plain English, and why
it matters. Technical names appear only as small grey captions. The four steps
are: your semester so far, learning how you score (Bayesian updating),
forecasting the final exam (gradient boosting), and playing out 10,000
semesters (Monte Carlo).
"""

import streamlit as st

from gradecast.course import Course
from gradecast.explain import belief_curves
from gradecast_ui.charts import PLOT_CONFIG, belief_figure
from gradecast_ui.compute import course_key
from gradecast_ui.styles import MUTED, SKY, SOFT_BLUE, esc, page_html

A_LINE = 85
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
        else:
            st.info("Coming up next.")
