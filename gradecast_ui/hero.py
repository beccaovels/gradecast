"""The top of the page: the animated answer, a plain sentence, and four metric cards.

The hero shows the chance of an A inside a ring that fills up while the number
counts up. It also handles the special cases: an A that is already secured,
one that is out of reach, and probabilities too small or too large to print as
a bare 0% or 100%.
"""

import re

import streamlit as st

from gradecast.course import Course
from gradecast.explain import a_status, format_probability, most_likely_letter, plain_summary
from gradecast_ui import motion
from gradecast_ui.course_table import reset_table
from gradecast_ui.styles import esc, page_html

A_LINE = 85


def percent_center(probability: float, uid: str, label: str = "Chance of an A") -> tuple[str, bool]:
    """Build the text inside a probability ring.

    Args:
        probability: Chance of an A, 0 to 1.
        uid: Name for the count-up animation.
        label: Small caption under the number.

    Returns:
        The HTML for the ring's middle, and whether the chance is low (warm colour).
    """
    text = format_probability(probability)
    # Only plain percentages like "37%" can count up; "<1%" and ">99%" fade in as text.
    if re.fullmatch(r"\d+%", text):
        number = f'<div class="ring-num">{motion.count_up(text[:-1], uid)}%</div>'
    else:
        number = f'<div class="ring-num">{esc(text)}</div>'
    return f'{number}<div class="ring-label">{esc(label)}</div>', probability < 0.2


def _metric_cards(course: Course) -> str:
    """HTML for the four small cards under the hero."""
    required = course.required_average(A_LINE)
    # "Needed average" has three special states besides a plain number.
    if required is None:
        needed = "Nothing left"
    elif required == 0:
        needed = "Secured"
    elif required > 100:
        needed = "Out of reach"
    else:
        needed = f"{required:.1f}"
    weight = course.remaining_weight()
    cards = [
        ("Current total", motion.count_up(f"{course.current_total():.1f}", "m_total")),
        ("Remaining weight", motion.count_up(f"{weight:g}", "m_weight") + "%"),
        ("Needed average for A", motion.count_up(needed, "m_needed")),
        ("Best possible", motion.count_up(f"{course.best_case():.1f}", "m_best")),
    ]
    html_cards = "".join(
        f'<div class="metric" style="--i:{index}"><div class="k">{label}</div><div class="v">{value}</div></div>'
        for index, (label, value) in enumerate(cards)
    )
    return f'<div class="metric-grid">{html_cards}</div>'


def render_hero(course: Course, probabilities: dict[str, float]) -> None:
    """Draw the hero banner, the weight warning, and the metric cards.

    Args:
        course: The student's course.
        probabilities: Letter -> probability from the simulation.
    """
    status = a_status(course, A_LINE)
    chance = probabilities["A"]

    # Pick what goes in the middle of the ring for each of the three situations.
    if status == "secured":
        ring_html = motion.ring(
            1.0, "hero_secured",
            '<div class="ring-num small">Secured</div><div class="ring-label">Chance of an A</div>',
        )
    elif status == "impossible":
        ring_html = motion.ring(
            0.0, "hero_out",
            '<div class="ring-num small">Out of reach</div><div class="ring-label">Chance of an A</div>',
        )
    else:
        center, is_low = percent_center(chance, "hero")
        ring_html = motion.ring(chance, "hero", center, warm=is_low)

    sentence = esc(plain_summary(course, probabilities, A_LINE))
    if status == "impossible":
        sentence += " Every point still counts, so finish as strong as you can."

    page_html(
        f"""
        <div class="hero"><div class="hero-grid">
        <div class="hero-copy">
        <h1 class="hero-title">GradeCast</h1>
        <p class="hero-tagline">Your chance of an A, not just the score you need.</p>
        <span class="pill">Most likely grade: {esc(most_likely_letter(probabilities))}</span>
        <p class="hero-sentence">{sentence}</p>
        </div>
        {ring_html}
        </div></div>
        """
    )

    # Weights that do not add up to 100 are worth a gentle heads-up.
    if abs(course.total_weight() - 100) > 0.01:
        page_html(
            f'<span class="pill pill-warn">Weights add up to {course.total_weight():g}%, '
            "not 100% - results are still computed</span>"
        )
    page_html(_metric_cards(course))


def render_empty_state(preset_name: str) -> None:
    """Friendly hint (and a one-click fix) when no scores have been entered.

    Args:
        preset_name: Preset to load the example scores into.
    """
    page_html(
        '<div class="callout blue">No scores yet. Enter a few in the sidebar, or load an example '
        "to see GradeCast in action.</div>"
    )
    st.button(
        "Load example scores", key="empty_load", type="primary",
        on_click=reset_table, args=(preset_name, True),
    )
