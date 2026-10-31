"""The playground: try 'what if...' scenarios and watch the answer change live.

The student can pick a ready-made scenario (stable, bomb the final, skip a
quiz, locked in), then lock any remaining item to a score with a slider. A
mini version of the hero updates as they play, and a gauge shows the lowest
Final Exam score that keeps a 50% chance of an A.
"""

import math

import streamlit as st

from gradecast.course import Course
from gradecast.explain import format_probability, most_likely_letter
from gradecast.simulator import SCENARIOS
from gradecast_ui import motion
from gradecast_ui.compute import course_key, pins_key, probabilities_for, safe_zone_for
from gradecast_ui.hero import percent_center
from gradecast_ui.styles import esc, page_html

# Pastel tile colour for each scenario.
TILE_COLORS = {"Stable": "blue", "Bomb the final": "peach", "Skip a quiz": "peach", "Locked in": "green"}
DEFAULT_PIN = 80


def _select_scenario(name: str) -> None:
    """Button callback: make a scenario the active one."""
    st.session_state.active_scenario = name


def _reset() -> None:
    """Button callback: back to the Stable scenario with every lock off."""
    st.session_state.active_scenario = "Stable"
    for state_key in list(st.session_state.keys()):
        if state_key.startswith("lock_"):
            st.session_state[state_key] = False
        elif state_key.startswith("val_"):
            st.session_state[state_key] = DEFAULT_PIN


def _delta_text(points: int, is_baseline: bool) -> str:
    """Describe a scenario's change versus Stable in words and a sign (not colour alone)."""
    if is_baseline:
        return "baseline"
    if points == 0:
        return "no change"
    return f"{'+' if points > 0 else '−'}{abs(points)} pts vs Stable"


def _scenario_cards(course: Course, key: tuple, active: str) -> None:
    """Draw the four scenario tiles, each with its P(A), its change, and a select button."""
    baseline = probabilities_for(key)["A"]
    columns = st.columns(len(SCENARIOS))
    for index, (column, scenario) in enumerate(zip(columns, SCENARIOS)):
        pins, shift = scenario.build(course)
        chance = probabilities_for(key, pins_key(pins), shift)["A"]
        delta = round((chance - baseline) * 100)
        is_active = scenario.name == active
        # format_probability avoids a misleading 0% or 100%; esc() makes "<1%" safe in HTML.
        text = esc(format_probability(chance))
        with column:
            page_html(
                f'<div class="scn {TILE_COLORS[scenario.name]}{" active" if is_active else ""}" style="--i:{index}">'
                f'<div class="n">{esc(scenario.name)}</div><div class="p">{text}</div>'
                f'<div class="d">{esc(_delta_text(delta, scenario.name == "Stable"))}</div></div>'
            )
            st.button(
                "Selected" if is_active else "Use this",
                key=f"scn_{scenario.name}", type="primary" if is_active else "secondary",
                on_click=_select_scenario, args=(scenario.name,), width="stretch",
            )


def _custom_sliders(course: Course) -> dict[str, float]:
    """One lock toggle and slider per remaining item; locked items become pins."""
    pins: dict[str, float] = {}
    for item in course.remaining():
        toggle_col, slider_col = st.columns([1, 3])
        locked = toggle_col.toggle(f"Lock {item.name}", key=f"lock_{item.name}")
        # The starting value lives in session state only (giving it twice makes Streamlit warn).
        st.session_state.setdefault(f"val_{item.name}", DEFAULT_PIN)
        value = slider_col.slider(
            item.name, 0, 100, key=f"val_{item.name}",
            disabled=not locked, label_visibility="collapsed",
        )
        if locked:
            pins[item.name] = float(value)
    return pins


def _mini_hero(probabilities: dict[str, float]) -> None:
    """A small live version of the hero: ring, chance of an A, most likely grade."""
    center, is_low = percent_center(probabilities["A"], "mini")
    page_html(
        f"""
        <div class="mini">
        {motion.ring(probabilities["A"], "mini", center, light=True, warm=is_low)}
        <div>
        <div class="step-title">With these settings</div>
        <span class="pill">Most likely grade: {esc(most_likely_letter(probabilities))}</span>
        </div>
        </div>
        """
    )


def _safe_zone(course: Course, key: tuple, pins: dict[str, float], shift: float) -> None:
    """Gauge showing the lowest Final Exam score that keeps a 50% chance of an A."""
    # The Final Exam is what we search over, so it must not also be pinned.
    other_pins = {name: score for name, score in pins.items() if name != "Final Exam"}
    safe = safe_zone_for(key, "Final Exam", pins_key(other_pins), shift)
    if safe is None:
        sentence = "Even a perfect Final Exam would leave you under a 50% chance of an A with these settings."
        gauge = ""
    elif safe == 0:
        sentence = "You are already at least 50% likely to get an A, whatever you score on the Final Exam."
        gauge = '<div class="gauge"><div class="safe" style="left:0"></div></div>'
    else:
        needed = math.ceil(safe)
        sentence = f"Score at least {needed} on the Final Exam to keep a 50% chance of an A."
        gauge = (
            f'<div class="gauge"><div class="safe" style="left:{safe:.1f}%"></div>'
            f'<div class="mark" style="left:{safe:.1f}%"><span>{needed}</span></div></div>'
        )
    page_html(
        f'<div class="step-title">Safe zone</div><div>{esc(sentence)}</div>{gauge}'
        + ('<div class="gauge-scale"><span>0</span><span>Final Exam score</span><span>100</span></div>' if gauge else "")
    )
    st.caption("The green part of the bar is every Final Exam score that keeps you at least 50% likely to get an A.")


@st.fragment
def render_playground(course: Course) -> None:
    """Draw the playground.

    This is a fragment, so moving a slider redraws only this section.

    Args:
        course: The student's course.
    """
    with st.container(key="card_playground"):
        st.subheader("What if…? 🎛️")
        if not course.remaining():
            page_html('<div class="callout blue">Every item already has a score, so there is nothing to play with.</div>')
            return
        key = course_key(course)
        active = st.session_state.get("active_scenario", "Stable")
        _scenario_cards(course, key, active)

        scenario = next(s for s in SCENARIOS if s.name == active)
        scenario_pins, shift = scenario.build(course)
        st.markdown("**Custom what-if:** lock an item and drag its slider to see what happens.")
        custom_pins = _custom_sliders(course)
        # Locked sliders win over whatever the scenario pins.
        pins = {**scenario_pins, **custom_pins}
        if scenario_pins or shift:
            parts = [f"{name} = {score:g}" for name, score in scenario_pins.items()]
            if shift:
                parts.append(f"everything else {shift:+g} points")
            st.caption(f"Scenario “{active}” sets: " + ", ".join(parts))
        st.button("Reset", key="reset_play", on_click=_reset)

        _mini_hero(probabilities_for(key, pins_key(pins), shift))
        if "Final Exam" in {item.name for item in course.remaining()}:
            _safe_zone(course, key, pins, shift)
