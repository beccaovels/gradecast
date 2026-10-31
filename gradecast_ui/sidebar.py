"""The sidebar: preset picker, example buttons, the editable course table.

It returns the Course the student has typed in, so the rest of the page can
work from it. A small pill under the table shows how much of the course is
already graded.
"""

import streamlit as st

from gradecast.course import PRESETS, Course
from gradecast_ui.course_table import KINDS, reset_table, table_to_course
from gradecast_ui.styles import page_html


def completeness_text(course: Course) -> str:
    """Summarise how much of the course has been scored.

    Args:
        course: The student's course.

    Returns:
        Text like "6 of 10 items scored - 55% of the course graded".
    """
    scored_weight = sum(item.weight for item in course.taken())
    total_weight = course.total_weight()
    # Avoid dividing by zero for an empty table.
    percent = scored_weight / total_weight * 100 if total_weight else 0
    return f"{len(course.taken())} of {len(course.items)} items scored · {percent:.0f}% of the course graded"


def render_sidebar() -> tuple[Course, str]:
    """Draw the sidebar and return the course being edited.

    Returns:
        The edited Course and the selected preset name.
    """
    with st.sidebar:
        st.header("Your course")
        preset_name = st.selectbox("Preset", list(PRESETS), key="preset_choice")
        # Switching preset replaces the table with that preset's blank layout.
        if preset_name != st.session_state.preset:
            reset_table(preset_name, with_demo_scores=False)
        load_col, clear_col = st.columns(2)
        load_col.button(
            "Load example scores", width="stretch", type="primary",
            on_click=reset_table, args=(preset_name, True),
        )
        clear_col.button(
            "Clear scores", width="stretch",
            on_click=reset_table, args=(preset_name, False),
        )

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
        page_html(f'<div class="progress-pill">{completeness_text(course)}</div>')
        for warning in course.validate():
            st.warning(warning)
    return course, preset_name
