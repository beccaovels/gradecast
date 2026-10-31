"""The editable course table: converting between a Course and a DataFrame.

The sidebar shows the course as a table the student can edit. These helpers
convert a Course to that table and back, and reset the table to a preset
(with or without the example scores).
"""

import copy

import pandas as pd
import streamlit as st

from gradecast.course import DEMO_SCORES, PRESETS, Course, Item

TABLE_COLUMNS = ["Item", "Kind", "Weight (%)", "Score"]
KINDS = ["quiz", "exam", "other"]


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
