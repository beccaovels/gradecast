"""GradeCast: Streamlit interface that shows the model thinking.

This file only wires the page together: set up the look, read the student's
course from the sidebar, then draw the hero (the answer), the story of how
the model got there, and the playground. The pieces live in ``gradecast_ui/``
and all of the maths lives in ``gradecast/``.
"""

import streamlit as st

from gradecast_ui.compute import course_key, get_models, probabilities_for
from gradecast_ui.course_table import reset_table
from gradecast_ui.footer import render_footer
from gradecast_ui.hero import render_empty_state, render_hero
from gradecast_ui.playground import render_playground
from gradecast_ui.sidebar import render_sidebar
from gradecast_ui.story import render_story
from gradecast_ui.styles import inject_css

st.set_page_config(page_title="GradeCast", page_icon="🎯", layout="wide")
inject_css()

# The course table survives reruns by living in the session.
if "table" not in st.session_state:
    reset_table("Statistics & Probability", with_demo_scores=False)

course, preset_name = render_sidebar()

# Without the trained model nothing can be predicted, so explain how to fix it.
if get_models() is None:
    st.error("Trained model not found. Run `python scripts/train_model.py` first, then reload.")
    st.stop()

if not course.items:
    st.info("Add some items to the table in the sidebar to get started.")
    st.stop()

# 1. The answer
key = course_key(course)
probabilities = probabilities_for(key)
render_hero(course, probabilities)
if not course.taken():
    render_empty_state(preset_name)

# 2. How GradeCast got there
render_story(course)

# 3. Your playground
render_playground(course)

# Footer
render_footer()
