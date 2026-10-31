"""The page footer: honest limitations and a short About line."""

import streamlit as st

from gradecast_ui.styles import page_html

REPO_URL = "https://github.com/beccaovels/gradecast"


def render_footer() -> None:
    """Draw the limitations note (in an expander) and the About line."""
    with st.expander("Limitations: what GradeCast cannot know"):
        page_html(
            """
            <div class="callout peach">
            <ul>
            <li>The exam model learned from Portuguese secondary-school students, so it is a starting estimate for a university course, not a guarantee.</li>
            <li>With only a few scores the uncertainty is wide: early in the semester the answer rests mostly on a typical student.</li>
            <li>Grade curves are not modelled; raw weighted totals are used.</li>
            <li>Weights come from you. GradeCast can only check that they add up to 100.</li>
            </ul>
            </div>
            """
        )
    page_html(
        f'<p class="tech">About: GradeCast is an open-source project. '
        f'Code and data credits are on <a href="{REPO_URL}" target="_blank" rel="noopener">GitHub</a>.</p>'
    )
