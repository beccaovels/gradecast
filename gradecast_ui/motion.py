"""Pure-CSS motion helpers: count-up numbers and the filling progress ring.

Streamlit redraws the page on every interaction, and a CSS animation only
replays if the browser sees a NEW animation. The trick used here: every
animation gets its own keyframes name that includes the value being shown
(for example ``ring_hero_37``). When the value changes, the animation name
changes too, so the browser starts it again from the beginning.
"""

import math
import re

from gradecast_ui.styles import esc

# Radius of the ring in the 200 x 200 SVG, and its circumference (the full "dash").
RING_RADIUS = 84
RING_CIRCUMFERENCE = 2 * math.pi * RING_RADIUS


def count_up(text: str, uid: str) -> str:
    """A number that counts up from 0 to ``text`` when it appears or changes.

    Args:
        text: Already-formatted number such as "42", "42.6". Anything else
            (e.g. "Secured") is shown as plain text.
        uid: Name for this spot on the page (e.g. "metric_total").

    Returns:
        HTML (a style tag plus a span). The digits are drawn by CSS counters,
        so a hidden copy of the text is included for screen readers.
    """
    match = re.fullmatch(r"(\d+)(?:\.(\d))?", text)
    if not match:
        return f"<span>{esc(text)}</span>"
    whole, fraction = match.groups()
    # The value is part of the keyframes name, so a new value restarts the animation.
    name = f"cu_{uid}_{whole}_{fraction if fraction is not None else 'x'}"
    # The element's own value is the END of the animation; only the start is in keyframes.
    values = f"--cu-a:{whole};" + (f"--cu-b:{fraction};" if fraction is not None else "")
    css_class = "cu cu-dec" if fraction is not None else "cu"
    return (
        f"<style>@keyframes {name}{{from{{--cu-a:0;--cu-b:0}}}}</style>"
        f'<span class="{css_class}" style="{values}animation:{name} 900ms ease-out both">'
        f'<span class="sr-only">{esc(text)}</span></span>'
    )


def ring(fraction: float, uid: str, center_html: str, light: bool = False, warm: bool = False) -> str:
    """A circular progress ring that fills up to ``fraction``.

    Args:
        fraction: How full the ring is, 0 to 1.
        uid: Name for this ring (e.g. "hero").
        center_html: HTML shown in the middle of the ring.
        light: True for a ring on a white card (dark text), False on the blue hero.
        warm: True to use the warm orange accent (low chance).

    Returns:
        HTML for the ring.
    """
    fraction = min(max(fraction, 0.0), 1.0)
    # The dash offset is the part of the circle that stays EMPTY.
    offset = RING_CIRCUMFERENCE * (1 - fraction)
    name = f"ring_{uid}_{int(fraction * 1000)}"
    css_class = "ring-fill warm" if warm else "ring-fill"
    wrapper = "ring-wrap ring-light" if light else "ring-wrap"
    return (
        f"<style>@keyframes {name}{{from{{stroke-dashoffset:{RING_CIRCUMFERENCE:.2f}}}}}</style>"
        f'<div class="{wrapper}">'
        '<svg viewBox="0 0 200 200" aria-hidden="true">'
        f'<circle class="ring-track" cx="100" cy="100" r="{RING_RADIUS}"/>'
        f'<circle class="{css_class}" cx="100" cy="100" r="{RING_RADIUS}" '
        'transform="rotate(-90 100 100)" '
        f'style="stroke-dasharray:{RING_CIRCUMFERENCE:.2f};stroke-dashoffset:{offset:.2f};'
        f'animation:{name} 1000ms ease-out both"/>'
        "</svg>"
        f'<div class="ring-center">{center_html}</div></div>'
    )
