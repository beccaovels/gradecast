"""Design system for the GradeCast app: palette, fonts and one stylesheet.

Every colour is defined once here and exposed both as Python constants (for
Plotly charts) and as CSS variables (for the stylesheet). ``inject_css`` writes
the stylesheet into the page; ``page_html`` is the safe way to drop a chunk of
HTML into Streamlit.
"""

import html

import streamlit as st

# ---- Palette: the single source of truth for colours ----
SKY = "#2F80ED"  # primary
LIGHT_SKY = "#5EA8FF"  # start of the hero gradient
MINT = "#7EE0C3"  # positive
SOFT_GREEN = "#E6F6EA"  # soft green card
SOFT_BLUE = "#E3EEFF"  # soft blue card
SOFT_PEACH = "#FDEBDD"  # soft peach card
INK = "#111827"  # text and dark pills
MUTED = "#B8C4D6"  # muted chart colour
PAGE_BG = "#F5F9FF"  # page background
WARM = "#F2994A"  # danger / low chance (use sparingly)
GREY_TEXT = "#5B6678"  # secondary text, dark enough to read on white

PALETTE = {
    "sky": SKY, "light-sky": LIGHT_SKY, "mint": MINT, "soft-green": SOFT_GREEN,
    "soft-blue": SOFT_BLUE, "soft-peach": SOFT_PEACH, "ink": INK, "muted": MUTED,
    "page-bg": PAGE_BG, "warm": WARM, "grey-text": GREY_TEXT,
}

# The font is loaded from Google Fonts; if it cannot load, sans-serif is used.
FONT_IMPORT = (
    "@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:"
    "wght@400;500;600;700;800&display=swap');"
)

# The stylesheet. Palette values come from the CSS variables built in inject_css().
BASE_CSS = """
/* Integer properties the count-up animation can animate smoothly */
@property --cu-a { syntax: '<integer>'; inherits: false; initial-value: 0; }
@property --cu-b { syntax: '<integer>'; inherits: false; initial-value: 0; }

/* Fonts: set on the app and inherited, so icon fonts keep working */
.stApp { font-family: 'Plus Jakarta Sans', sans-serif; color: var(--ink); }
.stApp p, .stApp label, .stApp button, .stApp h1, .stApp h2, .stApp h3, .stApp h4,
.stApp input, .stApp textarea { font-family: inherit; }
.stApp h2, .stApp h3 { font-weight: 800; letter-spacing: -0.01em; }
.stApp p, .stApp li { font-size: 15.5px; line-height: 1.6; }
.block-container { max-width: 1180px; padding-top: 2rem; }

/* Entrance animation: fade and slide up */
@keyframes rise { from { opacity: 0; transform: translateY(14px); } to { opacity: 1; transform: none; } }
.rise { animation: rise 400ms ease both; animation-delay: calc(var(--i, 0) * 60ms); }

/* White rounded section cards (st.container(key="card_...")) */
[class*="st-key-card"] {
    background: #fff; border-radius: 20px; padding: 22px 24px; margin-bottom: 18px;
    box-shadow: 0 2px 14px rgba(47, 128, 237, 0.08); animation: rise 400ms ease both;
}

/* Buttons: dark primary pill, soft-blue secondary pill, gentle hover lift */
[data-testid="stBaseButton-primary"], [data-testid="stBaseButton-secondary"] {
    border-radius: 999px; border: none; padding: 0.35rem 1.3rem; font-weight: 700;
    transition: transform 150ms ease, box-shadow 150ms ease;
}
[data-testid="stBaseButton-primary"] { background: var(--ink); color: #fff; }
[data-testid="stBaseButton-primary"] p { color: #fff; }
[data-testid="stBaseButton-secondary"] { background: var(--soft-blue); color: var(--ink); }
[data-testid="stBaseButton-secondary"] p { color: var(--ink); }
[data-testid="stBaseButton-primary"]:hover, [data-testid="stBaseButton-secondary"]:hover {
    transform: translateY(-2px); box-shadow: 0 6px 16px rgba(17, 24, 39, 0.15);
}

/* Hero banner */
.hero {
    background: linear-gradient(135deg, var(--light-sky), var(--sky)); color: #fff;
    border-radius: 24px; padding: 32px; box-shadow: 0 8px 28px rgba(47, 128, 237, 0.25);
    animation: rise 400ms ease both;
}
.hero-grid { display: flex; flex-wrap: wrap; gap: 28px; align-items: center; }
.hero-copy { flex: 1 1 280px; }
.hero-title { font-size: clamp(2rem, 5vw, 3rem); font-weight: 800; line-height: 1.1; margin: 0; }
.hero-tagline { font-size: 1.05rem; opacity: 0.95; margin: 6px 0 18px 0; }
.hero-sentence { font-size: 1.1rem; font-weight: 600; line-height: 1.5; margin-top: 14px; }

/* Progress ring with a number inside */
.ring-wrap { position: relative; width: min(210px, 60vw); aspect-ratio: 1; flex: 0 0 auto; }
.ring-wrap svg { width: 100%; height: 100%; }
.ring-track { fill: none; stroke: rgba(255, 255, 255, 0.28); stroke-width: 16; }
.ring-fill { fill: none; stroke: var(--mint); stroke-width: 16; stroke-linecap: round; }
.ring-fill.warm { stroke: var(--warm); }
.ring-center { position: absolute; inset: 0; display: flex; flex-direction: column;
    align-items: center; justify-content: center; text-align: center; padding: 18%; }
.ring-num { font-size: 2.6rem; font-weight: 800; line-height: 1; }
.ring-num.small { font-size: 1.4rem; }
.ring-label { font-size: 0.85rem; opacity: 0.92; margin-top: 4px; }
.ring-light { color: var(--ink); }
.ring-light .ring-track { stroke: var(--soft-blue); }
.ring-light .ring-fill { stroke: var(--sky); }
.ring-light .ring-fill.warm { stroke: var(--warm); }

/* Count-up numbers: the digits are drawn by CSS counters while --cu-a/--cu-b animate */
.cu { counter-reset: cu-a var(--cu-a) cu-b var(--cu-b); }
.cu::before { content: counter(cu-a); }
.cu-dec::before { content: counter(cu-a) "." counter(cu-b); }
.sr-only { position: absolute; width: 1px; height: 1px; overflow: hidden; clip: rect(0, 0, 0, 0); }

/* Pills and call-outs */
.pill { display: inline-block; background: var(--mint); color: var(--ink);
    border-radius: 999px; padding: 4px 14px; font-weight: 700; }
.pill-soft { background: var(--soft-blue); }
.pill-warn { background: var(--soft-peach); }
.callout { border-radius: 16px; padding: 14px 18px; color: var(--ink); margin: 8px 0; }
.callout.blue { background: var(--soft-blue); }
.callout.green { background: var(--soft-green); }
.callout.peach { background: var(--soft-peach); }

/* Metric cards under the hero */
.metric-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: 14px; margin: 16px 0 22px 0; }
.metric { background: #fff; border-radius: 18px; padding: 16px 18px;
    box-shadow: 0 2px 14px rgba(47, 128, 237, 0.08);
    animation: rise 400ms ease both; animation-delay: calc(var(--i, 0) * 60ms); }
.metric .k { font-size: 0.75rem; color: var(--grey-text); text-transform: uppercase; letter-spacing: 0.05em; }
.metric .v { font-size: 1.9rem; font-weight: 800; line-height: 1.2; }

/* Small helper text */
.tech { color: var(--grey-text); font-size: 0.82rem; }
.step-title { font-size: 1.35rem; font-weight: 800; margin: 4px 0 2px 0; }

/* "What's happening" / "Why this matters" boxes */
.explain { display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 12px; margin-top: 12px; }
.explain div { border-radius: 16px; padding: 14px 16px; font-size: 0.95rem; line-height: 1.55; }
.explain .what { background: var(--soft-blue); }
.explain .why { background: var(--soft-green); }
.explain b { display: block; margin-bottom: 2px; }

/* Step 1: timeline of items */
.timeline { display: flex; flex-wrap: wrap; gap: 8px; margin: 10px 0 18px 0; }
.chip { flex: var(--w) 1 0; min-width: 92px; border-radius: 14px; padding: 10px 12px;
    border: 2px solid var(--sky); font-size: 0.8rem; cursor: default; }
.chip.done { background: var(--sky); color: #fff; }
.chip.todo { border-style: dashed; color: var(--ink); background: transparent; }
.chip .name { white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.chip .score { font-size: 1.15rem; font-weight: 800; }

/* Step 1: points bar with the A line */
.pbar { position: relative; height: 26px; border-radius: 999px; background: var(--muted); margin: 18px 0 10px 0; }
.pbar .earned { position: absolute; left: 0; top: 0; bottom: 0; background: var(--sky); border-radius: 999px 0 0 999px; }
.pbar .avail { position: absolute; top: 0; bottom: 0; background: repeating-linear-gradient(
    45deg, var(--soft-blue), var(--soft-blue) 6px, #fff 6px, #fff 12px); }
.pbar .aline { position: absolute; top: -8px; bottom: -8px; border-left: 3px dashed var(--ink); }
.pbar .aline span { position: absolute; top: -20px; left: -22px; font-size: 0.75rem; font-weight: 700; white-space: nowrap; }
.legend { display: flex; flex-wrap: wrap; gap: 6px 18px; font-size: 0.85rem; color: var(--grey-text); }
.dot { display: inline-block; width: 11px; height: 11px; border-radius: 50%; margin-right: 6px; }

/* Step 3: badges */
.badges { display: flex; flex-wrap: wrap; gap: 10px; margin: 8px 0; }
.badge { background: var(--soft-blue); border-radius: 14px; padding: 8px 14px; font-size: 0.9rem; }
.badge b { font-size: 1.1rem; }
.inputs { display: flex; flex-wrap: wrap; gap: 8px; margin: 6px 0 12px 0; }

/* Playground: scenario cards */
.scn { border-radius: 18px; padding: 16px 18px; border: 2px solid transparent; margin-bottom: 8px;
    animation: rise 400ms ease both; animation-delay: calc(var(--i, 0) * 60ms); }
.scn.blue { background: var(--soft-blue); } .scn.green { background: var(--soft-green); }
.scn.peach { background: var(--soft-peach); }
.scn.active { border-color: var(--sky); }
.scn .n { font-weight: 700; }
.scn .p { font-size: 1.9rem; font-weight: 800; line-height: 1.2; }
.scn .d { font-size: 0.85rem; font-weight: 700; }

/* Playground: live mini hero */
.mini { background: var(--soft-blue); border-radius: 20px; padding: 18px 22px; margin: 14px 0;
    display: flex; flex-wrap: wrap; gap: 22px; align-items: center; }
.mini .ring-wrap { width: min(150px, 50vw); }
.mini .ring-num { font-size: 1.9rem; }
.mini .ring-track { stroke: #fff; }

/* Playground: safe-zone gauge */
.gauge { position: relative; height: 22px; border-radius: 999px; background: var(--muted); margin: 34px 0 6px 0; }
.gauge .safe { position: absolute; top: 0; bottom: 0; right: 0; background: var(--mint); border-radius: 0 999px 999px 0; }
.gauge .mark { position: absolute; top: -8px; bottom: -8px; width: 4px; margin-left: -2px; background: var(--ink); border-radius: 2px; }
.gauge .mark span { position: absolute; top: -24px; left: 50%; transform: translateX(-50%); font-weight: 800; font-size: 0.9rem; white-space: nowrap; }
.gauge-scale { display: flex; justify-content: space-between; font-size: 0.8rem; color: var(--grey-text); }

/* Sidebar completeness pill */
.progress-pill { background: #fff; border-radius: 14px; padding: 8px 12px; font-size: 0.85rem; margin: 6px 0 10px 0; }

/* Phone-sized screens: less padding, smaller hero */
@media (max-width: 640px) {
    .hero { padding: 22px; }
    [class*="st-key-card"] { padding: 16px; }
}

/* People who prefer less motion get no animation at all */
@media (prefers-reduced-motion: reduce) {
    *, *::before, *::after { animation: none !important; transition: none !important; }
}
"""


def inject_css() -> None:
    """Write the font import, palette variables and stylesheet into the page."""
    variables = "".join(f"--{name}: {value};" for name, value in PALETTE.items())
    # @import has to come first in a stylesheet, then the :root variables.
    st.markdown(
        f"<style>{FONT_IMPORT}\n:root {{ {variables} }}\n{BASE_CSS}</style>",
        unsafe_allow_html=True,
    )


def page_html(markup: str) -> None:
    """Show a chunk of raw HTML.

    Markdown treats indented lines as code and blank lines as block breaks, so
    we strip indentation and blank lines first. Start the markup with a tag.

    Args:
        markup: HTML text.
    """
    cleaned = "\n".join(line.strip() for line in markup.splitlines() if line.strip())
    st.markdown(cleaned, unsafe_allow_html=True)


def esc(text: str) -> str:
    """Escape text (like an item name containing '&') for safe use in HTML."""
    return html.escape(str(text), quote=True)
