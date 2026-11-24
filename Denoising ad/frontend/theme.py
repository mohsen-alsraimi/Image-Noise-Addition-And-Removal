"""
The look of Image Noise Lab: a small amount of CSS and a few HTML pieces.

Nothing here reads results or processes images. The pixel tiles on the Home
page are a fixed drawing that explains the idea; they are not experiment data.
Colours are chosen to stay readable in both the light and the dark theme.
"""

import html

import numpy as np
import streamlit as st

# Accent: a deep steel blue ("lab ink"). Green and red stay reserved for
# "improved" and "not improved".
ACCENT_RGB = "31, 95, 139"

CSS = f"""
<style>
:root {{ --lab-accent: rgb({ACCENT_RGB}); --lab-tint: rgba({ACCENT_RGB}, 0.08); }}

/* Comfortable reading width on wide screens */
[data-testid="stMainBlockContainer"] {{ max-width: 1180px; padding-top: 2rem; padding-bottom: 4rem; }}

/* Brand line above the navigation */
.lab-brand {{ display: flex; align-items: baseline; gap: 0.9rem; flex-wrap: wrap; margin-bottom: 0.4rem; }}
.lab-brand-name {{ font-family: "Source Serif", "Source Serif 4", Georgia, serif; font-size: 1.7rem; font-weight: 600; letter-spacing: -0.01em; }}
.lab-brand-line {{ opacity: 0.7; font-size: 0.95rem; }}

/* Numbered steps: used only where the content really is a sequence */
.lab-step {{ display: flex; gap: 0.75rem; align-items: flex-start; margin: 0.1rem 0 0.6rem; }}
.lab-step-n {{ flex: none; width: 1.8rem; height: 1.8rem; border-radius: 50%; display: grid; place-items: center;
              font-weight: 700; font-size: 0.95rem; color: #fff; background: var(--lab-accent); }}
.lab-step-t {{ font-family: "Source Serif", "Source Serif 4", Georgia, serif; font-size: 1.2rem; font-weight: 600; line-height: 1.8rem; }}
.lab-step-h {{ opacity: 0.72; font-size: 0.92rem; line-height: 1.35; margin-top: 0.1rem; max-width: 70ch; }}

/* Home: the pipeline drawing */
.lab-pipeline {{ display: flex; align-items: flex-start; gap: 0.6rem; flex-wrap: wrap; margin: 1.2rem 0 0.4rem; }}
.lab-stage {{ display: flex; flex-direction: column; align-items: center; gap: 0.45rem; width: 128px; text-align: center; }}
.lab-stage svg {{ border-radius: 8px; box-shadow: 0 0 0 1px rgba(128, 128, 128, 0.35); }}
.lab-stage b {{ font-size: 0.95rem; }}
.lab-stage span {{ font-size: 0.82rem; opacity: 0.7; line-height: 1.3; }}
.lab-arrow {{ opacity: 0.45; margin-top: 41px; }}

/* The verdict box in Results and in Try an image */
.st-key-verdict, .st-key-try_verdict {{ background: var(--lab-tint); }}

/* Small grey labels inside summary boxes */
.lab-fact {{ font-size: 0.8rem; opacity: 0.68; margin-bottom: 0.1rem; }}
.lab-fact-v {{ font-size: 1.02rem; font-weight: 600; line-height: 1.35; }}

/* Keep long dropdown labels readable */
[data-baseweb="select"] div {{ white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
</style>
"""


def apply_style():
    st.markdown(CSS, unsafe_allow_html=True)


def brand():
    st.markdown(
        '<div class="lab-brand"><span class="lab-brand-name">Image Noise Lab</span>'
        '<span class="lab-brand-line">Add noise to an image, clean it with filters, '
        "and measure what worked.</span></div>",
        unsafe_allow_html=True,
    )


def step(number, title, hint=None):
    """Header of one numbered step."""
    hint_html = f'<div class="lab-step-h">{html.escape(hint)}</div>' if hint else ""
    st.markdown(
        f'<div class="lab-step"><span class="lab-step-n">{number}</span>'
        f'<div><div class="lab-step-t">{html.escape(title)}</div>{hint_html}</div></div>',
        unsafe_allow_html=True,
    )


def fact(column, label, value):
    """One small labelled fact, e.g. "Dataset" / "BSD68"."""
    column.markdown(
        f'<div class="lab-fact">{html.escape(label)}</div>'
        f'<div class="lab-fact-v">{html.escape(str(value))}</div>',
        unsafe_allow_html=True,
    )


# --------------------------------------------------
# Home: a fixed 8 x 8 pixel drawing of the four stages
# --------------------------------------------------

def _drawing():
    """Clean, noisy, filtered and difference tiles. A fixed illustration only."""
    y, x = np.mgrid[0:8, 0:8]
    clean = 0.25 + 0.05 * x
    clean[2:6, 3:7] = 0.85                                     # a bright square: an "object" with edges
    noise = np.random.default_rng(3).normal(0, 0.18, (8, 8))   # fixed drawing, not experiment noise
    noisy = np.clip(clean + noise, 0, 1)
    padded = np.pad(noisy, 1, mode="edge")
    filtered = sum(padded[dy:dy + 8, dx:dx + 8] for dy in range(3) for dx in range(3)) / 9
    difference = np.clip(np.abs(filtered - clean) * 3, 0, 1)
    return clean, noisy, filtered, difference


def _tile(values, size=96):
    cell = size / values.shape[0]
    cells = "".join(
        f'<rect x="{col * cell:.1f}" y="{row * cell:.1f}" width="{cell + 0.4:.1f}" height="{cell + 0.4:.1f}" '
        f'fill="rgb({int(v * 255)},{int(v * 255)},{int(v * 255)})"/>'
        for (row, col), v in np.ndenumerate(values)
    )
    return f'<svg width="{size}" height="{size}" viewBox="0 0 {size} {size}" role="img">{cells}</svg>'


ARROW = ('<svg class="lab-arrow" width="22" height="14" viewBox="0 0 22 14" aria-hidden="true">'
         '<path d="M1 7h18M13 1l6 6-6 6" fill="none" stroke="currentColor" stroke-width="1.8"/></svg>')


def pipeline():
    clean, noisy, filtered, difference = _drawing()
    stages = [
        (clean, "Original", "The clean image"),
        (noisy, "Add noise", "Damage it on purpose"),
        (filtered, "Apply a filter", "Try to repair it"),
        (difference, "Compare", "Dark means close to the original"),
    ]
    parts = []
    for index, (values, title, text) in enumerate(stages):
        if index:
            parts.append(ARROW)
        parts.append(
            f'<div class="lab-stage">{_tile(values)}<b>{title}</b><span>{text}</span></div>'
        )
    st.markdown(f'<div class="lab-pipeline">{"".join(parts)}</div>', unsafe_allow_html=True)
