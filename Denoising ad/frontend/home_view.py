"""The Home page and the "Learn the basics" page. Text only; no results are calculated."""

import streamlit as st

import app_helpers as helpers
import theme
from explanations import (
    DEGRADATION_TEXT,
    EDGE_CAUTION,
    FILTER_GOOD_FOR,
    FILTER_TEXT,
    GLOSSARY,
    NOISE_TEXT,
    PSNR_TEXT,
    SSIM_TEXT,
)


def go_to(view):
    """Button callback: switch to another page."""
    st.session_state["view"] = view


def show_home_view():
    st.title("See how filters repair a damaged image")
    st.write(
        "Image Noise Lab damages test images on purpose, cleans them with six classic "
        "filters, and measures how close each result is to the original."
    )
    theme.pipeline()

    actions = [
        ("try", ":material/science:", "Try an image",
         "Quick test on one image. Choose the noise and the filters, and see the result in seconds.",
         "primary"),
        ("run", ":material/lab_profile:", "Run an experiment",
         "Compare filters on a whole dataset, under one or all conditions. Use these runs for your report.",
         "secondary"),
        ("results", ":material/insights:", "Explore results",
         "Open saved experiments: summaries, scores, pictures and the exact settings used.",
         "secondary"),
    ]
    for column, (view, icon, title, text, kind) in zip(st.columns(3), actions):
        with column.container(border=True, height="stretch"):
            st.markdown(f"#### {title}")
            st.write(text)
            st.button(title, key=f"home_{view}", type=kind, icon=icon, on_click=go_to, args=(view,))

    runs = helpers.find_saved_runs()
    if runs:
        newest = next(iter(runs))
        st.caption(f"Latest saved experiment: {helpers.describe_run(newest, runs[newest])}.")

    st.button("New to image denoising? Learn the basics first", key="home_learn", type="tertiary",
              icon=":material/school:", on_click=go_to, args=("learn",))


def show_learn_view():
    st.header("Learn the basics")
    st.write("Short explanations of the words used in this lab. You can come back here at any time.")

    st.markdown("#### Key words")
    terms = list(GLOSSARY.items())
    half = (len(terms) + 1) // 2
    for column, part in zip(st.columns(2, gap="large"), (terms[:half], terms[half:])):
        column.markdown("\n".join(f"- **{term}:** {text}" for term, text in part))

    st.markdown("#### Noise and image damage")
    left, right = st.columns(2, gap="large")
    left.markdown("**Noise** (random)\n\n" + "\n".join(f"- **{name}:** {text}" for name, text in NOISE_TEXT.items()))
    right.markdown("**Image damage** (not random noise)\n\n"
                   + "\n".join(f"- **{name}:** {text}" for name, text in DEGRADATION_TEXT.items()))

    st.markdown("#### The six filters")
    columns = st.columns(3, gap="medium")
    for index, (name, text) in enumerate(FILTER_TEXT.items()):
        with columns[index % 3].container(border=True, height="stretch"):
            st.markdown(f"**{name}**")
            st.write(text)
            st.caption(FILTER_GOOD_FOR[name])

    st.markdown("#### How a result is scored")
    psnr, ssim, edges = st.columns(3, gap="large")
    psnr.markdown(PSNR_TEXT)
    ssim.markdown(SSIM_TEXT)
    edges.markdown(
        "**Edge F1** compares object outlines in the result with the original, from 0 to 1. "
        "It is shown under Details, because it can mislead: " + EDGE_CAUTION[len("Careful: "):]
    )
    st.info(
        "A filter only **helped** if it scores higher than the noisy or damaged image itself. "
        "Being the best of the filters is not enough: sometimes every filter makes the image worse.",
        icon=":material/lightbulb:",
    )
