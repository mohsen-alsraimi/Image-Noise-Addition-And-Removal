"""
"Try an image": a quick, guided test on one image.

It starts exactly the same single-image job as "Run an experiment"
(python main.py --job job.json), so the result is saved like any other run.
Only the controls are simpler: one settings choice for all filters, and
fine-tuning hidden until it is wanted. This file contains no image processing.
"""

import streamlit as st

import app_helpers as helpers
import theme
from explanations import (
    DATASET_NOTES,
    DEGRADATION_TEXT,
    FILTER_GOOD_FOR,
    GLOSSARY,
    NOISE_TEXT,
)
from results_view import show_result_brief, show_result_picture
from run_metadata import DEGRADATION_PARAMETER_MEANING, NOISE_PARAMETER_MEANING
from run_view import (
    DEFAULT_ONLY_TEXT,
    image_size,
    keep_valid_choice,
    preset_key,
    preset_options,
    show_filter_settings,
    start,
)

PREFIX = "try::"
NOISE = "Noise"
DAMAGE = "Image damage"
LEVELS = ["Low", "Medium", "High"]


def new_test():
    st.session_state.pop("try_result", None)


def show_try_view(is_running):
    import main as backend

    st.header("Try an image")
    st.write("Pick one image, damage it, and see which filter repairs it best. Each test takes a few seconds.")

    controls, preview = st.columns([5, 6], gap="large")

    with controls:
        # ---- 1. Image
        with st.container(border=True):
            theme.step(1, "Pick an image")
            dataset = st.selectbox("Dataset", backend.JOB_DATASETS, key="try_dataset",
                                   persist_state="page", help=GLOSSARY["Dataset"])
            paths = backend.find_image_paths(backend.PROJECT_DIR / "dataset" / dataset)
            names = [path.name for path in paths]
            keep_valid_choice("try_image", names)
            image = st.selectbox("Image", names, key="try_image", persist_state="page",
                                 help=f"{dataset} has {len(names)} images.")
            if dataset == "original_png":
                st.caption(DATASET_NOTES["original_png"][1])

        # ---- 2. Damage
        with st.container(border=True):
            theme.step(2, "Add noise or damage")
            kind = st.radio("Type of damage", [NOISE, DAMAGE], horizontal=True, key="try_kind",
                            persist_state="page",
                            help=f"{GLOSSARY['Noise']} {GLOSSARY['Image damage']}")
            degradation = kind == DAMAGE
            table = backend.DEGRADATION_LEVELS if degradation else backend.NOISE_LEVELS
            texts = DEGRADATION_TEXT if degradation else NOISE_TEXT
            meanings = DEGRADATION_PARAMETER_MEANING if degradation else NOISE_PARAMETER_MEANING

            noise_type = st.selectbox(
                "Damage type" if degradation else "Noise type", list(table),
                key="try_degradation" if degradation else "try_noise", persist_state="page",
            )
            level = st.segmented_control(
                "Strength", LEVELS, default="Medium", required=True, key="try_level",
                persist_state="page", help=GLOSSARY["Strength"],
            )
            value = table[noise_type][level]
            st.caption(f"{texts[noise_type]} {level} = {value} ({meanings[noise_type]}).")

        # ---- 3. Filters
        with st.container(border=True):
            theme.step(3, "Pick filters", GLOSSARY["Filter"])
            filter_names = list(backend.filters)
            chosen = st.pills(
                "Filters", filter_names, selection_mode="multi", default=filter_names,
                format_func=helpers.short_name, key="try_filters", persist_state="page",
                label_visibility="collapsed",
            ) or []
            chosen = [name for name in filter_names if name in chosen]
            if chosen:
                st.caption("  \n".join(
                    f"**{helpers.short_name(name)}:** {FILTER_GOOD_FOR[helpers.short_name(name)]}"
                    for name in chosen
                ))
            else:
                st.caption("Pick at least one filter.")

        # ---- 4. Settings
        with st.container(border=True):
            theme.step(4, "Filter settings")
            options = preset_options(degradation)
            keep_valid_choice("try_preset", options)
            preset = st.radio(
                "Start from", options, format_func=helpers.preset_label, key="try_preset",
                persist_state="page", help=GLOSSARY["Preset"],
            )
            if degradation:
                st.caption(DEFAULT_ONLY_TEXT)

            settings = {}
            with st.expander("Change values by hand (optional)"):
                if not chosen:
                    st.caption("Pick a filter first.")
                for name in chosen:
                    st.markdown(f"**{helpers.short_name(name)}**")
                    st.session_state[preset_key(name, PREFIX)] = preset
                    settings[name] = show_filter_settings(
                        name, backend.filters[name], noise_type, level, value,
                        degradation, prefix=PREFIX, show_preset=False,
                    )

        clicked = st.button(
            "Run this test", type="primary", key="try_run", icon=":material/play_arrow:",
            disabled=is_running or not chosen, width="stretch",
        )
        if is_running:
            st.caption("An experiment is running. Its progress is shown at the top of the page.")

    with preview:
        result = st.session_state.get("try_result")
        if result:
            st.markdown("#### Result of your test")
            show_result_brief(result)
            st.button("Start a new test", key="try_new", on_click=new_test, icon=":material/restart_alt:")
        else:
            path = paths[names.index(image)]
            width, height = image_size(path)
            st.markdown("#### Your image")
            st.image(str(path), width="stretch")
            st.caption(f"{image} from {dataset}, {width} × {height} pixels, shown before any damage.")
            st.markdown(
                f"When you run the test, **{level.lower()} {noise_type}** is added to this image. "
                f"Then {len(chosen)} filter{'s' if len(chosen) != 1 else ''} each try to repair the same "
                "damaged copy, and every result is compared with the original."
            )

    # The comparison picture needs the full width to be readable
    if st.session_state.get("try_result"):
        show_result_picture(st.session_state["try_result"])

    if clicked:
        job = {
            "type": "single_image",
            "dataset": dataset,
            "image": image,
            "noise_type": noise_type,
            "noise_level": level,
            "filters": settings,
        }
        if degradation:
            job["experiment"] = "degradation"
        start(job, try_page=True)
