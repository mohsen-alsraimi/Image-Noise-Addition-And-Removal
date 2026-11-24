"""
Image Noise Lab: learn how filters repair noisy images, run experiments,
and explore the saved results.

Launch from the "Denoising ad" folder:

    .venv\\Scripts\\python.exe -m streamlit run frontend\\app.py

The pages only read saved files or start main.py in a separate process
(python main.py --job job.json), so this frontend contains no image processing.
"""

import sys
from pathlib import Path

# Do not create Python cache files beside the project code.
sys.dont_write_bytecode = True

# The scientific code (main.py, run_metadata.py, ...) is one folder up.
PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import streamlit as st

import theme
from home_view import show_home_view, show_learn_view
from results_view import show_results_view
from run_view import show_job_status, show_run_view
from try_view import show_try_view

PAGES = {
    "home": ":material/home: Home",
    "try": ":material/science: Try an image",
    "run": ":material/lab_profile: Run an experiment",
    "results": ":material/insights: Explore results",
    "learn": ":material/school: Learn",
}

st.set_page_config(page_title="Image Noise Lab", page_icon=":material/blur_on:", layout="wide")
theme.apply_style()

# Open a finished or chosen experiment before the dropdowns are created.
if "open_result" in st.session_state:
    finished = st.session_state.pop("open_result")
    st.session_state["view"] = "results"
    st.session_state["saved_run"] = finished["run"]
    st.session_state["noise_type"] = finished["noise_type"]
    st.session_state["noise_level"] = finished["noise_level"]

# A finished "Try an image" test returns to that page.
if "open_view" in st.session_state:
    st.session_state["view"] = st.session_state.pop("open_view")

theme.brand()
# Set through session state (not "default"), because other pages switch views.
st.session_state.setdefault("view", "home")
view = st.segmented_control(
    "Page",
    list(PAGES),
    format_func=PAGES.get,
    required=True,
    key="view",
    label_visibility="collapsed",
)
is_running = show_job_status()

if view == "try":
    show_try_view(is_running)
elif view == "run":
    show_run_view(is_running)
elif view == "results":
    show_results_view()
elif view == "learn":
    show_learn_view()
else:
    show_home_view()
