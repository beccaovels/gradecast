"""Cached bridges between the Streamlit app and the plain-Python logic.

Streamlit reruns the page on every click, so anything expensive is cached here.
Courses are passed as a plain tuple of (name, weight, kind, score) so Streamlit
can compare inputs cheaply; the model is loaded once with ``cache_resource``.
"""

import json
from pathlib import Path

import numpy as np
import streamlit as st

from gradecast import explain, model, simulator
from gradecast.course import Course, Item
from gradecast.grading import LETTERS, letters_for

MODEL_PATH = Path("models/gb_quantile.joblib")
METRICS_PATH = Path("models/metrics.json")

# Hashable stand-ins for a course and a set of pins.
CourseKey = tuple[tuple[str, float, str, float | None], ...]
PinsKey = tuple[tuple[str, float], ...]


def course_key(course: Course) -> CourseKey:
    """Turn a course into a hashable tuple (the cache key)."""
    return tuple((i.name, i.weight, i.kind, i.score) for i in course.items)


def pins_key(pins: dict[str, float]) -> PinsKey:
    """Turn a pins dict into a hashable, order-independent tuple."""
    return tuple(sorted(pins.items()))


def _course_from_key(key: CourseKey) -> Course:
    """Rebuild a Course from its cache key."""
    return Course("course", [Item(name, weight, kind, score) for name, weight, kind, score in key])


@st.cache_resource
def get_models() -> model.Models | None:
    """Load the trained exam models once; None if the file is missing."""
    return model.load(MODEL_PATH) if MODEL_PATH.exists() else None


@st.cache_data
def get_metrics() -> dict | None:
    """Read the model's test results, or None if the file is missing."""
    return json.loads(METRICS_PATH.read_text()) if METRICS_PATH.exists() else None


@st.cache_data(show_spinner=False)
def totals_for(key: CourseKey, pins: PinsKey = (), shift: float = 0.0, n: int = 10_000, seed: int = 42) -> np.ndarray:
    """Simulated final totals (cached)."""
    return simulator.simulate(_course_from_key(key), get_models(), n, dict(pins), shift, seed)


@st.cache_data(show_spinner=False)
def probabilities_for(key: CourseKey, pins: PinsKey = (), shift: float = 0.0) -> dict[str, float]:
    """Probability of each letter grade (cached)."""
    letters = letters_for(totals_for(key, pins, shift))
    return {letter: float(np.mean(letters == letter)) for letter in LETTERS}


@st.cache_data(show_spinner=False)
def safe_zone_for(key: CourseKey, item_name: str, pins: PinsKey = (), shift: float = 0.0) -> float | None:
    """Lowest score on ``item_name`` that keeps a 50% chance of an A (cached)."""
    return simulator.safe_zone(
        _course_from_key(key), get_models(), item_name, pins=dict(pins), shift=shift
    )


@st.cache_data(show_spinner=False)
def forecasts_for(key: CourseKey) -> list[dict]:
    """Final-exam forecasts for every remaining exam (cached)."""
    return explain.exam_forecasts(_course_from_key(key), get_models())


@st.cache_data(show_spinner=False)
def paths_for(key: CourseKey, k: int = 25, seed: int = 42) -> dict:
    """Sample semester paths (cached)."""
    return explain.sample_paths(_course_from_key(key), get_models(), k=k, seed=seed)
