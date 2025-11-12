from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import List

import streamlit as st

from plotting import render_matplotlib, render_streamlit
from trajectory import (
    TrajectoryGenerator,
    TrajectoryInputs,
    TrajectoryParameters,
    TrajectoryResult,
)


DEFAULT_A_MAX = 4.0
DEFAULT_A_MIN = -3.0
DEFAULT_V_CRUISE = 100.0
DEFAULT_INITIAL_SPEED = 0.0
DEFAULT_FINAL_SPEED = 0.0
DEFAULT_DISTANCE = 5_000.0

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def display_phase_table(result: TrajectoryResult) -> None:
    st.subheader("Phase Durations")
    phase_names: List[str] = ["Acceleration", "Cruise", "Deceleration"]
    durations = [f"{phase.duration:.3f}" for phase in result.phases]
    accelerations = [
        f"{phase.acceleration:.3f}" if phase.acceleration is not None else "0.000"
        for phase in result.phases
    ]
    data = {
        "Phase": phase_names + ["Total"],
        "Duration (s)": durations + [f"{result.total_time:.3f}"],
        "Acceleration (m/s²)": accelerations + ["-"],
    }
    st.table(data)


def save_plot(result: TrajectoryResult) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = RESULTS_DIR / f"trajectory_{timestamp}.png"
    render_matplotlib(result, output_path)
    return output_path


def run_app() -> None:
    st.title("Tangential Trajectory Generator")

    st.sidebar.header("Parameters")
    a_max = st.sidebar.slider(
        "Max Acceleration (a_max)", min_value=0.1, max_value=10.0, value=DEFAULT_A_MAX, step=0.1
    )
    a_min = st.sidebar.slider(
        "Min Acceleration (a_min)", min_value=-10.0, max_value=-0.1, value=DEFAULT_A_MIN, step=0.1
    )
    v_cruise = st.sidebar.slider(
        "Cruise Speed (v_cruise)", min_value=0.5, max_value=150.0, value=DEFAULT_V_CRUISE, step=0.5
    )

    st.sidebar.header("Inputs")
    v_i = st.sidebar.slider(
        "Initial Speed (v_i)", min_value=0.0, max_value=150.0, value=DEFAULT_INITIAL_SPEED, step=0.5
    )
    v_f = st.sidebar.slider(
        "Target Speed (v_f)", min_value=0.0, max_value=150.0, value=DEFAULT_FINAL_SPEED, step=0.5
    )
    delta_d = st.sidebar.slider(
        "Distance (Δd)", min_value=1.0, max_value=50_000.0, value=DEFAULT_DISTANCE, step=10.0
    )

    parameters = TrajectoryParameters(a_max=a_max, a_min=a_min, v_cruise=v_cruise)
    generator = TrajectoryGenerator(parameters)
    inputs = TrajectoryInputs(v_initial=v_i, v_final=v_f, delta_distance=delta_d)

    try:
        result = generator.generate_trajectory(inputs)
    except ValueError as error:
        st.error(str(error))
        return

    display_phase_table(result)
    render_streamlit(result)

    if st.button("Save plot as PNG"):
        output_path = save_plot(result)
        st.success(f"Plot saved to {output_path}")


if __name__ == "__main__":
    # Governing Equations (documented for verification):
    # 1) Distance consistency: delta_d = d1 + d2 + d3
    #    d1 = 0.5 * (v_i + v_mid) * t1
    #    d2 = v_mid * t2               (cruise at v_mid which equals v_cruise when t2 > 0)
    #    d3 = 0.5 * (v_mid + v_f) * t3
    # 2) Velocity consistency: v_f = v_i + a1 * t1 + a3 * t3
    # 3) Peak velocity constraint: v_mid = v_i + a1 * t1 = v_cruise (only when cruise phase exists)
    #
    # If t2 < 0, set t2 = 0 and solve equations (1) and (2) only for t1 and t3.
    run_app()
