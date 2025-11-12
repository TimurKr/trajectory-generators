from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import List

import streamlit as st

from utils.plotting import render_matplotlib, render_streamlit
from src.base import TrajectoryInputs, TrajectoryParameters, TrajectoryResult
from src.trapezoidal import TrapezoidalTrajectoryGenerator
from src.scurve import SCurveTrajectoryGenerator


DEFAULT_A_MAX = 4.0
DEFAULT_A_MIN = -3.0
DEFAULT_V_CRUISE = 100.0
DEFAULT_J_MAX = 1.0
DEFAULT_J_MIN = -1.0
DEFAULT_INITIAL_SPEED = 0.0
DEFAULT_FINAL_SPEED = 0.0
DEFAULT_INITIAL_ACCEL = 0.0
DEFAULT_FINAL_ACCEL = 0.0
DEFAULT_DISTANCE = 5_000.0

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def display_phase_tables(
    trapezoidal_result: TrajectoryResult, 
    scurve_result: TrajectoryResult
) -> None:
    """Display phase information for both trajectory types side by side."""
    st.subheader("Phase Durations Comparison")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Trapezoidal Trajectory**")
        trap_phase_names: List[str] = ["Acceleration", "Cruise", "Deceleration"]
        trap_durations = [f"{phase.duration:.3f}" for phase in trapezoidal_result.phases]
        trap_accelerations = [
            f"{phase.acceleration:.3f}" if phase.acceleration is not None else (
                f"v={phase.velocity:.1f}" if phase.velocity is not None else "0.000"
            )
            for phase in trapezoidal_result.phases
        ]
        trap_data = {
            "Phase": trap_phase_names + ["Total"],
            "Duration (s)": trap_durations + [f"{trapezoidal_result.total_time:.3f}"],
            "Accel/Vel": trap_accelerations + ["-"],
        }
        st.table(trap_data)
    
    with col2:
        st.markdown("**S-Curve Trajectory**")
        # Count actual phases (non-zero duration)
        scurve_phase_info = []
        for idx, phase in enumerate(scurve_result.phases):
            if phase.jerk is not None:
                scurve_phase_info.append(
                    (f"Jerk {len(scurve_phase_info)+1}", f"{phase.duration:.3f}", f"j={phase.jerk:.3f}")
                )
            elif phase.acceleration is not None:
                scurve_phase_info.append(
                    (f"Accel {len(scurve_phase_info)+1}", f"{phase.duration:.3f}", f"a={phase.acceleration:.3f}")
                )
            elif phase.velocity is not None:
                scurve_phase_info.append(
                    (f"Cruise", f"{phase.duration:.3f}", f"v={phase.velocity:.1f}")
                )
        
        if scurve_phase_info:
            scurve_data = {
                "Phase": [p[0] for p in scurve_phase_info] + ["Total"],
                "Duration (s)": [p[1] for p in scurve_phase_info] + [f"{scurve_result.total_time:.3f}"],
                "Jerk/Accel/Vel": [p[2] for p in scurve_phase_info] + ["-"],
            }
            st.table(scurve_data)
        else:
            st.write("No phases generated")


def save_plot(
    trapezoidal_result: TrajectoryResult,
    scurve_result: TrajectoryResult,
) -> Path:
    """Save the comparison plot to a file."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = RESULTS_DIR / f"trajectory_comparison_{timestamp}.png"
    render_matplotlib(trapezoidal_result, scurve_result, output_path)
    return output_path


def run_app() -> None:
    """Main Streamlit application."""
    st.title("Tangential Trajectory Generator")
    st.markdown("Compare **Trapezoidal** and **S-Curve** trajectory profiles side-by-side")

    # Sidebar: Parameters
    st.sidebar.header("Trajectory Parameters")
    
    st.sidebar.subheader("Acceleration Limits")
    a_max = st.sidebar.slider(
        "Max Acceleration (a_max)",
        min_value=0.1,
        max_value=10.0,
        value=DEFAULT_A_MAX,
        step=0.1,
    )
    a_min = st.sidebar.slider(
        "Min Acceleration (a_min)",
        min_value=-10.0,
        max_value=-0.1,
        value=DEFAULT_A_MIN,
        step=0.1,
    )
    
    st.sidebar.subheader("Jerk Limits (S-Curve)")
    j_max = st.sidebar.slider(
        "Max Jerk (j_max)",
        min_value=0.1,
        max_value=5.0,
        value=DEFAULT_J_MAX,
        step=0.1,
    )
    j_min = st.sidebar.slider(
        "Min Jerk (j_min)",
        min_value=-5.0,
        max_value=-0.1,
        value=DEFAULT_J_MIN,
        step=0.1,
    )
    
    st.sidebar.subheader("Velocity")
    v_cruise = st.sidebar.slider(
        "Cruise Speed (v_cruise)",
        min_value=0.5,
        max_value=150.0,
        value=DEFAULT_V_CRUISE,
        step=0.5,
    )

    # Sidebar: Inputs
    st.sidebar.header("Trajectory Inputs")
    
    st.sidebar.subheader("Initial State")
    v_i = st.sidebar.slider(
        "Initial Speed (v_i)",
        min_value=0.0,
        max_value=150.0,
        value=DEFAULT_INITIAL_SPEED,
        step=0.5,
    )
    a_i = st.sidebar.slider(
        "Initial Acceleration (a_i)",
        min_value=-10.0,
        max_value=10.0,
        value=DEFAULT_INITIAL_ACCEL,
        step=0.1,
    )
    
    st.sidebar.subheader("Final State")
    v_f = st.sidebar.slider(
        "Final Speed (v_f)",
        min_value=0.0,
        max_value=150.0,
        value=DEFAULT_FINAL_SPEED,
        step=0.5,
    )
    a_f = st.sidebar.slider(
        "Final Acceleration (a_f)",
        min_value=-10.0,
        max_value=10.0,
        value=DEFAULT_FINAL_ACCEL,
        step=0.1,
    )
    
    st.sidebar.subheader("Distance")
    delta_d = st.sidebar.slider(
        "Distance (Δd)",
        min_value=1.0,
        max_value=50_000.0,
        value=DEFAULT_DISTANCE,
        step=10.0,
    )

    # Create parameters for both generators
    trap_parameters = TrajectoryParameters(
        a_max=a_max,
        a_min=a_min,
        v_cruise=v_cruise,
    )
    
    scurve_parameters = TrajectoryParameters(
        a_max=a_max,
        a_min=a_min,
        v_cruise=v_cruise,
        j_max=j_max,
        j_min=j_min,
    )
    
    # Create inputs (trapezoidal doesn't use initial/final acceleration)
    trap_inputs = TrajectoryInputs(
        v_initial=v_i,
        v_final=v_f,
        delta_distance=delta_d,
    )
    
    scurve_inputs = TrajectoryInputs(
        v_initial=v_i,
        v_final=v_f,
        delta_distance=delta_d,
        a_initial=a_i,
        a_final=a_f,
    )

    # Generate trajectories
    try:
        trap_generator = TrapezoidalTrajectoryGenerator(trap_parameters)
        trapezoidal_result = trap_generator.generate_trajectory(trap_inputs)
    except ValueError as error:
        st.error(f"Trapezoidal trajectory error: {error}")
        return

    try:
        scurve_generator = SCurveTrajectoryGenerator(scurve_parameters)
        scurve_result = scurve_generator.generate_trajectory(scurve_inputs)
    except ValueError as error:
        st.error(f"S-Curve trajectory error: {error}")
        return

    # Display results
    display_phase_tables(trapezoidal_result, scurve_result)
    
    # Display comparison metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            "Trapezoidal Total Time",
            f"{trapezoidal_result.total_time:.3f} s",
        )
    with col2:
        st.metric(
            "S-Curve Total Time",
            f"{scurve_result.total_time:.3f} s",
        )
    with col3:
        time_diff = scurve_result.total_time - trapezoidal_result.total_time
        st.metric(
            "Time Difference",
            f"{abs(time_diff):.3f} s",
            delta=f"{time_diff:.3f} s" if time_diff != 0 else "Same",
        )
    
    # Render plots
    render_streamlit(trapezoidal_result, scurve_result)

    # Save button
    if st.button("Save comparison plot as PNG"):
        output_path = save_plot(trapezoidal_result, scurve_result)
        st.success(f"Plot saved to {output_path}")


if __name__ == "__main__":
    # Governing Equations
    # 
    # TRAPEZOIDAL (3 phases):
    # 1) Distance consistency: delta_d = d1 + d2 + d3
    #    d1 = 0.5 * (v_i + v_mid) * t1
    #    d2 = v_mid * t2               (cruise at v_mid which equals v_cruise when t2 > 0)
    #    d3 = 0.5 * (v_mid + v_f) * t3
    # 2) Velocity consistency: v_f = v_i + a1 * t1 + a3 * t3
    # 3) Peak velocity constraint: v_mid = v_i + a1 * t1 = v_cruise (only when cruise phase exists)
    #
    # If t2 < 0, set t2 = 0 and solve equations (1) and (2) only for t1 and t3.
    #
    # S-CURVE (7 phases):
    # 1) Jerk phase: j1 applied, acceleration increases
    # 2) Constant acceleration: a_max maintained
    # 3) Jerk phase: j3 applied, acceleration decreases to 0
    # 4) Cruise phase: constant velocity
    # 5) Jerk phase: j5 applied, acceleration becomes negative
    # 6) Constant deceleration: a_min maintained
    # 7) Jerk phase: j7 applied, acceleration returns to a_final
    #
    # Constraints:
    # - Final velocity: v_f = v_i + ∫∫ jerk profile
    # - Final acceleration: a_f = a_i + ∫ jerk profile
    # - Total distance: delta_d = ∫∫∫ jerk profile
    # - Acceleration limits: |a| ≤ a_max (or ≥ a_min)
    # - Velocity limits: v ≤ v_cruise
    #
    # If phases have negative times, iteratively constrain them to zero.
    
    run_app()
