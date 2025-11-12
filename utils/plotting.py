from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from src.base import TrajectoryResult, compute_profiles


def render_matplotlib(
    trapezoidal_result: TrajectoryResult,
    scurve_result: TrajectoryResult,
    output_path: Path,
    *,
    resolution: int = 1000,
) -> None:
    """
    Render both trajectory results to a matplotlib figure and save to file.
    
    Args:
        trapezoidal_result: Result from trapezoidal trajectory generator
        scurve_result: Result from S-curve trajectory generator
        output_path: Path to save the plot
        resolution: Number of time points to sample
    """
    fig = _render_dual_profiles(trapezoidal_result, scurve_result, resolution=resolution)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def render_streamlit(
    trapezoidal_result: TrajectoryResult,
    scurve_result: TrajectoryResult,
    *,
    resolution: int = 1000,
) -> None:
    """
    Render both trajectory results in Streamlit.
    
    Args:
        trapezoidal_result: Result from trapezoidal trajectory generator
        scurve_result: Result from S-curve trajectory generator
        resolution: Number of time points to sample
    """
    fig = _render_dual_profiles(trapezoidal_result, scurve_result, resolution=resolution)
    st.pyplot(fig)


def _render_dual_profiles(
    trapezoidal_result: TrajectoryResult,
    scurve_result: TrajectoryResult,
    *,
    resolution: int = 1000,
) -> plt.Figure:
    """
    Create a figure with 4 subplots comparing trapezoidal and S-curve trajectories.
    
    Args:
        trapezoidal_result: Result from trapezoidal trajectory generator
        scurve_result: Result from S-curve trajectory generator
        resolution: Number of time points to sample
        
    Returns:
        Matplotlib figure with the comparison plots
    """
    # Compute profiles for both trajectories
    trap_time, trap_pos, trap_vel, trap_acc, trap_jerk = compute_profiles(
        trapezoidal_result, resolution=resolution
    )
    
    scurve_time, scurve_pos, scurve_vel, scurve_acc, scurve_jerk = compute_profiles(
        scurve_result, resolution=resolution
    )
    
    # Create figure with 4 subplots
    fig, axes = plt.subplots(4, 1, figsize=(12, 16), sharex=True)
    
    # Position plot
    axes[0].plot(trap_time, trap_pos, label="Trapezoidal", color="blue", linewidth=2)
    axes[0].plot(scurve_time, scurve_pos, label="S-Curve", color="orange", linewidth=2)
    axes[0].set_ylabel("Position (m)")
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend(loc="best")
    axes[0].set_title("Position Profile")
    
    # Velocity plot
    axes[1].plot(trap_time, trap_vel, label="Trapezoidal", color="blue", linewidth=2)
    axes[1].plot(scurve_time, scurve_vel, label="S-Curve", color="orange", linewidth=2)
    axes[1].axhline(
        trapezoidal_result.inputs.v_initial,
        linestyle="--",
        color="gray",
        alpha=0.5,
        label="Initial Speed",
    )
    axes[1].axhline(
        trapezoidal_result.inputs.v_final,
        linestyle="--",
        color="purple",
        alpha=0.5,
        label="Final Speed",
    )
    axes[1].axhline(
        trapezoidal_result.parameters.v_cruise,
        linestyle="--",
        color="green",
        alpha=0.5,
        label="Cruise Speed",
    )
    axes[1].set_ylabel("Velocity (m/s)")
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].legend(loc="best")
    axes[1].set_title("Velocity Profile")
    
    # Acceleration plot
    axes[2].plot(trap_time, trap_acc, label="Trapezoidal", color="blue", linewidth=2)
    axes[2].plot(scurve_time, scurve_acc, label="S-Curve", color="orange", linewidth=2)
    axes[2].axhline(
        trapezoidal_result.parameters.a_max,
        linestyle="--",
        color="green",
        alpha=0.5,
        label="Max Accel",
    )
    axes[2].axhline(
        trapezoidal_result.parameters.a_min,
        linestyle="--",
        color="red",
        alpha=0.5,
        label="Min Accel",
    )
    axes[2].axhline(0, linestyle="-", color="black", alpha=0.3, linewidth=0.5)
    axes[2].set_ylabel("Acceleration (m/s²)")
    axes[2].grid(True, linestyle="--", alpha=0.5)
    axes[2].legend(loc="best")
    axes[2].set_title("Acceleration Profile")
    
    # Jerk plot (only S-curve is meaningful)
    axes[3].plot(scurve_time, scurve_jerk, label="S-Curve", color="orange", linewidth=2)
    if scurve_result.parameters.j_max is not None:
        axes[3].axhline(
            scurve_result.parameters.j_max,
            linestyle="--",
            color="green",
            alpha=0.5,
            label="Max Jerk",
        )
    if scurve_result.parameters.j_min is not None:
        axes[3].axhline(
            scurve_result.parameters.j_min,
            linestyle="--",
            color="red",
            alpha=0.5,
            label="Min Jerk",
        )
    axes[3].axhline(0, linestyle="-", color="black", alpha=0.3, linewidth=0.5)
    axes[3].set_ylabel("Jerk (m/s³)")
    axes[3].set_xlabel("Time (s)")
    axes[3].grid(True, linestyle="--", alpha=0.5)
    axes[3].legend(loc="best")
    axes[3].set_title("Jerk Profile (S-Curve only)")
    
    # Ensure all axes have bottom labels enabled
    for axis in axes:
        axis.tick_params(labelbottom=True)
    
    fig.suptitle("Trajectory Comparison: Trapezoidal vs S-Curve", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    
    return fig
