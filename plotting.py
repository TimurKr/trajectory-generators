from __future__ import annotations

from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from trajectory import TrajectoryResult, compute_profiles


def render_matplotlib(
    result: TrajectoryResult,
    output_path: Path,
    *,
    resolution: int = 1000,
) -> None:
    time, position, velocity, acceleration = compute_profiles(result, resolution=resolution)
    _render_profiles(
        time,
        position,
        velocity,
        acceleration,
        v_initial=result.inputs.v_initial,
        v_final=result.inputs.v_final,
        v_cruise=result.parameters.v_cruise,
        final_position=position[-1],
        output_path=output_path,
    )


def render_streamlit(
    result: TrajectoryResult,
    *,
    resolution: int = 1000,
) -> None:
    time, position, velocity, acceleration = compute_profiles(result, resolution=resolution)
    fig = _render_profiles(
        time,
        position,
        velocity,
        acceleration,
        v_initial=result.inputs.v_initial,
        v_final=result.inputs.v_final,
        v_cruise=result.parameters.v_cruise,
        final_position=position[-1],
        output_path=None,
    )
    st.pyplot(fig)


def _render_profiles(
    time: np.ndarray,
    position: np.ndarray,
    velocity: np.ndarray,
    acceleration: np.ndarray,
    *,
    v_initial: float,
    v_final: float,
    v_cruise: float,
    final_position: float,
    output_path: Path | None,
) -> plt.Figure:
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    profiles: Tuple[Tuple[str, np.ndarray, plt.Axes], ...] = (
        ("Position", position, axes[0]),
        ("Velocity", velocity, axes[1]),
        ("Acceleration", acceleration, axes[2]),
    )

    for label, data, axis in profiles:
        axis.plot(time, data, label=label)
        axis.set_ylabel(label)
        axis.grid(True, linestyle="--", alpha=0.5)

    axes[0].axhline(final_position, linestyle="--", color="purple", label="Final Position")
    axes[0].legend()
    axes[1].axhline(v_initial, linestyle="--", color="blue", label="Initial Speed")
    axes[1].axhline(v_final, linestyle="--", color="orange", label="Final Speed")
    axes[1].axhline(v_cruise, linestyle="--", color="gray", label="Cruise Speed")
    axes[1].legend()

    for axis in axes:
        axis.tick_params(labelbottom=True)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Trajectory Profiles")
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    if output_path is not None:
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        return fig

    return fig

