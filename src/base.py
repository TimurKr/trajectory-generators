from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class TrajectoryParameters:
    """Parameters defining physical constraints for trajectory generation."""
    a_max: float
    a_min: float
    v_cruise: float
    j_max: Optional[float] = None
    j_min: Optional[float] = None


@dataclass(frozen=True)
class TrajectoryInputs:
    """Initial conditions and target state for trajectory generation."""
    v_initial: float
    v_final: float
    delta_distance: float
    a_initial: Optional[float] = None
    a_final: Optional[float] = None


@dataclass(frozen=True)
class Phase:
    """
    Represents a single phase of a trajectory.
    
    At most one of acceleration, velocity, or jerk should be set:
    - If jerk is set: constant jerk phase (acceleration changes linearly)
    - If acceleration is set: constant acceleration phase (velocity changes linearly)
    - If velocity is set: constant velocity phase (cruise)
    """
    duration: float
    acceleration: Optional[float] = None
    velocity: Optional[float] = None
    jerk: Optional[float] = None


@dataclass(frozen=True)
class TrajectoryResult:
    """Complete trajectory result including parameters, inputs, and computed phases."""
    parameters: TrajectoryParameters
    inputs: TrajectoryInputs
    phases: List[Phase]

    @property
    def total_time(self) -> float:
        """
        Calculate the total duration of the trajectory.
        
        Returns:
            Sum of all phase durations.
        """
        return sum(phase.duration for phase in self.phases)


def compute_profiles(
    result: TrajectoryResult, 
    resolution: int = 1000
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute position, velocity, acceleration, and jerk profiles over time for a trajectory.
    
    Generates arrays of time-discretized values by integrating through each phase:
    - Jerk phases: constant jerk applied (acceleration and velocity change)
    - Acceleration phases: constant acceleration applied (velocity changes linearly)
    - Cruise phases: constant velocity (no acceleration or jerk)
    
    Args:
        result: TrajectoryResult containing the phases and parameters.
        resolution: Number of time points to sample (default: 1000).
        
    Returns:
        Tuple of (time, position, velocity, acceleration, jerk) arrays, each of length resolution.
        
    Raises:
        ValueError: If total trajectory duration is non-positive.
    """
    durations = np.array([phase.duration for phase in result.phases], dtype=float)
    total_time = durations.sum()
    if total_time <= 0.0:
        raise ValueError("Total trajectory duration must be positive.")

    timeline = np.linspace(0.0, total_time, num=resolution)
    position = np.zeros_like(timeline)
    velocity = np.zeros_like(timeline)
    acceleration = np.zeros_like(timeline)
    jerk = np.zeros_like(timeline)

    phase_start_times = np.cumsum(np.insert(durations, 0, 0.0))
    
    # Initialize with inputs
    current_velocity = result.inputs.v_initial
    current_position = 0.0
    current_acceleration = result.inputs.a_initial if result.inputs.a_initial is not None else 0.0

    for idx, phase in enumerate(result.phases):
        start_time = phase_start_times[idx]
        end_time = phase_start_times[idx + 1]
        mask = (timeline >= start_time) & (timeline <= end_time)
        
        if not mask.any():
            continue
            
        local_time = timeline[mask] - start_time

        if phase.jerk is not None:
            # Constant jerk phase
            j = phase.jerk
            acceleration[mask] = current_acceleration + j * local_time
            velocity[mask] = current_velocity + current_acceleration * local_time + 0.5 * j * local_time**2
            position[mask] = (
                current_position 
                + current_velocity * local_time 
                + 0.5 * current_acceleration * local_time**2
                + (1.0 / 6.0) * j * local_time**3
            )
            jerk[mask] = j
            
            # Update state at end of phase
            if phase.duration > 0:
                dt = phase.duration
                current_position += current_velocity * dt + 0.5 * current_acceleration * dt**2 + (1.0/6.0) * j * dt**3
                current_velocity += current_acceleration * dt + 0.5 * j * dt**2
                current_acceleration += j * dt
            
        elif phase.acceleration is not None:
            # Constant acceleration phase
            a = phase.acceleration
            acceleration[mask] = a
            velocity[mask] = current_velocity + a * local_time
            position[mask] = current_position + current_velocity * local_time + 0.5 * a * local_time**2
            jerk[mask] = 0.0
            
            # Update state at end of phase
            if phase.duration > 0:
                dt = phase.duration
                current_position += current_velocity * dt + 0.5 * a * dt**2
                current_velocity += a * dt
                current_acceleration = a
                
        elif phase.velocity is not None:
            # Constant velocity phase (cruise)
            velocity[mask] = phase.velocity
            position[mask] = current_position + phase.velocity * local_time
            acceleration[mask] = 0.0
            jerk[mask] = 0.0
            
            # Update state at end of phase
            if phase.duration > 0:
                dt = phase.duration
                current_position += phase.velocity * dt
                current_velocity = phase.velocity
                current_acceleration = 0.0
                
        else:
            # Zero duration or unspecified phase - maintain current state
            velocity[mask] = current_velocity
            position[mask] = current_position
            acceleration[mask] = current_acceleration
            jerk[mask] = 0.0

    return timeline, position, velocity, acceleration, jerk

