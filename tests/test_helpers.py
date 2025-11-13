"""Shared test helpers for trajectory generator tests."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Tuple

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.base import TrajectoryResult, compute_profiles


def validate_trajectory(
    result: TrajectoryResult,
    expected_v_final: float,
    expected_distance: float,
    expected_a_final: float = None,
    tolerance: float = 1e-3,
) -> Tuple[bool, str]:
    """
    Validate that a trajectory satisfies its constraints.
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    # Compute the actual profiles
    timeline, position, velocity, acceleration, jerk = compute_profiles(result, resolution=10000)
    
    # Check final velocity
    actual_v_final = velocity[-1]
    if abs(actual_v_final - expected_v_final) > tolerance:
        return False, f"Final velocity mismatch: expected {expected_v_final:.3f}, got {actual_v_final:.3f}"
    
    # Check final position (distance)
    actual_distance = position[-1]
    if abs(actual_distance - expected_distance) > tolerance:
        return False, f"Distance mismatch: expected {expected_distance:.3f}, got {actual_distance:.3f}"
    
    # Check final acceleration for S-curve (only if expected_a_final is specified)
    if expected_a_final is not None:
        actual_a_final = acceleration[-1]
        if abs(actual_a_final - expected_a_final) > tolerance:
            return False, f"Final acceleration mismatch: expected {expected_a_final:.3f}, got {actual_a_final:.3f}"
    
    # Check that acceleration stays within bounds
    if result.parameters.a_max is not None:
        max_accel = acceleration.max()
        if max_accel > result.parameters.a_max + tolerance:
            return False, f"Acceleration exceeds a_max: {max_accel:.3f} > {result.parameters.a_max:.3f}"
    
    if result.parameters.a_min is not None:
        min_accel = acceleration.min()
        if min_accel < result.parameters.a_min - tolerance:
            return False, f"Acceleration below a_min: {min_accel:.3f} < {result.parameters.a_min:.3f}"
    
    # Check that velocity stays within bounds (allowing for initial/final velocities to exceed cruise)
    # The trajectory may start or end above v_cruise, which is valid
    v_initial = result.inputs.v_initial
    v_final = result.inputs.v_final
    max_allowed_velocity = max(result.parameters.v_cruise, v_initial, v_final) + tolerance * 10
    
    max_velocity = velocity.max()
    if max_velocity > max_allowed_velocity:
        return False, f"Velocity exceeds maximum allowed: {max_velocity:.3f} > {max_allowed_velocity:.3f}"
    
    return True, "OK"

