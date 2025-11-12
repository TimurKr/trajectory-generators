from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
from sympy import Eq, Symbol, symbols, solve


@dataclass(frozen=True)
class TrajectoryParameters:
    a_max: float
    a_min: float
    v_cruise: float


@dataclass(frozen=True)
class TrajectoryInputs:
    v_initial: float
    v_final: float
    delta_distance: float


@dataclass(frozen=True)
class Phase:
    duration: float
    acceleration: Optional[float]


@dataclass(frozen=True)
class TrajectoryResult:
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


class TrajectoryGenerator:
    def __init__(self, parameters: TrajectoryParameters) -> None:
        """
        Initialize the trajectory generator with physical constraints.
        
        Args:
            parameters: Trajectory parameters containing a_max, a_min, and v_cruise.
            
        Raises:
            ValueError: If any parameter violates physical constraints.
        """
        if parameters.a_max <= 0.0:
            raise ValueError("a_max must be positive.")
        if parameters.a_min >= 0.0:
            raise ValueError("a_min must be negative.")
        if parameters.v_cruise <= 0.0:
            raise ValueError("v_cruise must be positive.")

        self.parameters = parameters

    def generate_trajectory(self, inputs: TrajectoryInputs) -> TrajectoryResult:
        """
        Generate a three-phase trajectory (acceleration, cruise, deceleration) that satisfies
        the given initial/final velocities and total distance constraints.
        
        The trajectory consists of:
        - Phase 1: Acceleration from v_initial to v_cruise (or directly to v_final if no cruise)
        - Phase 2: Cruise at constant v_cruise (may be zero duration)
        - Phase 3: Deceleration from v_cruise to v_final
        
        Args:
            inputs: Trajectory inputs containing v_initial, v_final, and delta_distance.
            
        Returns:
            TrajectoryResult containing the computed phases and all input/parameter information.
            
        Raises:
            ValueError: If delta_distance is non-positive or no valid trajectory can be found.
        """
        if inputs.delta_distance <= 0.0:
            raise ValueError("delta_distance must be positive.")

        params = self.parameters
        v_i = inputs.v_initial
        v_f = inputs.v_final
        delta_d = inputs.delta_distance

        a1 = params.a_min if v_i > params.v_cruise else params.a_max
        a3 = params.a_max if v_f > params.v_cruise else params.a_min

        t1, t2, t3 = self._solve_phases(a1, a3, v_i, v_f, delta_d)

        v_mid_check = v_i + a1 * t1
        v_end_check = v_mid_check + a3 * t3
        d1_check = 0.5 * (v_i + v_mid_check) * t1
        d2_check = v_mid_check * t2 if t2 > 0 else 0.0
        d3_check = 0.5 * (v_mid_check + v_end_check) * t3
        total_distance_check = d1_check + d2_check + d3_check

        if abs(v_end_check - v_f) > 1e-6 or abs(total_distance_check - delta_d) > 1e-6:
            raise ValueError("No valid trajectory satisfies the requested constraints.")

        phases: List[Phase] = [
            Phase(duration=t1, acceleration=a1),
            Phase(duration=t2, acceleration=None),
            Phase(duration=t3, acceleration=a3),
        ]
        return TrajectoryResult(parameters=params, inputs=inputs, phases=phases)

    def _solve_phases(
        self,
        a1: float,
        a3: float,
        v_i: float,
        v_f: float,
        delta_d: float,
    ) -> Tuple[float, float, float]:
        """
        Solve for phase durations (t1, t2, t3) using symbolic equation solving.
        
        Attempts to find a solution in two stages:
        1. Three-phase: Solve with cruise phase constraint (v_mid = v_cruise)
        2. Two-phase: If no valid solution, add constraint t2 = 0 and solve again
        
        The base equations are:
        - Total distance: d1 + d2 + d3 = delta_d
        - Final velocity: v_f = v_i + a1*t1 + a3*t3
        
        Args:
            a1: Acceleration for phase 1 (positive for acceleration, negative for deceleration).
            a3: Acceleration for phase 3 (positive for acceleration, negative for deceleration).
            v_i: Initial velocity.
            v_f: Final velocity.
            delta_d: Total distance to travel.
            
        Returns:
            Tuple of (t1, t2, t3) representing durations of each phase.
            
        Raises:
            ValueError: If no valid solution is found for either three-phase or two-phase system.
        """
        params = self.parameters
        t1, t2, t3 = symbols("t1 t2 t3", real=True)
        
        # Define the fundamental equations (same for both attempts)
        v_mid = v_i + a1 * t1
        v_end = v_mid + a3 * t3
        d1 = 0.5 * (v_i + v_mid) * t1
        d2 = v_mid * t2
        d3 = 0.5 * (v_mid + v_end) * t3
        
        # Base equations (same for both attempts)
        base_equations = (
            Eq(d1 + d2 + d3, delta_d),  # Total distance
            Eq(v_end, v_f),              # Final velocity
        )
        
        # Try each constraint in sequence
        constraints = [
            (Eq(v_mid, params.v_cruise), "three-phase"),  # First: cruise velocity constraint
            (Eq(t2, 0.0), "two-phase"),                    # Second: no cruise phase
        ]
        
        for constraint, context in constraints:
            equations = base_equations + (constraint,)
            raw_solutions = solve(equations, (t1, t2, t3), dict=True)
            if not raw_solutions:
                continue

            solution = self._select_solution(
                raw_solutions,
                (t1, t2, t3),
                require_non_negative=True,
            )
            return solution

        raise ValueError("No real solution found for the trajectory constraints.")

    @staticmethod
    def _select_solution(
        solutions: Sequence[dict],
        symbol_order: Sequence[Symbol],
        *,
        require_non_negative: bool,
        tolerance: float = 1e-9,
    ) -> Tuple[float, ...]:
        """
        Select the valid real solution with the lowest sum of all time components.
        
        Filters solutions to find valid candidates that:
        - Have all real components (no imaginary parts)
        - If require_non_negative is True, have all non-negative components
        
        Among all valid candidates, returns the one with the minimum sum of all time values.
        
        Args:
            solutions: List of solution dictionaries from sympy.solve.
            symbol_order: Order of symbols to extract values for.
            require_non_negative: If True, only accept solutions with all non-negative values.
            tolerance: Tolerance for checking if a value is real or non-negative.
            
        Returns:
            Tuple of numeric values corresponding to the symbol order, with minimum total time.
            
        Raises:
            ValueError: If no valid solution is found in the candidates.
        """
        valid_candidates: List[Tuple[float, ...]] = []
        
        for candidate in solutions:
            numeric_values: List[float] = []
            all_real = True
            for symbol in symbol_order:
                value = candidate[symbol]
                evaluated = complex(value.evalf())
                if abs(evaluated.imag) > tolerance:
                    all_real = False
                    break
                numeric_values.append(float(evaluated.real))

            if not all_real:
                continue

            if require_non_negative:
                min_component = min(numeric_values)
                if min_component < -tolerance:
                    continue
                numeric_values = [max(0.0, value) for value in numeric_values]

            valid_candidates.append(tuple(numeric_values))

        if not valid_candidates:
            raise ValueError("No real solution found for the trajectory constraints.")

        # Return the candidate with the minimum sum of all time components
        return min(valid_candidates, key=sum)


def compute_profiles(result: TrajectoryResult, resolution: int = 1000) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute position, velocity, and acceleration profiles over time for a trajectory.
    
    Generates arrays of time-discretized values by integrating through each phase:
    - Acceleration phases: constant acceleration applied
    - Cruise phases: constant velocity (v_cruise)
    
    Args:
        result: TrajectoryResult containing the phases and parameters.
        resolution: Number of time points to sample (default: 1000).
        
    Returns:
        Tuple of (time, position, velocity, acceleration) arrays, each of length resolution.
        
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

    phase_start_times = np.cumsum(np.insert(durations, 0, 0.0))
    current_velocity = result.inputs.v_initial
    current_position = 0.0

    for idx, phase in enumerate(result.phases):
        start_time = phase_start_times[idx]
        end_time = phase_start_times[idx + 1]
        mask = (timeline >= start_time) & (timeline <= end_time)
        local_time = timeline[mask] - start_time

        if phase.acceleration is None:
            if not mask.any():
                continue
            if phase.duration <= 0.0:
                velocity[mask] = current_velocity
                position[mask] = current_position
                acceleration[mask] = 0.0
                continue
            velocity[mask] = result.parameters.v_cruise
            position[mask] = current_position + result.parameters.v_cruise * local_time
            acceleration[mask] = 0.0
            current_velocity = result.parameters.v_cruise
        else:
            a = phase.acceleration
            velocity[mask] = current_velocity + a * local_time
            position[mask] = current_position + current_velocity * local_time + 0.5 * a * local_time**2
            acceleration[mask] = a

        current_position = position[mask][-1] if mask.any() else current_position
        current_velocity = velocity[mask][-1] if mask.any() else current_velocity

    return timeline, position, velocity, acceleration

