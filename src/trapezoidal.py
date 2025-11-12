from __future__ import annotations

import logging
from typing import List, Sequence, Tuple

from sympy import Eq, Symbol, solve, symbols

from .base import (
    Phase,
    TrajectoryInputs,
    TrajectoryParameters,
    TrajectoryResult,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class TrapezoidalTrajectoryGenerator:
    """
    Generates trapezoidal velocity profile trajectories.
    
    A trapezoidal trajectory consists of three phases:
    1. Acceleration phase: constant acceleration from v_initial towards v_cruise
    2. Cruise phase: constant velocity at v_cruise (may be zero duration)
    3. Deceleration phase: constant acceleration from v_cruise to v_final
    """
    
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

        # Determine acceleration directions based on initial/final velocities
        a1 = params.a_min if v_i > params.v_cruise else params.a_max
        a3 = params.a_max if v_f > params.v_cruise else params.a_min

        logger.info("="*70)
        logger.info(f"Solving trapezoidal: v_i={v_i:.1f}, v_f={v_f:.1f}, Δd={delta_d:.1f}")
        logger.info(f"  a1={a1:.2f}, a3={a3:.2f}, v_cruise={params.v_cruise:.1f}")

        t1, t2, t3 = self._solve_phases(a1, a3, v_i, v_f, delta_d)

        # Validate solution
        logger.info(f"\nValidating solution:")
        v_mid_check = v_i + a1 * t1
        v_end_check = v_mid_check + a3 * t3
        d1_check = 0.5 * (v_i + v_mid_check) * t1
        d2_check = v_mid_check * t2 if t2 > 0 else 0.0
        d3_check = 0.5 * (v_mid_check + v_end_check) * t3
        total_distance_check = d1_check + d2_check + d3_check

        logger.info(f"  v_mid={v_mid_check:.6f}, v_end={v_end_check:.6f}, v_f={v_f:.6f}")
        logger.info(f"  d1={d1_check:.6f}, d2={d2_check:.6f}, d3={d3_check:.6f}")
        logger.info(f"  total_distance={total_distance_check:.6f}, target={delta_d:.6f}")
        logger.info(f"  v_end error: {abs(v_end_check - v_f):.9f}")
        logger.info(f"  distance error: {abs(total_distance_check - delta_d):.9f}")

        if abs(v_end_check - v_f) > 1e-6 or abs(total_distance_check - delta_d) > 1e-6:
            logger.info("  ✗ Validation failed!")
            raise ValueError("No valid trajectory satisfies the requested constraints.")
        
        logger.info("  ✓ Validation passed!")

        phases: List[Phase] = []
        
        # Phase 1: Acceleration
        if t1 > 1e-9:
            phases.append(Phase(duration=t1, acceleration=a1))
        
        # Phase 2: Cruise (only if duration > 0)
        if t2 > 1e-9:
            phases.append(Phase(duration=t2, velocity=params.v_cruise))
        elif t2 < -1e-9:
            # This shouldn't happen, but log it if it does
            logger.warning(f"Negative cruise phase duration detected: t2={t2}")
        
        # Phase 3: Deceleration
        if t3 > 1e-9:
            phases.append(Phase(duration=t3, acceleration=a3))
        
        logger.info(f"Created {len(phases)} active phases (t1={t1:.3f}, t2={t2:.3f}, t3={t3:.3f})")
        
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
            logger.info(f"\nAttempt: {context}")
            logger.info(f"  Constraint: {constraint}")
            equations = base_equations + (constraint,)
            raw_solutions = solve(equations, (t1, t2, t3), dict=True)
            
            if not raw_solutions:
                logger.info(f"  No solutions found")
                continue

            logger.info(f"  Found {len(raw_solutions)} solution(s)")
            
            try:
                solution = self._select_solution(
                    raw_solutions,
                    (t1, t2, t3),
                    require_non_negative=True,
                )
                logger.info(f"  Solution: t1={solution[0]:.3f}, t2={solution[1]:.3f}, t3={solution[2]:.3f}")
                
                # Check if cruise speed is actually reached
                v_mid_actual = v_i + a1 * solution[0]
                logger.info(f"  v_mid_actual={v_mid_actual:.3f}, v_cruise={params.v_cruise:.1f}")
                
                return solution
            except ValueError as e:
                logger.info(f"  Solution selection failed: {e}")
                continue

        logger.info("\n✗ No valid solution found")
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
        
        for idx, candidate in enumerate(solutions):
            logger.info(f"    Candidate {idx+1}: {candidate}")
            numeric_values: List[float] = []
            all_real = True
            for symbol in symbol_order:
                value = candidate[symbol]
                evaluated = complex(value.evalf())
                if abs(evaluated.imag) > tolerance:
                    logger.info(f"      Has imaginary part: {evaluated}")
                    all_real = False
                    break
                numeric_values.append(float(evaluated.real))

            if not all_real:
                logger.info(f"      Rejected: not real")
                continue

            logger.info(f"      Values: {numeric_values}")

            if require_non_negative:
                min_component = min(numeric_values)
                if min_component < -tolerance:
                    logger.info(f"      Rejected: has negative value {min_component:.6f}")
                    continue
                numeric_values = [max(0.0, value) for value in numeric_values]

            valid_candidates.append(tuple(numeric_values))
            logger.info(f"      Accepted as valid candidate")

        if not valid_candidates:
            logger.info(f"    No valid candidates found")
            raise ValueError("No real solution found for the trajectory constraints.")

        # Return the candidate with the minimum sum of all time components
        best = min(valid_candidates, key=sum)
        logger.info(f"    Selected best candidate: {best} (sum={sum(best):.3f})")
        return best

