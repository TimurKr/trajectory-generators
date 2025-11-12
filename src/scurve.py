from __future__ import annotations

import logging
from itertools import combinations
from typing import Dict, List, Optional, Sequence, Set, Tuple

from sympy import Eq, Symbol, solve, symbols

from .base import (
    Phase,
    TrajectoryInputs,
    TrajectoryParameters,
    TrajectoryResult,
)

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class SCurveTrajectoryGenerator:
    """
    Generates S-curve velocity profile trajectories.
    
    An S-curve trajectory consists of seven phases:
    1. Positive jerk: acceleration increases from a_initial towards a_max
    2. Constant acceleration: maintains a_max (zero jerk)
    3. Negative jerk: acceleration decreases from a_max towards 0
    4. Constant velocity: cruise at v_cruise (zero jerk, zero acceleration)
    5. Negative jerk: acceleration decreases from 0 towards a_min
    6. Constant acceleration: maintains a_min (zero jerk)
    7. Positive jerk: acceleration increases from a_min to a_final
    """
    
    def __init__(self, parameters: TrajectoryParameters) -> None:
        """
        Initialize the S-curve trajectory generator with physical constraints.
        
        Args:
            parameters: Trajectory parameters containing a_max, a_min, v_cruise, j_max, j_min.
            
        Raises:
            ValueError: If any parameter violates physical constraints.
        """
        if parameters.a_max <= 0.0:
            raise ValueError("a_max must be positive.")
        if parameters.a_min >= 0.0:
            raise ValueError("a_min must be negative.")
        if parameters.v_cruise <= 0.0:
            raise ValueError("v_cruise must be positive.")
        if parameters.j_max is None or parameters.j_max <= 0.0:
            raise ValueError("j_max must be positive for S-curve trajectories.")
        if parameters.j_min is None or parameters.j_min >= 0.0:
            raise ValueError("j_min must be negative for S-curve trajectories.")

        self.parameters = parameters

    def generate_trajectory(self, inputs: TrajectoryInputs) -> TrajectoryResult:
        """
        Generate a seven-phase S-curve trajectory that satisfies the given constraints.
        
        Args:
            inputs: Trajectory inputs containing v_initial, v_final, delta_distance, 
                    a_initial, and a_final.
            
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
        a_i = inputs.a_initial if inputs.a_initial is not None else 0.0
        a_f = inputs.a_final if inputs.a_final is not None else 0.0

        # Determine the phase structure (accelerations and jerks) based on trajectory requirements

        
        # Solve for the phase durations
        phases = self._solve_phases(v_i, v_f, a_i, a_f, delta_d)
        
        # Filter out phases with zero or negligible duration
        active_phases = [p for p in phases if p.duration > 1e-9]

        return TrajectoryResult(parameters=params, inputs=inputs, phases=active_phases)

    def _determine_phase_structure(
        self,
        v_i: float,
        v_f: float,
        a_i: float,
        a_f: float,
    ) -> List[Phase]:
        """
        Determine the complete phase structure (accelerations, jerks, velocities) for all 7 phases.
        
        The structure is determined by analyzing the trajectory requirements:
        - Phase 1 (jerk): Transition from a_i towards target acceleration for phase 2
        - Phase 2 (const accel): Maintain acceleration (a_max if accelerating, a_min if decelerating)
        - Phase 3 (jerk): Transition from phase 2 acceleration to cruise (0 acceleration)
        - Phase 4 (cruise): Constant velocity at v_cruise
        - Phase 5 (jerk): Transition from 0 to target acceleration for phase 6
        - Phase 6 (const accel): Maintain deceleration/acceleration
        - Phase 7 (jerk): Transition from phase 6 acceleration to a_f
        
        Args:
            v_i: Initial velocity
            v_f: Final velocity
            a_i: Initial acceleration
            a_f: Final acceleration
            
        Returns:
            List of 7 Phase objects with accelerations/jerks/velocities set, durations = 0.0
        """
        params = self.parameters
        phases = []
        
        # Determine if we're accelerating or decelerating in the first half
        # If initial speed > cruise speed, we need to decelerate first
        if v_i > params.v_cruise:
            a2_target = params.a_min  # Decelerate
        else:
            a2_target = params.a_max  # Accelerate
        
        # Phase 1: Jerk to reach a2_target
        # If current accel > target accel, need negative jerk; otherwise positive
        if a_i > a2_target:
            j1 = params.j_min
        else:
            j1 = params.j_max
        phases.append(Phase(duration=0.0, jerk=j1))
        
        # Phase 2: Constant acceleration at a2_target
        phases.append(Phase(duration=0.0, acceleration=a2_target))
        
        # Phase 3: Jerk from a2_target to 0 (cruise condition)
        # If a2_target > 0, need negative jerk to bring to 0; if < 0, need positive
        if a2_target > 0:
            j3 = params.j_min
        else:
            j3 = params.j_max
        phases.append(Phase(duration=0.0, jerk=j3))
        
        # Phase 4: Cruise at constant velocity
        phases.append(Phase(duration=0.0, velocity=params.v_cruise))
        
        # Determine if we're accelerating or decelerating in the second half
        # If final speed > cruise speed, we need to accelerate; otherwise decelerate
        if v_f > params.v_cruise:
            a6_target = params.a_max  # Accelerate
        else:
            a6_target = params.a_min  # Decelerate
        
        # Phase 5: Jerk from 0 to a6_target
        # If 0 > a6_target, need negative jerk; otherwise positive
        if 0.0 > a6_target:
            j5 = params.j_min
        else:
            j5 = params.j_max
        phases.append(Phase(duration=0.0, jerk=j5))
        
        # Phase 6: Constant acceleration at a6_target
        phases.append(Phase(duration=0.0, acceleration=a6_target))
        
        # Phase 7: Jerk from a6_target to a_f
        # If a6_target > a_f, need negative jerk; otherwise positive
        if a6_target > a_f:
            j7 = params.j_min
        else:
            j7 = params.j_max
        phases.append(Phase(duration=0.0, jerk=j7))
        
        logger.info(f"Phase structure determined:")
        logger.info(f"  Phase 1: jerk={j1:.2f}")
        logger.info(f"  Phase 2: accel={a2_target:.2f}")
        logger.info(f"  Phase 3: jerk={j3:.2f}")
        logger.info(f"  Phase 4: velocity={params.v_cruise:.2f}")
        logger.info(f"  Phase 5: jerk={j5:.2f}")
        logger.info(f"  Phase 6: accel={a6_target:.2f}")
        logger.info(f"  Phase 7: jerk={j7:.2f}")
        
        return phases

    def _solve_phases(
        self,
        v_i: float,
        v_f: float,
        a_i: float,
        a_f: float,
        delta_d: float,
    ) -> List[Phase]:
        """
        Solve for the seven phase durations using symbolic equation solving.
        
        Strategy:
        1. First attempt: solve with all 7 phases active
        2. If solution has negative times, try constraining phases to zero
        3. Common constraints: t4=0 (no cruise), t2=0 or t6=0 (no constant accel)
        
        Args:
            phase_templates: List of 7 Phase objects with jerks/accels/velocities set, durations=0
            v_i: Initial velocity
            v_f: Final velocity
            a_i: Initial acceleration
            a_f: Final acceleration
            delta_d: Total distance to travel
            
        Returns:
            List of 7 Phase objects with durations filled in.
            
        Raises:
            ValueError: If no valid solution is found.
        """

        phases = self._determine_phase_structure(v_i, v_f, a_i, a_f)
        
        # Extract jerks and accelerations from phase templates
        j1 = phases[0].jerk
        a2 = phases[1].acceleration
        j3 = phases[2].jerk
        v_cruise = phases[3].velocity
        j5 = phases[4].jerk
        a6 = phases[5].acceleration
        j7 = phases[6].jerk
        
        # Define symbolic variables
        t1, t2, t3, t4, t5, t6, t7 = symbols("t1 t2 t3 t4 t5 t6 t7", real=True)
        
        # Build the equations for all 7 phases
        # Track acceleration through phases
        a1_end = a_i + j1 * t1
        a2_end = a1_end
        a3_end = a2_end + j3 * t3
        a4_end = a3_end
        a5_end = a4_end + j5 * t5
        a6_end = a5_end
        a7_end = a6_end + j7 * t7
        
        # Track velocity through phases
        v1_end = v_i + a_i * t1 + 0.5 * j1 * t1**2
        v2_end = v1_end + a1_end * t2
        v3_end = v2_end + a2_end * t3 + 0.5 * j3 * t3**2
        v4_end = v3_end
        v5_end = v4_end + a4_end * t5 + 0.5 * j5 * t5**2
        v6_end = v5_end + a5_end * t6
        v7_end = v6_end + a6_end * t7 + 0.5 * j7 * t7**2
        
        # Track distance through phases
        d1 = v_i * t1 + 0.5 * a_i * t1**2 + (1.0/6.0) * j1 * t1**3
        d2 = v1_end * t2 + 0.5 * a1_end * t2**2
        d3 = v2_end * t3 + 0.5 * a2_end * t3**2 + (1.0/6.0) * j3 * t3**3
        d4 = v3_end * t4 
        d5 = v4_end * t5 + 0.5 * a4_end * t5**2 + (1.0/6.0) * j5 * t5**3
        d6 = v5_end * t6 + 0.5 * a5_end * t6**2
        d7 = v6_end * t7 + 0.5 * a6_end * t7**2 + (1.0/6.0) * j7 * t7**3
        
        # Base equations (always apply)
        base_equations = [
            Eq(a7_end, a_f),  # Final acceleration constraint
            Eq(v7_end, v_f),  # Final velocity constraint
            Eq(d1 + d2 + d3 + d4 + d5 + d6 + d7, delta_d),  # Distance constraint
        ]
        
        # Try different constraint combinations
        # Priority: fewest constraints first (try all phases active)
        constraint_sets = [
            # All phases active with specific velocity/acceleration constraints
            [
                Eq(a1_end, a2),  # Phase 1 reaches target accel
                Eq(a3_end, 0.0),  # Phase 3 brings accel to 0
                Eq(v3_end, v_cruise),  # Reach cruise velocity
                Eq(a5_end, a6),  # Phase 5 reaches target decel
            ],
            # No cruise phase (t4 = 0)
            [
                Eq(t4, 0.0),
                Eq(a1_end, a2),
                Eq(a3_end, 0.0),
                Eq(a5_end, a6),
            ],
            # No constant acceleration in first half (t2 = 0)
            [
                Eq(t2, 0.0),
                Eq(a3_end, 0.0),
                Eq(v3_end, v_cruise),
                Eq(a5_end, a6),
            ],
            # No constant acceleration in second half (t6 = 0)
            [
                Eq(t6, 0.0),
                Eq(a1_end, a2),
                Eq(a3_end, 0.0),
                Eq(v3_end, v_cruise),
            ],
            # No cruise, no constant accel in first half
            [
                Eq(t2, 0.0),
                Eq(t4, 0.0),
                Eq(a3_end, 0.0),
                Eq(a5_end, a6),
            ],
            # No cruise, no constant accel in second half
            [
                Eq(t4, 0.0),
                Eq(t6, 0.0),
                Eq(a1_end, a2),
                Eq(a3_end, 0.0),
            ],
            # No constant accel phases at all
            [
                Eq(t2, 0.0),
                Eq(t4, 0.0),
                Eq(t6, 0.0),
                Eq(a3_end, 0.0),
            ],
            # Minimal constraints - just structural
            [
                Eq(t2, 0.0),
                Eq(t4, 0.0),
                Eq(t6, 0.0),
            ],
        ]
        
        all_symbols = (t1, t2, t3, t4, t5, t6, t7)
        
        # Try solving with full constraints first
        logger.info("="*70)
        logger.info(f"Solving S-curve: v_i={v_i:.1f}, v_f={v_f:.1f}, a_i={a_i:.1f}, a_f={a_f:.1f}, Δd={delta_d:.1f}")
        logger.info(f"Jerks: j1={j1:.2f}, j3={j3:.2f}, j5={j5:.2f}, j7={j7:.2f}")
        logger.info("\nAttempt 1: Full 7-phase solution")
        
        full_constraints = constraint_sets[0]  # All phases active
        try:
            equations = base_equations + full_constraints
            raw_solutions = solve(equations, all_symbols, dict=True)
            
            if raw_solutions:
                logger.info(f"  Found {len(raw_solutions)} solution(s)")
                # Analyze the best solution
                best_sol, negative_phases = self._analyze_solution_for_negatives(raw_solutions, all_symbols)
                
                if not negative_phases:
                    logger.info(f"  Times: t={[f'{t:.3f}' for t in best_sol]}")
                    if self._validate_solution(best_sol, v_i, v_f, a_i, a_f, delta_d, j1, j3, j5, j7, a2, a6):
                        logger.info("  ✓ Valid solution!")
                        return self._create_phases_with_durations(phases, best_sol)
                else:
                    logger.info(f"  Negative phases detected: {sorted(negative_phases)}")
                    logger.info(f"  Times: t={[f'{t:.3f}' for t in best_sol]}")
                    
                    # Try again with those phases set to zero
                    attempt_num = 2
                    for phase in sorted(negative_phases):
                        logger.info(f"\nAttempt {attempt_num}: Setting {phase} = 0")
                        attempt_num += 1
                        
                        # Find the appropriate constraint set
                        phase_to_idx = {'t2': 2, 't4': 1, 't6': 3}
                        if phase in phase_to_idx:
                            idx = phase_to_idx[phase]
                            constraints_to_try = constraint_sets[idx]
                            
                            try:
                                equations = base_equations + constraints_to_try
                                raw_solutions = solve(equations, all_symbols, dict=True)
                                if raw_solutions:
                                    solution = self._select_solution(raw_solutions, all_symbols, require_non_negative=True)
                                    logger.info(f"  Times: t={[f'{t:.3f}' for t in solution]}")
                                    if self._validate_solution(solution, v_i, v_f, a_i, a_f, delta_d, j1, j3, j5, j7, a2, a6):
                                        logger.info("  ✓ Valid solution!")
                                        return self._create_phases_with_durations(phases, solution)
                            except (ValueError, Exception) as e:
                                logger.info(f"  Failed: {e}")
        except Exception as e:
            logger.info(f"  Failed: {e}")
        
        # Try other constraint sets
        logger.info("\nTrying fallback constraints...")
        for attempt_num, additional_constraints in enumerate(constraint_sets[1:], start=2):
            equations = base_equations + additional_constraints
            try:
                raw_solutions = solve(equations, all_symbols, dict=True)
                if not raw_solutions:
                    continue
                
                solution = self._select_solution(
                    raw_solutions,
                    all_symbols,
                    require_non_negative=True,
                )
                
                logger.info(f"Attempt {attempt_num}: t={[f'{t:.3f}' for t in solution]}")
                
                # Validate solution
                if self._validate_solution(solution, v_i, v_f, a_i, a_f, delta_d, j1, j3, j5, j7, a2, a6):
                    logger.info("  ✓ Valid!")
                    return self._create_phases_with_durations(phases, solution)
            except (ValueError, Exception):
                continue
        
        logger.info("✗ No solution found")
        raise ValueError("No real solution found for the S-curve trajectory constraints.")
    
    def _create_phases_with_durations(
        self,
        phase_templates: List[Phase],
        durations: Tuple[float, ...]
    ) -> List[Phase]:
        """
        Create Phase objects with durations filled in from the solution.
        
        Args:
            phase_templates: List of 7 Phase objects with jerks/accels/velocities set
            durations: Tuple of 7 durations (t1, t2, t3, t4, t5, t6, t7)
            
        Returns:
            List of 7 Phase objects with durations set
        """
        phases = []
        for template, duration in zip(phase_templates, durations):
            if template.jerk is not None:
                phases.append(Phase(duration=duration, jerk=template.jerk))
            elif template.acceleration is not None:
                phases.append(Phase(duration=duration, acceleration=template.acceleration))
            elif template.velocity is not None:
                phases.append(Phase(duration=duration, velocity=template.velocity))
            else:
                # Shouldn't happen, but handle it
                phases.append(Phase(duration=duration))
        return phases
    
    def _analyze_solution_for_negatives(
        self,
        raw_solutions: List[Dict],
        symbol_order: Sequence[Symbol],
        tolerance: float = 1e-9
    ) -> Tuple[Tuple[float, ...], Set[str]]:
        """
        Analyze solutions to find best one and identify negative phases.
        
        Returns:
            (solution_tuple, set_of_negative_phase_names)
        """
        best_solution = None
        best_score = float('inf')
        best_negatives = set()
        
        for candidate in raw_solutions:
            values = []
            all_real = True
            
            for symbol in symbol_order:
                val = candidate[symbol]
                evaluated = complex(val.evalf())
                if abs(evaluated.imag) > tolerance:
                    all_real = False
                    break
                values.append(float(evaluated.real))
            
            if not all_real:
                continue
            
            # Find negative phases
            negatives = set()
            phase_names = ['t1', 't2', 't3', 't4', 't5', 't6', 't7']
            for i, v in enumerate(values):
                if v < -tolerance:
                    negatives.add(phase_names[i])
            
            # Score: fewer negatives is better
            num_neg = len(negatives)
            abs_sum = sum(abs(v) for v in values if v < -tolerance)
            score = (num_neg, abs_sum)
            
            if best_solution is None or score < best_score:
                best_score = score
                best_solution = tuple(max(0.0, v) for v in values)
                best_negatives = negatives
        
        if best_solution is None:
            raise ValueError("No real solutions")
        
        return best_solution, best_negatives

    def _validate_solution(
        self,
        times: Tuple[float, ...],
        v_i: float,
        v_f: float,
        a_i: float,
        a_f: float,
        delta_d: float,
        j1: float,
        j3: float,
        j5: float,
        j7: float,
        a2: float,
        a6: float,
        tolerance: float = 1e-4,
    ) -> bool:
        """
        Validate that a solution satisfies all constraints.
        
        Args:
            times: Tuple of (t1, t2, t3, t4, t5, t6, t7)
            v_i: Initial velocity
            v_f: Final velocity
            a_i: Initial acceleration
            a_f: Final acceleration
            delta_d: Total distance
            j1, j3, j5, j7: Jerk values for jerk phases
            a2, a6: Target accelerations for constant acceleration phases
            tolerance: Numerical tolerance for validation
            
        Returns:
            True if solution is valid, False otherwise.
        """
        t1, t2, t3, t4, t5, t6, t7 = times
        params = self.parameters
        
        # Track state through all phases
        a = a_i
        v = v_i
        d = 0.0
        
        # Phase 1: jerk
        if t1 > 0:
            d += v * t1 + 0.5 * a * t1**2 + (1.0/6.0) * j1 * t1**3
            v += a * t1 + 0.5 * j1 * t1**2
            a += j1 * t1
        
        # Phase 2: constant acceleration
        if t2 > 0:
            d += v * t2 + 0.5 * a * t2**2
            v += a * t2
        
        # Phase 3: jerk
        if t3 > 0:
            d += v * t3 + 0.5 * a * t3**2 + (1.0/6.0) * j3 * t3**3
            v += a * t3 + 0.5 * j3 * t3**2
            a += j3 * t3
        
        # Phase 4: cruise
        if t4 > 0:
            d += v * t4
        
        # Phase 5: jerk
        if t5 > 0:
            d += v * t5 + 0.5 * a * t5**2 + (1.0/6.0) * j5 * t5**3
            v += a * t5 + 0.5 * j5 * t5**2
            a += j5 * t5
        
        # Phase 6: constant acceleration
        if t6 > 0:
            d += v * t6 + 0.5 * a * t6**2
            v += a * t6
        
        # Phase 7: jerk
        if t7 > 0:
            d += v * t7 + 0.5 * a * t7**2 + (1.0/6.0) * j7 * t7**3
            v += a * t7 + 0.5 * j7 * t7**2
            a += j7 * t7
        
        # Check if final state matches requirements
        velocity_error = abs(v - v_f)
        acceleration_error = abs(a - a_f)
        distance_error = abs(d - delta_d)
        
        return (
            velocity_error < tolerance and
            acceleration_error < tolerance and
            distance_error < tolerance
        )

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
                # Clamp small negative values to zero
                numeric_values = [max(0.0, value) for value in numeric_values]

            valid_candidates.append(tuple(numeric_values))

        if not valid_candidates:
            raise ValueError("No real solution found for the trajectory constraints.")

        # Return the candidate with the minimum sum of all time components
        return min(valid_candidates, key=sum)

