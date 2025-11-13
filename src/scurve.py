from __future__ import annotations

import logging
from dataclasses import dataclass
from itertools import combinations
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

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

@dataclass
class PhaseSymbolic:
    """Symbolic representation of a phase during trajectory solving."""
    index: int  # 0-6 for phases 1-7
    time_symbol: Symbol
    jerk: Optional[float] = None
    acceleration: Optional[float] = None
    velocity: Optional[float] = None
    
    # Symbolic expressions (computed during solving)
    accel_start: Any = None
    accel_end: Any = None
    velocity_start: Any = None
    velocity_end: Any = None
    distance: Any = None
    
    def get_profile_key(self) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """Get a key representing the phase profile (jerk, accel, velocity)."""
        return (self.jerk, self.acceleration, self.velocity)

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

    def _create_symbolic_phases(
        self, phase_templates: List[Phase], v_i: float, a_i: float
    ) -> List[PhaseSymbolic]:
        """Create symbolic phase objects from templates."""
        t1, t2, t3, t4, t5, t6, t7 = symbols("t1 t2 t3 t4 t5 t6 t7", real=True)
        time_symbols = [t1, t2, t3, t4, t5, t6, t7]
        
        symbolic_phases = []
        for i, (template, t_sym) in enumerate(zip(phase_templates, time_symbols)):
            phase = PhaseSymbolic(
                index=i,
                time_symbol=t_sym,
                jerk=template.jerk,
                acceleration=template.acceleration,
                velocity=template.velocity
            )
            symbolic_phases.append(phase)
        
        # Set initial conditions for first phase
        symbolic_phases[0].accel_start = a_i
        symbolic_phases[0].velocity_start = v_i
        
        return symbolic_phases
    
    def _compute_phase_expressions(self, phases: List[PhaseSymbolic]) -> None:
        """Compute symbolic expressions for acceleration, velocity, and distance through all phases."""
        for i, phase in enumerate(phases):
            t = phase.time_symbol
            
            # Get starting conditions from previous phase (or use initial if first phase)
            if i == 0:
                a_start = phase.accel_start
                v_start = phase.velocity_start
            else:
                a_start = phases[i-1].accel_end
                v_start = phases[i-1].velocity_end
            
            phase.accel_start = a_start
            phase.velocity_start = v_start
            
            # Compute end conditions based on phase type
            if phase.jerk is not None:
                # Jerk phase
                phase.accel_end = a_start + phase.jerk * t
                phase.velocity_end = v_start + a_start * t + 0.5 * phase.jerk * t**2
                phase.distance = v_start * t + 0.5 * a_start * t**2 + (1.0/6.0) * phase.jerk * t**3
            elif phase.acceleration is not None:
                # Constant acceleration phase
                phase.accel_end = a_start  # Acceleration doesn't change
                phase.velocity_end = v_start + a_start * t
                phase.distance = v_start * t + 0.5 * a_start * t**2
            elif phase.velocity is not None:
                # Constant velocity phase (cruise)
                phase.accel_end = a_start  # Should be 0
                phase.velocity_end = v_start  # Should equal phase.velocity
                phase.distance = v_start * t
            else:
                # Shouldn't happen
                phase.accel_end = a_start
                phase.velocity_end = v_start
                phase.distance = 0
    
    def _build_constraints(
        self, 
        symbolic_phases: List[PhaseSymbolic],
        phase_templates: List[Phase],
        skipped_phases: Set[str],
        a_f: float,
        v_f: float,
        delta_d: float,
    ) -> List[Eq]:
        """Build constraint equations based on which phases are active."""
        constraints = [
            Eq(symbolic_phases[-1].accel_end, a_f),  # Final acceleration
            Eq(symbolic_phases[-1].velocity_end, v_f),  # Final velocity
            Eq(sum(p.distance for p in symbolic_phases), delta_d),  # Total distance
        ]
        
        # Phase 2 (index 1): constant accel - should reach target acceleration
        if 't2' not in skipped_phases:
            constraints.append(Eq(symbolic_phases[0].accel_end, phase_templates[1].acceleration))
        
        # Phase 4 (index 3): cruise - should reach cruise velocity with zero acceleration
        if 't4' not in skipped_phases:
            constraints.append(Eq(symbolic_phases[2].accel_end, 0.0))
            constraints.append(Eq(symbolic_phases[2].velocity_end, phase_templates[3].velocity))
        
        # Phase 6 (index 5): constant accel - should reach target acceleration
        if 't6' not in skipped_phases:
            constraints.append(Eq(symbolic_phases[4].accel_end, phase_templates[5].acceleration))
        
        # Add zero constraints for skipped phases
        for phase_name in skipped_phases:
            phase_idx = int(phase_name[1]) - 1
            constraints.append(Eq(symbolic_phases[phase_idx].time_symbol, 0.0))
            
            # If 2 phases around the skipped phase have the same profile, set their times to be equal
            if symbolic_phases[phase_idx-1].get_profile_key() == symbolic_phases[phase_idx + 1].get_profile_key():
                constraints.append(Eq(symbolic_phases[phase_idx-1].time_symbol, symbolic_phases[phase_idx + 1].time_symbol))
        
        return constraints
    
    def _validate_solution_simple(
        self,
        symbolic_phases: List[PhaseSymbolic],
        times: Tuple[float, ...],
        v_f: float,
        a_f: float,
        delta_d: float,
        tolerance: float = 1e-4,
    ) -> bool:
        """Validate solution by substituting times into symbolic expressions."""
        substitutions = {p.time_symbol: t for p, t in zip(symbolic_phases, times)}
        
        final_velocity = float(symbolic_phases[-1].velocity_end.evalf(subs=substitutions))
        final_accel = float(symbolic_phases[-1].accel_end.evalf(subs=substitutions))
        total_distance = float(sum(p.distance for p in symbolic_phases).evalf(subs=substitutions))
        
        velocity_error = abs(final_velocity - v_f)
        accel_error = abs(final_accel - a_f)
        distance_error = abs(total_distance - delta_d)
        
        return (velocity_error < tolerance and 
                accel_error < tolerance and 
                distance_error < tolerance)

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
        3. Detect underdetermined cases (consecutive phases with same profile)
        
        Returns:
            List of 7 Phase objects with durations filled in.
            
        Raises:
            ValueError: If no valid solution is found.
        """
        phase_templates = self._determine_phase_structure(v_i, v_f, a_i, a_f)
        
        # Create symbolic phase objects
        symbolic_phases = self._create_symbolic_phases(phase_templates, v_i, a_i)
        
        logger.info("="*70)
        logger.info(f"Solving S-curve: v_i={v_i:.1f}, v_f={v_f:.1f}, a_i={a_i:.1f}, a_f={a_f:.1f}, Δd={delta_d:.1f}")
        
        # Compute symbolic expressions for each phase
        self._compute_phase_expressions(symbolic_phases)
        
        # Phases that can be skipped (constant accel/velocity phases)
        skippable_phases = {'t2', 't4', 't6'}
        skipped_phases = set()
        attempt_num = 1
        all_symbols = tuple(p.time_symbol for p in symbolic_phases)
        
        while True:
            # Build constraints
            equations = self._build_constraints(symbolic_phases, phase_templates, skipped_phases, a_f, v_f, delta_d)

            # Log attempt
            logger.info(f"\nAttempt {attempt_num}: {'All phases active' if not skipped_phases else f'Skipped phases: {sorted(skipped_phases)}'}")
            
            try:
                raw_solutions = solve(equations, all_symbols, dict=True)
                
                if not raw_solutions:
                    logger.info("  No solutions found (possibly underdetermined)")
                    break
                
                logger.info(f"  Found {len(raw_solutions)} solution(s):")
                for idx, sol in enumerate(raw_solutions, 1):
                    readable = ', '.join(f"{str(k)}={float(v):.4f}" for k, v in sol.items())
                    logger.info(f"    Solution {idx}: {readable}")
                
                # Try to select a non-negative solution first
                try:
                    solution = self._select_solution(raw_solutions, all_symbols, require_non_negative=True)
                    if self._validate_solution_simple(symbolic_phases, solution, v_f, a_f, delta_d):
                        logger.info("  ✓ Valid solution!")
                        return self._create_phases_with_durations(phase_templates, solution)
                except ValueError:
                    pass

                # Otherwise, select best solution (may have negatives)
                solution = self._select_solution(raw_solutions, all_symbols, require_non_negative=False)
                logger.info(f"  Times: {[f'{t:.3f}' for t in solution]}")
                
                # Check for negative phases
                negative_phases = {f't{i+1}' for i, t in enumerate(solution) if t < -1e-9}
                
                if negative_phases:
                    negative_skippable = negative_phases & skippable_phases
                    if negative_skippable:
                        phase_to_skip = sorted(negative_skippable)[0]
                        skipped_phases.add(phase_to_skip)
                        logger.info(f"  ✗ Negative phase: {sorted(negative_phases)} → skipping {phase_to_skip}")
                        attempt_num += 1
                        continue
                    else:
                        logger.info(f"  ✗ Negative non-skippable phase: {sorted(negative_phases)}")
                        break
                    
            except Exception as e:
                logger.info(f"  Failed: {e}")
                break
        
        logger.info("\n✗ No solution found")
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
