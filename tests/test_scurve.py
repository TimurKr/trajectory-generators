#!/usr/bin/env python3
"""Test suite for S-curve trajectory generator."""

from __future__ import annotations

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.base import TrajectoryInputs, TrajectoryParameters
from src.scurve import SCurveTrajectoryGenerator
from tests.test_helpers import validate_trajectory


def run_scurve_test(
    test_name: str,
    params: TrajectoryParameters,
    inputs: TrajectoryInputs,
) -> bool:
    """Run a single S-curve trajectory test."""
    print(f"\n{'='*70}")
    print(f"TEST: {test_name}")
    print(f"{'='*70}")
    print(f"Inputs: v_i={inputs.v_initial:.1f}, v_f={inputs.v_final:.1f}, distance={inputs.delta_distance:.1f}")
    if inputs.a_initial is not None:
        print(f"        a_i={inputs.a_initial:.1f}, a_f={inputs.a_final:.1f}")
    
    try:
        gen = SCurveTrajectoryGenerator(params)
        result = gen.generate_trajectory(inputs)
        
        print(f"Total time: {result.total_time:.3f} s")
        print(f"Phases ({len(result.phases)}):")
        for i, phase in enumerate(result.phases):
            if phase.jerk is not None:
                phase_type = "Jerk"
                phase_value = phase.jerk
            elif phase.acceleration is not None:
                phase_type = "Accel"
                phase_value = phase.acceleration
            elif phase.velocity is not None:
                phase_type = "Cruise"
                phase_value = phase.velocity
            else:
                phase_type = "Unknown"
                phase_value = 0.0
            print(f"  Phase {i+1} ({phase_type}): t={phase.duration:.3f}s, value={phase_value:.3f}")
        
        # Validate
        expected_a_final = inputs.a_final if inputs.a_final is not None else 0.0
        is_valid, error_msg = validate_trajectory(
            result,
            inputs.v_final,
            inputs.delta_distance,
            expected_a_final,
        )
        
        if is_valid:
            print("✓ PASSED")
            return True
        else:
            print(f"✗ FAILED: {error_msg}")
            return False
            
    except ValueError as e:
        print(f"✗ FAILED: {e}")
        return False


def main() -> int:
    """Run all S-curve trajectory tests."""
    print("="*70)
    print("S-CURVE TRAJECTORY GENERATOR TEST SUITE")
    print("="*70)
    
    all_passed = True
    
    # Test 1: Basic case - start and end at rest
    params = TrajectoryParameters(a_max=4.0, a_min=-3.0, v_cruise=100.0, j_max=1.0, j_min=-1.0)
    inputs = TrajectoryInputs(v_initial=0.0, v_final=0.0, delta_distance=5000.0, a_initial=0.0, a_final=0.0)
    if not run_scurve_test("Basic: Rest to Rest", params, inputs):
        all_passed = False
    
    # Test 2: Short distance (no cruise phase)
    inputs = TrajectoryInputs(v_initial=0.0, v_final=0.0, delta_distance=500.0, a_initial=0.0, a_final=0.0)
    if not run_scurve_test("Short Distance (No Cruise)", params, inputs):
        all_passed = False
    
    # Test 3: Initial velocity non-zero
    inputs = TrajectoryInputs(v_initial=50.0, v_final=0.0, delta_distance=2000.0, a_initial=0.0, a_final=0.0)
    if not run_scurve_test("Initial Velocity 50 m/s", params, inputs):
        all_passed = False
    
    # Test 4: Non-zero final velocity
    inputs = TrajectoryInputs(v_initial=0.0, v_final=80.0, delta_distance=3000.0, a_initial=0.0, a_final=0.0)
    if not run_scurve_test("Final Velocity 80 m/s", params, inputs):
        all_passed = False
    
    # Test 5: Both velocities non-zero
    inputs = TrajectoryInputs(v_initial=30.0, v_final=70.0, delta_distance=4000.0, a_initial=0.0, a_final=0.0)
    if not run_scurve_test("Both Velocities Non-zero", params, inputs):
        all_passed = False
    
    # Test 6: Very short distance
    inputs = TrajectoryInputs(v_initial=0.0, v_final=0.0, delta_distance=100.0, a_initial=0.0, a_final=0.0)
    if not run_scurve_test("Very Short Distance", params, inputs):
        all_passed = False
    
    # Test 7: High cruise speed
    params = TrajectoryParameters(a_max=5.0, a_min=-5.0, v_cruise=150.0, j_max=2.0, j_min=-2.0)
    inputs = TrajectoryInputs(v_initial=0.0, v_final=0.0, delta_distance=10000.0, a_initial=0.0, a_final=0.0)
    if not run_scurve_test("High Speed Long Distance", params, inputs):
        all_passed = False
    
    # Test 8: Non-zero initial acceleration
    params = TrajectoryParameters(a_max=4.0, a_min=-3.0, v_cruise=100.0, j_max=1.0, j_min=-1.0)
    inputs = TrajectoryInputs(v_initial=20.0, v_final=20.0, delta_distance=3000.0, a_initial=2.0, a_final=0.0)
    if not run_scurve_test("Non-zero Initial Acceleration", params, inputs):
        all_passed = False
    
    # Test 9: Asymmetric acceleration limits
    params = TrajectoryParameters(a_max=6.0, a_min=-2.0, v_cruise=80.0, j_max=1.5, j_min=-1.5)
    inputs = TrajectoryInputs(v_initial=0.0, v_final=0.0, delta_distance=4000.0, a_initial=0.0, a_final=0.0)
    if not run_scurve_test("Asymmetric Acceleration Limits", params, inputs):
        all_passed = False
    
    # Test 10: Low jerk (soft S-curve)
    params = TrajectoryParameters(a_max=4.0, a_min=-3.0, v_cruise=100.0, j_max=0.5, j_min=-0.5)
    inputs = TrajectoryInputs(v_initial=0.0, v_final=0.0, delta_distance=5000.0, a_initial=0.0, a_final=0.0)
    if not run_scurve_test("Low Jerk (Soft S-curve)", params, inputs):
        all_passed = False
    
    # Test 11: High initial velocity (tests consecutive duplicate phase detection)
    # When v_initial > v_cruise, the system decelerates, which can cause
    # consecutive phases to have identical profiles (e.g., both with j_min)
    params = TrajectoryParameters(a_max=4.0, a_min=-3.0, v_cruise=50.0, j_max=1.0, j_min=-1.0)
    inputs = TrajectoryInputs(v_initial=80.0, v_final=0.0, delta_distance=2000.0, a_initial=0.0, a_final=0.0)
    if not run_scurve_test("High Initial Velocity (Deceleration)", params, inputs):
        all_passed = False
    
    # Final summary
    print(f"\n{'='*70}")
    if all_passed:
        print("✓ ALL TESTS PASSED")
        print(f"{'='*70}")
        return 0
    else:
        print("✗ SOME TESTS FAILED")
        print(f"{'='*70}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

