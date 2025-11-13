# Trajectory Generators

A Streamlit application for generating and comparing **Trapezoidal** and **S-Curve** trajectory profiles.

## Features

- **Dual Trajectory Generation**: Generate both trapezoidal and S-curve trajectories simultaneously
- **Side-by-Side Comparison**: Visualize both trajectory types on the same graphs
- **Comprehensive Profiles**: View position, velocity, acceleration, and jerk profiles
- **Interactive Parameters**: Adjust acceleration limits, jerk limits, cruise speed, and initial conditions
- **Export Capability**: Save trajectory comparisons as PNG images

## Trajectory Types

### Trapezoidal Trajectory (3 Phases)

1. **Acceleration Phase**: Constant acceleration from initial to cruise velocity
2. **Cruise Phase**: Constant velocity at cruise speed (may be zero duration)
3. **Deceleration Phase**: Constant acceleration from cruise to final velocity

### S-Curve Trajectory (7 Phases)

1. **Jerk Phase 1**: Acceleration increases from initial acceleration towards max
2. **Constant Acceleration**: Maintains maximum acceleration (zero jerk)
3. **Jerk Phase 2**: Acceleration decreases to zero
4. **Cruise Phase**: Constant velocity (zero jerk, zero acceleration)
5. **Jerk Phase 3**: Deceleration begins (negative jerk)
6. **Constant Deceleration**: Maintains minimum acceleration (zero jerk)
7. **Jerk Phase 4**: Deceleration decreases to final acceleration

## Installation

1. Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Run the Streamlit App

```bash
streamlit run trajectory_generator.py
```

The app provides interactive sliders for:

- **Acceleration Limits**: `a_max` and `a_min`
- **Jerk Limits** (S-Curve): `j_max` and `j_min`
- **Cruise Speed**: `v_cruise`
- **Initial State**: Initial velocity and acceleration
- **Final State**: Final velocity and acceleration
- **Distance**: Total distance to travel

### Run Tests

To validate the implementation with various parameter combinations:

```bash
python test_trajectories.py
```

The test suite includes 10 test cases covering:

- Rest to rest motion
- Short and long distances
- Non-zero initial/final velocities
- Non-zero initial/final accelerations
- Asymmetric acceleration limits
- Various jerk settings

## File Structure

```
trajectory-generators/
├── trajectory_base.py           # Shared base classes and data structures
├── trajectory_trapezoidal.py    # Trapezoidal trajectory generator
├── trajectory_scurve.py         # S-curve trajectory generator
├── trajectory_generator.py      # Streamlit application
├── plotting.py                  # Visualization functions
├── test_trajectories.py         # Test suite
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## Key Differences: Trapezoidal vs S-Curve

| Feature               | Trapezoidal                  | S-Curve                        |
| --------------------- | ---------------------------- | ------------------------------ |
| **Phases**            | 3                            | 7                              |
| **Jerk**              | Infinite (discontinuous)     | Limited by j_max/j_min         |
| **Smoothness**        | Step changes in acceleration | Smooth acceleration changes    |
| **Complexity**        | Simpler equations            | More complex constraint system |
| **Flexibility**       | May fail on short distances  | More adaptable to constraints  |
| **Mechanical Stress** | Higher                       | Lower (smoother motion)        |

## Algorithm Details

### Trapezoidal Solver

The trapezoidal trajectory uses symbolic equation solving with two constraint attempts:

1. **Three-phase solution**: Assumes cruise phase reaches `v_cruise`
2. **Two-phase solution**: If no cruise phase (t2 = 0), solves directly

### S-Curve Solver

The S-curve trajectory uses iterative constraint solving:

1. **Initial attempt**: Solve with all 7 phases active
2. **Constraint relaxation**: If phases have negative times, iteratively set them to zero:
   - No cruise phase (t4 = 0)
   - No constant acceleration phases (t2 = 0 or t6 = 0)
   - Various combinations until a valid solution is found
3. **Validation**: Verify final state matches constraints within tolerance

## Governing Equations

### Trapezoidal

```
Distance: Δd = d1 + d2 + d3
  where d1 = ½(v_i + v_mid)·t1  (acceleration)
        d2 = v_mid·t2            (cruise)
        d3 = ½(v_mid + v_f)·t3   (deceleration)

Velocity: v_f = v_i + a1·t1 + a3·t3
Peak velocity: v_mid = v_cruise (when t2 > 0)
```

### S-Curve

```
Distance: Δd = Σ(i=1 to 7) di
  where each phase contributes based on initial velocity, acceleration, and jerk

Velocity: v_f = v_i + ∫∫(jerk profile over all phases)
Acceleration: a_f = a_i + ∫(jerk profile over all phases)

Constraints:
  - |a| ≤ a_max (or ≥ a_min)
  - v ≤ v_cruise
  - |j| ≤ j_max (or ≥ j_min)
```

## Notes

- S-curve trajectories are more flexible and can handle cases where trapezoidal profiles fail (very short distances, high initial velocities with limited space)
- The "soft" default jerk limits (±1.0 m/s³) provide smooth motion suitable for robotics and automation
- All trajectory generators use SymPy for symbolic equation solving to ensure exact solutions
- The visualization uses high-resolution sampling (1000+ points) for smooth curves

## License

This project is provided as-is for educational and research purposes.
