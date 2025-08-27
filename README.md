# 3D Electromagnetic Particle-in-Cell (PIC) Simulation

This is a C++ implementation of a 3D electromagnetic particle-in-cell simulation using:
- **Yee's method** (FDTD) for electromagnetic field updates
- **Boris method** for relativistic particle pushing
- **Periodic boundary conditions** in all three directions
- **Cloud-In-Cell (CIC)** charge deposition

## Features

### Core Algorithms

1. **Electromagnetic Field Solver (Yee's Method)**
   - Staggered grid arrangement for E and B fields
   - Second-order accurate in space and time
   - Implements Maxwell's equations:
     - Faraday's law: ∂B/∂t = -∇×E
     - Ampere's law: ∂E/∂t = c²(∇×B - μ₀J)

2. **Boris Particle Pusher**
   - Relativistic particle motion in electromagnetic fields
   - Separates E and B field effects for numerical stability
   - Conserves energy to machine precision for constant fields

3. **Periodic Boundary Conditions**
   - Particles wrap around simulation domain
   - Fields use modular arithmetic for array indexing
   - Maintains charge neutrality

4. **Charge and Current Deposition**
   - Cloud-In-Cell (CIC) interpolation for smooth field coupling
   - Distributes particle quantities to 8 nearest grid points

## Building and Running

### Compilation

```bash
# Build optimized release version
make

# Build debug version with symbols
make debug

# Build with OpenMP parallelization
make parallel

# Clean build artifacts
make clean
```

### Running the Simulation

```bash
# Run with default parameters
./pic3d

# The simulation will output:
# - Console progress updates
# - pic3d_diagnostics.txt: Energy conservation diagnostics
# - particles_initial.txt: Initial particle positions
# - particles_final.txt: Final particle positions
```

## Code Structure

### Main Components

- **`SimulationParams`**: Configuration parameters (grid size, timestep, etc.)
- **`Particle`**: Individual particle data (position, velocity, charge, mass)
- **`Field3D`**: 3D array class with periodic boundary conditions
- **`PIC3D`**: Main simulation class containing:
  - Field arrays (Ex, Ey, Ez, Bx, By, Bz, Jx, Jy, Jz, rho)
  - Particle array
  - Core simulation methods

### Key Methods

- `initializeParticles()`: Sets up initial particle distribution
- `depositCharge()`: Maps particle charge to grid
- `depositCurrent()`: Maps particle current to grid
- `updateEField()`: Advances E field using Ampere's law
- `updateBField()`: Advances B field using Faraday's law
- `interpolateFields()`: Gets field values at particle positions
- `pushParticles()`: Advances particle positions and velocities
- `run()`: Main simulation loop

## Physical Parameters

### Default Configuration
- Grid: 32×32×32 cells
- Cell size: 1 μm
- Particles: 10,000 (5,000 electrons + 5,000 protons)
- Time step: Set by CFL condition (Δt = 0.5 × Δx/c)
- Initial velocities: Maxwellian distribution

### Stability Considerations

The simulation requires careful parameter tuning:
1. **CFL Condition**: Δt ≤ Δx/(√3 × c) for stability
2. **Debye Length**: Grid spacing should resolve λD
3. **Plasma Frequency**: Time step should resolve ωp
4. **Particle Density**: Sufficient particles per cell for statistics

## Output Files

1. **pic3d_diagnostics.txt**: Time evolution of:
   - Field energy
   - Particle kinetic energy
   - Total energy (should be conserved)

2. **particles_initial.txt**: Initial particle state
   - Columns: x, y, z, vx, vy, vz, species

3. **particles_final.txt**: Final particle state
   - Same format as initial file

## Visualization

The particle position files can be visualized using:
- ParaView (recommended)
- Python with matplotlib
- gnuplot

Example Python visualization:
```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load particle data
data = np.loadtxt('particles_final.txt')
electrons = data[data[:, 6] == 0]
ions = data[data[:, 6] == 1]

# 3D scatter plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(electrons[:, 0], electrons[:, 1], electrons[:, 2], c='b', s=1, alpha=0.5, label='Electrons')
ax.scatter(ions[:, 0], ions[:, 1], ions[:, 2], c='r', s=1, alpha=0.5, label='Ions')
ax.legend()
plt.show()
```

## Known Limitations

1. **Simplified Current Deposition**: Uses basic interpolation rather than charge-conserving Esirkepov method
2. **No Collision Physics**: Pure electromagnetic interactions only
3. **Fixed Time Step**: No adaptive time stepping
4. **No Field Boundary Conditions**: Only periodic boundaries implemented
5. **No Relativistic Corrections**: Boris pusher is non-relativistic approximation

## Potential Improvements

- Implement Esirkepov charge-conserving current deposition
- Add OpenMP/MPI parallelization
- Implement PML absorbing boundary conditions
- Add collision operators
- Include ionization physics
- Implement field/particle diagnostics
- Add HDF5 output for large-scale data

## Physical Units

- Length: meters (m)
- Time: seconds (s)  
- Mass: kilograms (kg)
- Charge: coulombs (C)
- Electric field: V/m
- Magnetic field: Tesla (T)
- Energy: Joules (J)

## References

1. Birdsall, C.K. and Langdon, A.B., "Plasma Physics via Computer Simulation"
2. Hockney, R.W. and Eastwood, J.W., "Computer Simulation Using Particles"
3. Yee, K., "Numerical solution of initial boundary value problems involving Maxwell's equations"
4. Boris, J.P., "Relativistic plasma simulation-optimization of a hybrid code"
