// Stable configuration version of the 3D PIC simulation
// This version uses normalized units for better numerical stability

#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <fstream>
#include <iomanip>

// Normalized units (plasma units)
// Length unit: Debye length
// Time unit: 1/plasma frequency
// Velocity unit: thermal velocity

// Simulation parameters structure
struct SimulationParams {
    // Grid parameters
    int nx, ny, nz;           // Number of grid cells
    double dx, dy, dz;         // Cell sizes (in Debye lengths)
    double dt;                 // Time step (in 1/ωp)
    double dt_max;             // Maximum allowed timestep
    double dt_min;             // Minimum allowed timestep
    
    // Simulation parameters
    int num_timesteps;         // Number of time steps
    int num_particles;         // Number of particles
    
    // Physical parameters (normalized)
    double vth;                // Thermal velocity (normalized to c)
    double wp;                 // Plasma frequency (normalized)
    double c;                  // Speed of light (normalized)
    double cfl_factor;         // CFL safety factor (< 1.0)
    
    SimulationParams() :
        nx(16), ny(16), nz(16),   // Smaller grid for stability
        dx(1.0), dy(1.0), dz(1.0), // Grid spacing = 1 Debye length
        dt(0.01),                  // Initial time step = 0.01/ωp (much smaller)
        dt_max(0.05),              // Maximum timestep (reduced)
        dt_min(0.0001),            // Minimum timestep
        num_timesteps(100),
        num_particles(1000),       // Fewer particles for testing
        vth(0.01),                 // Much lower thermal velocity
        wp(1.0),                   // Normalized plasma frequency
        c(1.0),                    // Speed of light (normalized)
        cfl_factor(0.3) {}         // CFL safety factor for stability (more conservative)
};

// Particle structure
struct Particle {
    double x, y, z;       // Position
    double vx, vy, vz;    // Velocity
    double q;             // Charge (normalized)
    double m;             // Mass (normalized)
    int species;          // Species identifier
    
    Particle() : x(0), y(0), z(0), vx(0), vy(0), vz(0), q(-1.0), m(1.0), species(0) {}
};

// 3D Field array class
class Field3D {
private:
    std::vector<double> data;
    int nx, ny, nz;
    
public:
    Field3D() : nx(0), ny(0), nz(0) {}
    
    Field3D(int nx_, int ny_, int nz_) : nx(nx_), ny(ny_), nz(nz_) {
        data.resize(nx * ny * nz, 0.0);
    }
    
    void resize(int nx_, int ny_, int nz_) {
        nx = nx_; ny = ny_; nz = nz_;
        data.resize(nx * ny * nz, 0.0);
    }
    
    double& operator()(int i, int j, int k) {
        // Apply periodic boundary conditions
        i = ((i % nx) + nx) % nx;
        j = ((j % ny) + ny) % ny;
        k = ((k % nz) + nz) % nz;
        return data[i + nx * (j + ny * k)];
    }
    
    const double& operator()(int i, int j, int k) const {
        // Apply periodic boundary conditions
        i = ((i % nx) + nx) % nx;
        j = ((j % ny) + ny) % ny;
        k = ((k % nz) + nz) % nz;
        return data[i + nx * (j + ny * k)];
    }
    
    void clear() {
        std::fill(data.begin(), data.end(), 0.0);
    }
};

// Main PIC simulation class
class PIC3D {
private:
    SimulationParams params;
    
    // Field components
    Field3D Ex, Ey, Ez;
    Field3D Bx, By, Bz;
    Field3D Jx, Jy, Jz;
    Field3D rho;
    
    // Particles
    std::vector<Particle> particles;
    
    // Random number generator
    std::mt19937 rng;
    std::uniform_real_distribution<double> uniform_dist;
    std::normal_distribution<double> normal_dist;
    
public:
    PIC3D(const SimulationParams& p) : params(p), 
                                        rng(42),  // Fixed seed for reproducibility
                                        uniform_dist(0.0, 1.0),
                                        normal_dist(0.0, 1.0) {
        // Initialize fields
        Ex.resize(params.nx+1, params.ny, params.nz);
        Ey.resize(params.nx, params.ny+1, params.nz);
        Ez.resize(params.nx, params.ny, params.nz+1);
        
        Bx.resize(params.nx, params.ny+1, params.nz+1);
        By.resize(params.nx+1, params.ny, params.nz+1);
        Bz.resize(params.nx+1, params.ny+1, params.nz);
        
        Jx.resize(params.nx+1, params.ny, params.nz);
        Jy.resize(params.nx, params.ny+1, params.nz);
        Jz.resize(params.nx, params.ny, params.nz+1);
        
        rho.resize(params.nx, params.ny, params.nz);
        
        // Initialize particles
        initializeParticles();
        
        // Add small initial perturbation to trigger physics
        addPerturbation();
    }
    
    void initializeParticles() {
        particles.resize(params.num_particles);
        
        double Lx = params.nx * params.dx;
        double Ly = params.ny * params.dy;
        double Lz = params.nz * params.dz;
        
        for (int i = 0; i < params.num_particles; ++i) {
            Particle& p = particles[i];
            
            // Alternate between electrons and ions
            if (i % 2 == 0) {
                // Electron
                p.species = 0;
                p.q = -1.0;  // Normalized charge
                p.m = 1.0;   // Normalized mass
            } else {
                // Ion
                p.species = 1;
                p.q = 1.0;
                p.m = 1836.0;  // Mass ratio
            }
            
            // Random position
            p.x = uniform_dist(rng) * Lx;
            p.y = uniform_dist(rng) * Ly;
            p.z = uniform_dist(rng) * Lz;
            
            // Maxwellian velocity (scaled by species mass)
            double vth_species = params.vth / std::sqrt(p.m);
            p.vx = vth_species * normal_dist(rng);
            p.vy = vth_species * normal_dist(rng);
            p.vz = vth_species * normal_dist(rng);
        }
    }
    
    void addPerturbation() {
        // Add a small sinusoidal perturbation to the electric field
        double amplitude = 0.001;
        for (int i = 0; i <= params.nx; ++i) {
            for (int j = 0; j < params.ny; ++j) {
                for (int k = 0; k < params.nz; ++k) {
                    double x = i * params.dx;
                    Ex(i,j,k) = amplitude * std::sin(2.0 * M_PI * x / (params.nx * params.dx));
                }
            }
        }
    }
    
    void depositCharge() {
        rho.clear();
        
        for (const auto& p : particles) {
            double x_norm = p.x / params.dx;
            double y_norm = p.y / params.dy;
            double z_norm = p.z / params.dz;
            
            int i = static_cast<int>(x_norm);
            int j = static_cast<int>(y_norm);
            int k = static_cast<int>(z_norm);
            
            double fx = x_norm - i;
            double fy = y_norm - j;
            double fz = z_norm - k;
            
            // Weight for each particle (assuming uniform density)
            double weight = p.q * params.nx * params.ny * params.nz / params.num_particles;
            
            // CIC deposition
            rho(i,   j,   k  ) += weight * (1-fx) * (1-fy) * (1-fz);
            rho(i+1, j,   k  ) += weight * fx     * (1-fy) * (1-fz);
            rho(i,   j+1, k  ) += weight * (1-fx) * fy     * (1-fz);
            rho(i+1, j+1, k  ) += weight * fx     * fy     * (1-fz);
            rho(i,   j,   k+1) += weight * (1-fx) * (1-fy) * fz;
            rho(i+1, j,   k+1) += weight * fx     * (1-fy) * fz;
            rho(i,   j+1, k+1) += weight * (1-fx) * fy     * fz;
            rho(i+1, j+1, k+1) += weight * fx     * fy     * fz;
        }
    }
    
    void depositCurrent() {
        Jx.clear();
        Jy.clear();
        Jz.clear();
        
        for (const auto& p : particles) {
            double x_norm = p.x / params.dx;
            double y_norm = p.y / params.dy;
            double z_norm = p.z / params.dz;
            
            int i = static_cast<int>(x_norm);
            int j = static_cast<int>(y_norm);
            int k = static_cast<int>(z_norm);
            
            double fx = x_norm - i;
            double fy = y_norm - j;
            double fz = z_norm - k;
            
            // Current weight
            double weight = p.q * params.nx * params.ny * params.nz / params.num_particles;
            double jx = weight * p.vx;
            double jy = weight * p.vy;
            double jz = weight * p.vz;
            
            // Simplified deposition (not charge-conserving)
            Jx(i,   j,   k  ) += jx * (1-fy) * (1-fz) * 0.5;
            Jx(i+1, j,   k  ) += jx * (1-fy) * (1-fz) * 0.5;
            
            Jy(i,   j,   k  ) += jy * (1-fx) * (1-fz) * 0.5;
            Jy(i,   j+1, k  ) += jy * (1-fx) * (1-fz) * 0.5;
            
            Jz(i,   j,   k  ) += jz * (1-fx) * (1-fy) * 0.5;
            Jz(i,   j,   k+1) += jz * (1-fx) * (1-fy) * 0.5;
        }
    }
    
    void updateEField() {
        double c2dt = params.dt;  // In normalized units
        
        // Update Ex
        for (int i = 0; i <= params.nx; ++i) {
            for (int j = 0; j < params.ny; ++j) {
                for (int k = 0; k < params.nz; ++k) {
                    Ex(i,j,k) += c2dt * (
                        (Bz(i,j+1,k) - Bz(i,j,k)) / params.dy -
                        (By(i,j,k+1) - By(i,j,k)) / params.dz -
                        Jx(i,j,k)
                    );
                }
            }
        }
        
        // Update Ey
        for (int i = 0; i < params.nx; ++i) {
            for (int j = 0; j <= params.ny; ++j) {
                for (int k = 0; k < params.nz; ++k) {
                    Ey(i,j,k) += c2dt * (
                        (Bx(i,j,k+1) - Bx(i,j,k)) / params.dz -
                        (Bz(i+1,j,k) - Bz(i,j,k)) / params.dx -
                        Jy(i,j,k)
                    );
                }
            }
        }
        
        // Update Ez
        for (int i = 0; i < params.nx; ++i) {
            for (int j = 0; j < params.ny; ++j) {
                for (int k = 0; k <= params.nz; ++k) {
                    Ez(i,j,k) += c2dt * (
                        (By(i+1,j,k) - By(i,j,k)) / params.dx -
                        (Bx(i,j+1,k) - Bx(i,j,k)) / params.dy -
                        Jz(i,j,k)
                    );
                }
            }
        }
    }
    
    void updateBField() {
        // Update Bx
        for (int i = 0; i < params.nx; ++i) {
            for (int j = 0; j <= params.ny; ++j) {
                for (int k = 0; k <= params.nz; ++k) {
                    Bx(i,j,k) -= params.dt * (
                        (Ez(i,j,k) - Ez(i,j-1,k)) / params.dy -
                        (Ey(i,j,k) - Ey(i,j,k-1)) / params.dz
                    );
                }
            }
        }
        
        // Update By
        for (int i = 0; i <= params.nx; ++i) {
            for (int j = 0; j < params.ny; ++j) {
                for (int k = 0; k <= params.nz; ++k) {
                    By(i,j,k) -= params.dt * (
                        (Ex(i,j,k) - Ex(i,j,k-1)) / params.dz -
                        (Ez(i,j,k) - Ez(i-1,j,k)) / params.dx
                    );
                }
            }
        }
        
        // Update Bz
        for (int i = 0; i <= params.nx; ++i) {
            for (int j = 0; j <= params.ny; ++j) {
                for (int k = 0; k < params.nz; ++k) {
                    Bz(i,j,k) -= params.dt * (
                        (Ey(i,j,k) - Ey(i-1,j,k)) / params.dx -
                        (Ex(i,j,k) - Ex(i,j-1,k)) / params.dy
                    );
                }
            }
        }
    }
    
    void interpolateFields(const Particle& p, double& Ex_p, double& Ey_p, double& Ez_p,
                          double& Bx_p, double& By_p, double& Bz_p) {
        double x_norm = p.x / params.dx;
        double y_norm = p.y / params.dy;
        double z_norm = p.z / params.dz;
        
        // Proper CIC (Cloud-In-Cell) interpolation for fields
        int i = static_cast<int>(x_norm);
        int j = static_cast<int>(y_norm);
        int k = static_cast<int>(z_norm);
        
        double fx = x_norm - i;
        double fy = y_norm - j;
        double fz = z_norm - k;
        
        // Interpolate E field (staggered grid)
        // Ex is on (i+1/2, j, k) faces
        Ex_p = Ex(i,j,k) * (1-fy) * (1-fz)
             + Ex(i,j+1,k) * fy * (1-fz)
             + Ex(i,j,k+1) * (1-fy) * fz
             + Ex(i,j+1,k+1) * fy * fz;
        
        // Ey is on (i, j+1/2, k) faces
        Ey_p = Ey(i,j,k) * (1-fx) * (1-fz)
             + Ey(i+1,j,k) * fx * (1-fz)
             + Ey(i,j,k+1) * (1-fx) * fz
             + Ey(i+1,j,k+1) * fx * fz;
        
        // Ez is on (i, j, k+1/2) faces
        Ez_p = Ez(i,j,k) * (1-fx) * (1-fy)
             + Ez(i+1,j,k) * fx * (1-fy)
             + Ez(i,j+1,k) * (1-fx) * fy
             + Ez(i+1,j+1,k) * fx * fy;
        
        // Interpolate B field (also staggered)
        // Bx is on (i, j+1/2, k+1/2) edges
        Bx_p = Bx(i,j,k) * (1-fy) * (1-fz)
             + Bx(i,j+1,k) * fy * (1-fz)
             + Bx(i,j,k+1) * (1-fy) * fz
             + Bx(i,j+1,k+1) * fy * fz;
        
        // By is on (i+1/2, j, k+1/2) edges
        By_p = By(i,j,k) * (1-fx) * (1-fz)
             + By(i+1,j,k) * fx * (1-fz)
             + By(i,j,k+1) * (1-fx) * fz
             + By(i+1,j,k+1) * fx * fz;
        
        // Bz is on (i+1/2, j+1/2, k) edges
        Bz_p = Bz(i,j,k) * (1-fx) * (1-fy)
             + Bz(i+1,j,k) * fx * (1-fy)
             + Bz(i,j+1,k) * (1-fx) * fy
             + Bz(i+1,j+1,k) * fx * fy;
    }
    
    void pushParticles() {
        double Lx = params.nx * params.dx;
        double Ly = params.ny * params.dy;
        double Lz = params.nz * params.dz;
        
        for (auto& p : particles) {
            double Ex_p, Ey_p, Ez_p, Bx_p, By_p, Bz_p;
            interpolateFields(p, Ex_p, Ey_p, Ez_p, Bx_p, By_p, Bz_p);
            
            // Boris algorithm (normalized units)
            double qdt_2m = p.q * params.dt / (2.0 * p.m);
            
            // Half acceleration from E
            double vx_minus = p.vx + qdt_2m * Ex_p;
            double vy_minus = p.vy + qdt_2m * Ey_p;
            double vz_minus = p.vz + qdt_2m * Ez_p;
            
            // Rotation from B
            double tx = qdt_2m * Bx_p;
            double ty = qdt_2m * By_p;
            double tz = qdt_2m * Bz_p;
            
            double t_mag2 = tx*tx + ty*ty + tz*tz;
            double sx = 2.0 * tx / (1.0 + t_mag2);
            double sy = 2.0 * ty / (1.0 + t_mag2);
            double sz = 2.0 * tz / (1.0 + t_mag2);
            
            double vx_prime = vx_minus + vy_minus * tz - vz_minus * ty;
            double vy_prime = vy_minus + vz_minus * tx - vx_minus * tz;
            double vz_prime = vz_minus + vx_minus * ty - vy_minus * tx;
            
            double vx_plus = vx_minus + vy_prime * sz - vz_prime * sy;
            double vy_plus = vy_minus + vz_prime * sx - vx_prime * sz;
            double vz_plus = vz_minus + vx_prime * sy - vy_prime * sx;
            
            // Final half acceleration
            p.vx = vx_plus + qdt_2m * Ex_p;
            p.vy = vy_plus + qdt_2m * Ey_p;
            p.vz = vz_plus + qdt_2m * Ez_p;
            
            // Update position
            p.x += p.vx * params.dt;
            p.y += p.vy * params.dt;
            p.z += p.vz * params.dt;
            
            // Periodic boundaries
            while (p.x < 0) p.x += Lx;
            while (p.x >= Lx) p.x -= Lx;
            while (p.y < 0) p.y += Ly;
            while (p.y >= Ly) p.y -= Ly;
            while (p.z < 0) p.z += Lz;
            while (p.z >= Lz) p.z -= Lz;
        }
    }
    
    double calculateTotalEnergy() {
        double energy = 0.0;
        
        // Field energy
        for (int i = 0; i <= params.nx; ++i) {
            for (int j = 0; j < params.ny; ++j) {
                for (int k = 0; k < params.nz; ++k) {
                    energy += 0.5 * Ex(i,j,k) * Ex(i,j,k);
                }
            }
        }
        
        for (int i = 0; i < params.nx; ++i) {
            for (int j = 0; j <= params.ny; ++j) {
                for (int k = 0; k < params.nz; ++k) {
                    energy += 0.5 * Ey(i,j,k) * Ey(i,j,k);
                }
            }
        }
        
        for (int i = 0; i < params.nx; ++i) {
            for (int j = 0; j < params.ny; ++j) {
                for (int k = 0; k <= params.nz; ++k) {
                    energy += 0.5 * Ez(i,j,k) * Ez(i,j,k);
                }
            }
        }
        
        // Magnetic energy
        for (int i = 0; i < params.nx; ++i) {
            for (int j = 0; j <= params.ny; ++j) {
                for (int k = 0; k <= params.nz; ++k) {
                    energy += 0.5 * Bx(i,j,k) * Bx(i,j,k);
                }
            }
        }
        
        for (int i = 0; i <= params.nx; ++i) {
            for (int j = 0; j < params.ny; ++j) {
                for (int k = 0; k <= params.nz; ++k) {
                    energy += 0.5 * By(i,j,k) * By(i,j,k);
                }
            }
        }
        
        for (int i = 0; i <= params.nx; ++i) {
            for (int j = 0; j <= params.ny; ++j) {
                for (int k = 0; k < params.nz; ++k) {
                    energy += 0.5 * Bz(i,j,k) * Bz(i,j,k);
                }
            }
        }
        
        // Particle kinetic energy
        for (const auto& p : particles) {
            double v2 = p.vx*p.vx + p.vy*p.vy + p.vz*p.vz;
            energy += 0.5 * p.m * v2;
        }
        
        return energy;
    }
    
    double calculateMaxParticleVelocity() {
        double v_max = 0.0;
        for (const auto& p : particles) {
            double v = std::sqrt(p.vx*p.vx + p.vy*p.vy + p.vz*p.vz);
            if (v > v_max) {
                v_max = v;
            }
        }
        return v_max;
    }
    
    double calculateMaxFieldStrength() {
        double e_max = 0.0;
        double b_max = 0.0;
        
        // Check E field maximum
        for (int i = 0; i <= params.nx; ++i) {
            for (int j = 0; j < params.ny; ++j) {
                for (int k = 0; k < params.nz; ++k) {
                    double e = std::abs(Ex(i,j,k));
                    if (e > e_max) e_max = e;
                }
            }
        }
        
        for (int i = 0; i < params.nx; ++i) {
            for (int j = 0; j <= params.ny; ++j) {
                for (int k = 0; k < params.nz; ++k) {
                    double e = std::abs(Ey(i,j,k));
                    if (e > e_max) e_max = e;
                }
            }
        }
        
        for (int i = 0; i < params.nx; ++i) {
            for (int j = 0; j < params.ny; ++j) {
                for (int k = 0; k <= params.nz; ++k) {
                    double e = std::abs(Ez(i,j,k));
                    if (e > e_max) e_max = e;
                }
            }
        }
        
        // Check B field maximum
        for (int i = 0; i < params.nx; ++i) {
            for (int j = 0; j <= params.ny; ++j) {
                for (int k = 0; k <= params.nz; ++k) {
                    double b = std::abs(Bx(i,j,k));
                    if (b > b_max) b_max = b;
                }
            }
        }
        
        for (int i = 0; i <= params.nx; ++i) {
            for (int j = 0; j < params.ny; ++j) {
                for (int k = 0; k <= params.nz; ++k) {
                    double b = std::abs(By(i,j,k));
                    if (b > b_max) b_max = b;
                }
            }
        }
        
        for (int i = 0; i <= params.nx; ++i) {
            for (int j = 0; j <= params.ny; ++j) {
                for (int k = 0; k < params.nz; ++k) {
                    double b = std::abs(Bz(i,j,k));
                    if (b > b_max) b_max = b;
                }
            }
        }
        
        return std::max(e_max, b_max);
    }
    
    double calculateCFLTimestep() {
        // CFL condition for electromagnetic waves: dt < dx/c * 1/sqrt(3) for 3D
        double dx_min = std::min({params.dx, params.dy, params.dz});
        double dt_em = params.cfl_factor * dx_min / (params.c * std::sqrt(3.0));
        
        // CFL condition for particle motion: dt < dx/v_max
        double v_max = calculateMaxParticleVelocity();
        double dt_particle = std::numeric_limits<double>::max();
        if (v_max > 1e-10) {  // Avoid division by zero
            dt_particle = params.cfl_factor * dx_min / v_max;
        }
        
        // Constraint based on plasma frequency (need to resolve plasma oscillations)
        double dt_plasma = params.cfl_factor * 1.0 / params.wp;  // More conservative
        
        // Constraint based on cyclotron frequency (if B field present)
        double b_max = calculateMaxFieldStrength();
        double dt_cyclotron = std::numeric_limits<double>::max();
        if (b_max > 1e-10) {  // Avoid division by zero
            // Electron cyclotron frequency: ωc = eB/m
            double omega_c = b_max;  // In normalized units
            dt_cyclotron = params.cfl_factor * 1.0 / omega_c;  // More conservative
        }
        
        // For explicit PIC, also need dt < ωp^-1 for stability
        double dt_pic_stability = 0.1 / params.wp;  // Very conservative for explicit scheme
        
        // Take minimum of all constraints
        double dt_new = std::min({dt_em, dt_particle, dt_plasma, dt_cyclotron, dt_pic_stability});
        
        // Apply limits
        dt_new = std::max(dt_new, params.dt_min);
        dt_new = std::min(dt_new, params.dt_max);
        
        return dt_new;
    }
    
    void adjustTimestep() {
        double dt_old = params.dt;
        params.dt = calculateCFLTimestep();
        
        if (std::abs(params.dt - dt_old) / dt_old > 0.1) {
            std::cout << "  [CFL] Timestep adjusted: " << dt_old << " -> " << params.dt << "\n";
        }
    }
    
    void run() {
        std::cout << "Starting Stable 3D PIC Simulation with CFL Control\n";
        std::cout << "Grid: " << params.nx << "x" << params.ny << "x" << params.nz << "\n";
        std::cout << "Particles: " << params.num_particles << "\n";
        std::cout << "Time steps: " << params.num_timesteps << "\n";
        std::cout << "Initial dt = " << params.dt << " (1/ωp)\n";
        std::cout << "CFL safety factor = " << params.cfl_factor << "\n\n";
        
        // Calculate initial CFL timestep
        params.dt = calculateCFLTimestep();
        std::cout << "CFL-adjusted dt = " << params.dt << " (1/ωp)\n\n";
        
        std::ofstream outfile("pic3d_stable_diagnostics.txt");
        outfile << "# Step, Time, dt, Total_Energy, Energy_Error(%), Max_Velocity\n";
        
        double initial_energy = calculateTotalEnergy();
        double simulation_time = 0.0;
        
        for (int step = 0; step < params.num_timesteps; ++step) {
            // Adjust timestep based on CFL condition
            adjustTimestep();
            
            depositCurrent();
            updateBField();
            updateEField();
            updateBField();
            pushParticles();
            
            simulation_time += params.dt;
            
            if (step % 10 == 0) {
                double energy = calculateTotalEnergy();
                double energy_error = (energy - initial_energy) / initial_energy;
                double v_max = calculateMaxParticleVelocity();
                
                std::cout << "Step " << step << "/" << params.num_timesteps
                         << " | Time: " << std::fixed << std::setprecision(3) << simulation_time
                         << " | dt: " << std::scientific << std::setprecision(2) << params.dt
                         << " | Energy: " << std::fixed << std::setprecision(6) << energy
                         << " | Error: " << std::setprecision(2) << energy_error * 100 << "%"
                         << " | v_max: " << std::scientific << v_max << "\n";
                
                outfile << step << " " << simulation_time << " " << params.dt << " "
                       << energy << " " << energy_error * 100 << " " << v_max << "\n";
            }
        }
        
        outfile.close();
        saveParticlePositions("particles_stable_final.txt");
        std::cout << "\nSimulation complete.\n";
        std::cout << "Final simulation time: " << simulation_time << " (1/ωp)\n";
    }
    
    void saveParticlePositions(const std::string& filename) {
        std::ofstream outfile(filename);
        outfile << "# x, y, z, vx, vy, vz, species\n";
        
        for (const auto& p : particles) {
            outfile << p.x << " " << p.y << " " << p.z << " "
                   << p.vx << " " << p.vy << " " << p.vz << " "
                   << p.species << "\n";
        }
        
        outfile.close();
    }
};

int main() {
    SimulationParams params;
    PIC3D simulation(params);
    simulation.run();
    return 0;
}