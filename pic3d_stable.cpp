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
        dt(0.005),                 // Initial time step = 0.005/ωp (very small)
        dt_max(0.01),              // Maximum timestep (very conservative)
        dt_min(0.0001),            // Minimum timestep
        num_timesteps(200),        // More timesteps for shorter dt
        num_particles(1000),       // Fewer particles for testing
        vth(0.01),                 // Much lower thermal velocity
        wp(1.0),                   // Normalized plasma frequency
        c(1.0),                    // Speed of light (normalized)
        cfl_factor(0.2) {}         // CFL safety factor for stability (very conservative)
};

// Particle structure
struct Particle {
    double x, y, z;       // Position
    double vx, vy, vz;    // Velocity
    double q;             // Charge (normalized)
    double m;             // Mass (normalized)
    int species;          // Species identifier
    double x_old, y_old, z_old;  // Previous position for current deposition
    
    Particle() : x(0), y(0), z(0), vx(0), vy(0), vz(0), q(-1.0), m(1.0), species(0),
                 x_old(0), y_old(0), z_old(0) {}
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
            
            // Initialize old positions
            p.x_old = p.x;
            p.y_old = p.y;
            p.z_old = p.z;
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
        // Improved current deposition with CIC weighting and momentum conservation
        Jx.clear();
        Jy.clear();
        Jz.clear();
        
        for (const auto& p : particles) {
            // Use average position for current deposition (momentum-conserving)
            double x_avg = 0.5 * (p.x + p.x_old);
            double y_avg = 0.5 * (p.y + p.y_old);
            double z_avg = 0.5 * (p.z + p.z_old);
            
            double x_norm = x_avg / params.dx;
            double y_norm = y_avg / params.dy;
            double z_norm = z_avg / params.dz;
            
            int i = static_cast<int>(x_norm);
            int j = static_cast<int>(y_norm);
            int k = static_cast<int>(z_norm);
            
            double fx = x_norm - i;
            double fy = y_norm - j;
            double fz = z_norm - k;
            
            // Current weight
            double weight = p.q * params.nx * params.ny * params.nz / params.num_particles;
            
            // Current components (using velocity at n+1/2)
            double jx = weight * p.vx;
            double jy = weight * p.vy;
            double jz = weight * p.vz;
            
            // CIC deposition for Jx (on x-faces)
            Jx(i,   j,   k  ) += jx * (1-fx) * (1-fy) * (1-fz);
            Jx(i+1, j,   k  ) += jx * fx     * (1-fy) * (1-fz);
            Jx(i,   j+1, k  ) += jx * (1-fx) * fy     * (1-fz);
            Jx(i+1, j+1, k  ) += jx * fx     * fy     * (1-fz);
            Jx(i,   j,   k+1) += jx * (1-fx) * (1-fy) * fz;
            Jx(i+1, j,   k+1) += jx * fx     * (1-fy) * fz;
            Jx(i,   j+1, k+1) += jx * (1-fx) * fy     * fz;
            Jx(i+1, j+1, k+1) += jx * fx     * fy     * fz;
            
            // CIC deposition for Jy (on y-faces)
            Jy(i,   j,   k  ) += jy * (1-fx) * (1-fy) * (1-fz);
            Jy(i+1, j,   k  ) += jy * fx     * (1-fy) * (1-fz);
            Jy(i,   j+1, k  ) += jy * (1-fx) * fy     * (1-fz);
            Jy(i+1, j+1, k  ) += jy * fx     * fy     * (1-fz);
            Jy(i,   j,   k+1) += jy * (1-fx) * (1-fy) * fz;
            Jy(i+1, j,   k+1) += jy * fx     * (1-fy) * fz;
            Jy(i,   j+1, k+1) += jy * (1-fx) * fy     * fz;
            Jy(i+1, j+1, k+1) += jy * fx     * fy     * fz;
            
            // CIC deposition for Jz (on z-faces)
            Jz(i,   j,   k  ) += jz * (1-fx) * (1-fy) * (1-fz);
            Jz(i+1, j,   k  ) += jz * fx     * (1-fy) * (1-fz);
            Jz(i,   j+1, k  ) += jz * (1-fx) * fy     * (1-fz);
            Jz(i+1, j+1, k  ) += jz * fx     * fy     * (1-fz);
            Jz(i,   j,   k+1) += jz * (1-fx) * (1-fy) * fz;
            Jz(i+1, j,   k+1) += jz * fx     * (1-fy) * fz;
            Jz(i,   j+1, k+1) += jz * (1-fx) * fy     * fz;
            Jz(i+1, j+1, k+1) += jz * fx     * fy     * fz;
        }
        
        // Apply smoothing filter to reduce high-frequency noise
        smoothCurrents();
    }
    
    void smoothCurrents() {
        // Simple 3-point smoothing filter for stability
        Field3D Jx_temp = Jx;
        Field3D Jy_temp = Jy;
        Field3D Jz_temp = Jz;
        
        double alpha = 0.25;  // Smoothing strength
        
        // Smooth Jx
        for (int i = 1; i < params.nx; ++i) {
            for (int j = 1; j < params.ny-1; ++j) {
                for (int k = 1; k < params.nz-1; ++k) {
                    Jx(i,j,k) = (1-2*alpha) * Jx_temp(i,j,k) +
                               alpha * (Jx_temp(i,j+1,k) + Jx_temp(i,j-1,k));
                }
            }
        }
        
        // Smooth Jy
        for (int i = 1; i < params.nx-1; ++i) {
            for (int j = 1; j < params.ny; ++j) {
                for (int k = 1; k < params.nz-1; ++k) {
                    Jy(i,j,k) = (1-2*alpha) * Jy_temp(i,j,k) +
                               alpha * (Jy_temp(i+1,j,k) + Jy_temp(i-1,j,k));
                }
            }
        }
        
        // Smooth Jz
        for (int i = 1; i < params.nx-1; ++i) {
            for (int j = 1; j < params.ny-1; ++j) {
                for (int k = 1; k < params.nz; ++k) {
                    Jz(i,j,k) = (1-2*alpha) * Jz_temp(i,j,k) +
                               alpha * (Jz_temp(i+1,j,k) + Jz_temp(i-1,j,k));
                }
            }
        }
    }
    
    void updateEField() {
        // Energy-conserving E field update using Faraday's law
        // E^{n+1} = E^n + c^2 dt (∇ × B^{n+1/2} - J^{n+1/2})
        double c2dt = params.dt * params.c * params.c;  // c^2 * dt
        
        // Update Ex on (i+1/2, j, k) edges
        for (int i = 0; i <= params.nx; ++i) {
            for (int j = 0; j < params.ny; ++j) {
                for (int k = 0; k < params.nz; ++k) {
                    // Curl of B using centered differences
                    double curl_Bx = (Bz(i,j+1,k) - Bz(i,j,k)) / params.dy -
                                     (By(i,j,k+1) - By(i,j,k)) / params.dz;
                    Ex(i,j,k) += c2dt * (curl_Bx - Jx(i,j,k));
                }
            }
        }
        
        // Update Ey on (i, j+1/2, k) edges
        for (int i = 0; i < params.nx; ++i) {
            for (int j = 0; j <= params.ny; ++j) {
                for (int k = 0; k < params.nz; ++k) {
                    // Curl of B using centered differences
                    double curl_By = (Bx(i,j,k+1) - Bx(i,j,k)) / params.dz -
                                     (Bz(i+1,j,k) - Bz(i,j,k)) / params.dx;
                    Ey(i,j,k) += c2dt * (curl_By - Jy(i,j,k));
                }
            }
        }
        
        // Update Ez on (i, j, k+1/2) edges
        for (int i = 0; i < params.nx; ++i) {
            for (int j = 0; j < params.ny; ++j) {
                for (int k = 0; k <= params.nz; ++k) {
                    // Curl of B using centered differences
                    double curl_Bz = (By(i+1,j,k) - By(i,j,k)) / params.dx -
                                     (Bx(i,j+1,k) - Bx(i,j,k)) / params.dy;
                    Ez(i,j,k) += c2dt * (curl_Bz - Jz(i,j,k));
                }
            }
        }
    }
    
    void updateBFieldHalfStep() {
        // Energy-conserving B field half-step update using Ampere's law
        // B^{n+1/2} = B^{n-1/2} - dt/2 ∇ × E^n
        double dt_half = 0.5 * params.dt;
        
        // Update Bx on (i, j+1/2, k+1/2) faces
        for (int i = 0; i < params.nx; ++i) {
            for (int j = 0; j <= params.ny; ++j) {
                for (int k = 0; k <= params.nz; ++k) {
                    // Use boundary-safe indexing
                    int jm = (j > 0) ? j - 1 : params.ny - 1;
                    int km = (k > 0) ? k - 1 : params.nz - 1;
                    
                    // Curl of E using centered differences
                    double curl_Ex = (Ez(i,j,k) - Ez(i,jm,k)) / params.dy -
                                     (Ey(i,j,k) - Ey(i,j,km)) / params.dz;
                    Bx(i,j,k) -= dt_half * curl_Ex;
                }
            }
        }
        
        // Update By on (i+1/2, j, k+1/2) faces
        for (int i = 0; i <= params.nx; ++i) {
            for (int j = 0; j < params.ny; ++j) {
                for (int k = 0; k <= params.nz; ++k) {
                    // Use boundary-safe indexing
                    int im = (i > 0) ? i - 1 : params.nx - 1;
                    int km = (k > 0) ? k - 1 : params.nz - 1;
                    
                    // Curl of E using centered differences
                    double curl_Ey = (Ex(i,j,k) - Ex(i,j,km)) / params.dz -
                                     (Ez(i,j,k) - Ez(im,j,k)) / params.dx;
                    By(i,j,k) -= dt_half * curl_Ey;
                }
            }
        }
        
        // Update Bz on (i+1/2, j+1/2, k) faces
        for (int i = 0; i <= params.nx; ++i) {
            for (int j = 0; j <= params.ny; ++j) {
                for (int k = 0; k < params.nz; ++k) {
                    // Use boundary-safe indexing
                    int im = (i > 0) ? i - 1 : params.nx - 1;
                    int jm = (j > 0) ? j - 1 : params.ny - 1;
                    
                    // Curl of E using centered differences
                    double curl_Ez = (Ey(i,j,k) - Ey(im,j,k)) / params.dx -
                                     (Ex(i,j,k) - Ex(i,jm,k)) / params.dy;
                    Bz(i,j,k) -= dt_half * curl_Ez;
                }
            }
        }
    }
    
    void updateBField() {
        // Full step B field update (two half steps)
        updateBFieldHalfStep();
        updateBFieldHalfStep();
    }
    
    double calculateDivergenceB(int i, int j, int k) {
        // Calculate ∇·B at cell center (i,j,k)
        double divB = (Bx(i+1,j,k) - Bx(i,j,k)) / params.dx +
                      (By(i,j+1,k) - By(i,j,k)) / params.dy +
                      (Bz(i,j,k+1) - Bz(i,j,k)) / params.dz;
        return divB;
    }
    
    double calculateDivergenceE(int i, int j, int k) {
        // Calculate ∇·E at cell center (i,j,k)
        double divE = (Ex(i+1,j,k) - Ex(i,j,k)) / params.dx +
                      (Ey(i,j+1,k) - Ey(i,j,k)) / params.dy +
                      (Ez(i,j,k+1) - Ez(i,j,k)) / params.dz;
        return divE;
    }
    
    void cleanDivergenceB() {
        // Project B field to ensure ∇·B = 0
        // Using a simple iterative projection method (Helmholtz decomposition)
        
        Field3D phi(params.nx, params.ny, params.nz);  // Scalar potential
        Field3D phi_new(params.nx, params.ny, params.nz);
        
        // Calculate divergence errors
        for (int i = 0; i < params.nx; ++i) {
            for (int j = 0; j < params.ny; ++j) {
                for (int k = 0; k < params.nz; ++k) {
                    phi(i,j,k) = calculateDivergenceB(i, j, k);
                }
            }
        }
        
        // Solve Poisson equation ∇²φ = ∇·B using Jacobi iteration
        double h2 = params.dx * params.dx;  // Assuming dx=dy=dz
        double omega = 1.0;  // Relaxation parameter
        
        for (int iter = 0; iter < 20; ++iter) {
            for (int i = 0; i < params.nx; ++i) {
                for (int j = 0; j < params.ny; ++j) {
                    for (int k = 0; k < params.nz; ++k) {
                        double sum = phi(i+1,j,k) + phi(i-1,j,k) +
                                    phi(i,j+1,k) + phi(i,j-1,k) +
                                    phi(i,j,k+1) + phi(i,j,k-1);
                        phi_new(i,j,k) = (1-omega) * phi(i,j,k) +
                                        omega * (sum - h2 * calculateDivergenceB(i,j,k)) / 6.0;
                    }
                }
            }
            std::swap(phi, phi_new);
        }
        
        // Correct B field: B_corrected = B - ∇φ
        for (int i = 0; i < params.nx; ++i) {
            for (int j = 0; j <= params.ny; ++j) {
                for (int k = 0; k <= params.nz; ++k) {
                    // Bx correction using gradient in x direction
                    if (i > 0) {
                        Bx(i,j,k) -= 0.5 * (phi(i,j,k) - phi(i-1,j,k)) / params.dx;
                    }
                }
            }
        }
        
        for (int i = 0; i <= params.nx; ++i) {
            for (int j = 0; j < params.ny; ++j) {
                for (int k = 0; k <= params.nz; ++k) {
                    // By correction using gradient in y direction
                    if (j > 0) {
                        By(i,j,k) -= 0.5 * (phi(i,j,k) - phi(i,j-1,k)) / params.dy;
                    }
                }
            }
        }
        
        for (int i = 0; i <= params.nx; ++i) {
            for (int j = 0; j <= params.ny; ++j) {
                for (int k = 0; k < params.nz; ++k) {
                    // Bz correction using gradient in z direction
                    if (k > 0) {
                        Bz(i,j,k) -= 0.5 * (phi(i,j,k) - phi(i,j,k-1)) / params.dz;
                    }
                }
            }
        }
    }
    
    void cleanDivergenceE() {
        // Correct E field to satisfy Gauss's law: ∇·E = ρ/ε₀
        // In normalized units: ∇·E = ρ
        
        Field3D phi(params.nx, params.ny, params.nz);  // Scalar potential
        Field3D phi_new(params.nx, params.ny, params.nz);
        Field3D divE_error(params.nx, params.ny, params.nz);
        
        // Calculate divergence error: ∇·E - ρ
        for (int i = 0; i < params.nx; ++i) {
            for (int j = 0; j < params.ny; ++j) {
                for (int k = 0; k < params.nz; ++k) {
                    double divE = calculateDivergenceE(i, j, k);
                    divE_error(i,j,k) = divE - rho(i,j,k);
                    phi(i,j,k) = divE_error(i,j,k);
                }
            }
        }
        
        // Solve Poisson equation ∇²φ = (∇·E - ρ) using Jacobi iteration
        double h2 = params.dx * params.dx;  // Assuming dx=dy=dz
        double omega = 1.0;  // Relaxation parameter
        
        for (int iter = 0; iter < 20; ++iter) {
            for (int i = 0; i < params.nx; ++i) {
                for (int j = 0; j < params.ny; ++j) {
                    for (int k = 0; k < params.nz; ++k) {
                        double sum = phi(i+1,j,k) + phi(i-1,j,k) +
                                    phi(i,j+1,k) + phi(i,j-1,k) +
                                    phi(i,j,k+1) + phi(i,j,k-1);
                        phi_new(i,j,k) = (1-omega) * phi(i,j,k) +
                                        omega * (sum - h2 * divE_error(i,j,k)) / 6.0;
                    }
                }
            }
            std::swap(phi, phi_new);
        }
        
        // Correct E field: E_corrected = E - ∇φ
        for (int i = 0; i <= params.nx; ++i) {
            for (int j = 0; j < params.ny; ++j) {
                for (int k = 0; k < params.nz; ++k) {
                    // Ex correction
                    if (i > 0 && i < params.nx) {
                        Ex(i,j,k) -= 0.5 * (phi(i,j,k) - phi(i-1,j,k)) / params.dx;
                    }
                }
            }
        }
        
        for (int i = 0; i < params.nx; ++i) {
            for (int j = 0; j <= params.ny; ++j) {
                for (int k = 0; k < params.nz; ++k) {
                    // Ey correction
                    if (j > 0 && j < params.ny) {
                        Ey(i,j,k) -= 0.5 * (phi(i,j,k) - phi(i,j-1,k)) / params.dy;
                    }
                }
            }
        }
        
        for (int i = 0; i < params.nx; ++i) {
            for (int j = 0; j < params.ny; ++j) {
                for (int k = 0; k <= params.nz; ++k) {
                    // Ez correction
                    if (k > 0 && k < params.nz) {
                        Ez(i,j,k) -= 0.5 * (phi(i,j,k) - phi(i,j,k-1)) / params.dz;
                    }
                }
            }
        }
    }
    
    double calculateMaxDivergenceErrors() {
        double max_divB = 0.0;
        double max_divE_error = 0.0;
        
        for (int i = 1; i < params.nx-1; ++i) {
            for (int j = 1; j < params.ny-1; ++j) {
                for (int k = 1; k < params.nz-1; ++k) {
                    double divB = std::abs(calculateDivergenceB(i, j, k));
                    double divE = calculateDivergenceE(i, j, k);
                    double divE_error = std::abs(divE - rho(i,j,k));
                    
                    max_divB = std::max(max_divB, divB);
                    max_divE_error = std::max(max_divE_error, divE_error);
                }
            }
        }
        
        return std::max(max_divB, max_divE_error);
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
            
            // Store old position for current deposition
            p.x_old = p.x;
            p.y_old = p.y;
            p.z_old = p.z;
            
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
        outfile << "# Step, Time, dt, Total_Energy, Energy_Error(%), Max_Velocity, Max_DivError\n";
        
        double initial_energy = calculateTotalEnergy();
        double simulation_time = 0.0;
        
        for (int step = 0; step < params.num_timesteps; ++step) {
            // Adjust timestep based on CFL condition
            adjustTimestep();
            
            // Symplectic leapfrog integration for energy conservation:
            // 1. Update B by half timestep: B^{n-1/2} -> B^{n}
            updateBFieldHalfStep();
            
            // 2. Push particles: x^n -> x^{n+1}, v^{n-1/2} -> v^{n+1/2}
            pushParticles();
            
            // 3. Deposit current at time n+1/2 using v^{n+1/2}
            depositCurrent();
            
            // 4. Update E field: E^n -> E^{n+1}
            updateEField();
            
            // 5. Update B by half timestep: B^n -> B^{n+1/2}
            updateBFieldHalfStep();
            
            simulation_time += params.dt;
            
            if (step % 10 == 0) {
                double energy = calculateTotalEnergy();
                double energy_error = (energy - initial_energy) / initial_energy;
                double v_max = calculateMaxParticleVelocity();
                double max_div_error = calculateMaxDivergenceErrors();
                
                std::cout << "Step " << step << "/" << params.num_timesteps
                         << " | Time: " << std::fixed << std::setprecision(3) << simulation_time
                         << " | dt: " << std::scientific << std::setprecision(2) << params.dt
                         << " | Energy: " << std::fixed << std::setprecision(6) << energy
                         << " | Error: " << std::setprecision(2) << energy_error * 100 << "%"
                         << " | v_max: " << std::scientific << v_max
                         << " | div_err: " << max_div_error << "\n";
                
                outfile << step << " " << simulation_time << " " << params.dt << " "
                       << energy << " " << energy_error * 100 << " " << v_max << " "
                       << max_div_error << "\n";
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