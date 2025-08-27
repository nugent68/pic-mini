// 3D Electromagnetic Particle-in-Cell Simulation
// Uses Yee's method for field updates and Boris method for particle pushing
// Implements periodic boundary conditions in all directions

#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <fstream>
#include <iomanip>

// Physical constants
const double C = 299792458.0;        // Speed of light (m/s)
const double EPS0 = 8.854187817e-12; // Vacuum permittivity (F/m)
const double MU0 = 1.256637061e-6;   // Vacuum permeability (H/m)
const double QE = 1.602176634e-19;   // Elementary charge (C)
const double ME = 9.109383701e-31;   // Electron mass (kg)
const double MP = 1.672621923e-27;   // Proton mass (kg)

// Simulation parameters structure
struct SimulationParams {
    // Grid parameters
    int nx, ny, nz;           // Number of grid cells
    double dx, dy, dz;         // Cell sizes
    double dt;                 // Time step
    
    // Simulation parameters
    int num_timesteps;         // Number of time steps
    int num_particles;         // Number of particles
    
    // Physical parameters
    double plasma_frequency;   // Plasma frequency
    double debye_length;       // Debye length
    
    SimulationParams() : 
        nx(32), ny(32), nz(32),
        dx(1e-6), dy(1e-6), dz(1e-6),
        dt(0.0),  // Will be set based on CFL condition
        num_timesteps(1000),
        num_particles(10000),
        plasma_frequency(1e9),
        debye_length(1e-6) {
        // Set dt based on CFL condition
        double min_dx = std::min({dx, dy, dz});
        dt = 0.5 * min_dx / C;  // CFL factor of 0.5 for stability
    }
};

// Particle structure
struct Particle {
    double x, y, z;       // Position
    double vx, vy, vz;    // Velocity
    double q;             // Charge
    double m;             // Mass
    int species;          // Species identifier (0: electron, 1: ion)
    
    Particle() : x(0), y(0), z(0), vx(0), vy(0), vz(0), q(-QE), m(ME), species(0) {}
    
    Particle(double x_, double y_, double z_, 
             double vx_, double vy_, double vz_,
             double q_, double m_, int species_) :
        x(x_), y(y_), z(z_), vx(vx_), vy(vy_), vz(vz_),
        q(q_), m(m_), species(species_) {}
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
        i = (i + nx) % nx;
        j = (j + ny) % ny;
        k = (k + nz) % nz;
        return data[i + nx * (j + ny * k)];
    }
    
    const double& operator()(int i, int j, int k) const {
        // Apply periodic boundary conditions
        i = (i + nx) % nx;
        j = (j + ny) % ny;
        k = (k + nz) % nz;
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
    
    // Electric field components (staggered grid)
    Field3D Ex, Ey, Ez;
    
    // Magnetic field components (staggered grid)
    Field3D Bx, By, Bz;
    
    // Current density components
    Field3D Jx, Jy, Jz;
    
    // Charge density
    Field3D rho;
    
    // Particles
    std::vector<Particle> particles;
    
    // Random number generator
    std::mt19937 rng;
    std::uniform_real_distribution<double> uniform_dist;
    std::normal_distribution<double> normal_dist;
    
public:
    PIC3D(const SimulationParams& p) : params(p), 
                                        rng(std::random_device{}()),
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
    }
    
    // Initialize particles with random positions and Maxwellian velocity distribution
    void initializeParticles() {
        particles.resize(params.num_particles);
        
        double Lx = params.nx * params.dx;
        double Ly = params.ny * params.dy;
        double Lz = params.nz * params.dz;
        
        // Thermal velocity (assuming kT/m ~ (0.1*c)^2 for demonstration)
        double vth_electron = 0.1 * C;
        double vth_ion = vth_electron * std::sqrt(ME / MP);
        
        for (int i = 0; i < params.num_particles; ++i) {
            Particle& p = particles[i];
            
            // Alternate between electrons and ions for charge neutrality
            if (i % 2 == 0) {
                // Electron
                p.species = 0;
                p.q = -QE;
                p.m = ME;
                
                // Maxwellian velocity distribution
                p.vx = vth_electron * normal_dist(rng);
                p.vy = vth_electron * normal_dist(rng);
                p.vz = vth_electron * normal_dist(rng);
            } else {
                // Ion (proton)
                p.species = 1;
                p.q = QE;
                p.m = MP;
                
                // Maxwellian velocity distribution
                p.vx = vth_ion * normal_dist(rng);
                p.vy = vth_ion * normal_dist(rng);
                p.vz = vth_ion * normal_dist(rng);
            }
            
            // Random position in the box
            p.x = uniform_dist(rng) * Lx;
            p.y = uniform_dist(rng) * Ly;
            p.z = uniform_dist(rng) * Lz;
        }
    }
    
    // Deposit charge to grid using Cloud-In-Cell (CIC) method
    void depositCharge() {
        rho.clear();
        
        for (const auto& p : particles) {
            // Get grid indices
            double x_norm = p.x / params.dx;
            double y_norm = p.y / params.dy;
            double z_norm = p.z / params.dz;
            
            int i = static_cast<int>(x_norm);
            int j = static_cast<int>(y_norm);
            int k = static_cast<int>(z_norm);
            
            // Get fractional positions
            double fx = x_norm - i;
            double fy = y_norm - j;
            double fz = z_norm - k;
            
            // Deposit charge to 8 surrounding grid points (CIC)
            double q_cell = p.q / (params.dx * params.dy * params.dz);
            
            rho(i,   j,   k  ) += q_cell * (1-fx) * (1-fy) * (1-fz);
            rho(i+1, j,   k  ) += q_cell * fx     * (1-fy) * (1-fz);
            rho(i,   j+1, k  ) += q_cell * (1-fx) * fy     * (1-fz);
            rho(i+1, j+1, k  ) += q_cell * fx     * fy     * (1-fz);
            rho(i,   j,   k+1) += q_cell * (1-fx) * (1-fy) * fz;
            rho(i+1, j,   k+1) += q_cell * fx     * (1-fy) * fz;
            rho(i,   j+1, k+1) += q_cell * (1-fx) * fy     * fz;
            rho(i+1, j+1, k+1) += q_cell * fx     * fy     * fz;
        }
    }
    
    // Deposit current to grid using Esirkepov's charge-conserving method (simplified)
    void depositCurrent() {
        Jx.clear();
        Jy.clear();
        Jz.clear();
        
        // Simplified current deposition - in practice, use Esirkepov's method
        for (const auto& p : particles) {
            // Get grid indices
            double x_norm = p.x / params.dx;
            double y_norm = p.y / params.dy;
            double z_norm = p.z / params.dz;
            
            int i = static_cast<int>(x_norm);
            int j = static_cast<int>(y_norm);
            int k = static_cast<int>(z_norm);
            
            // Get fractional positions
            double fx = x_norm - i;
            double fy = y_norm - j;
            double fz = z_norm - k;
            
            // Current density
            double jx = p.q * p.vx / (params.dx * params.dy * params.dz);
            double jy = p.q * p.vy / (params.dx * params.dy * params.dz);
            double jz = p.q * p.vz / (params.dx * params.dy * params.dz);
            
            // Deposit to staggered grid (simplified)
            // Jx on (i+1/2, j, k) grid
            Jx(i,   j,   k  ) += jx * (1-fy) * (1-fz) * 0.5;
            Jx(i+1, j,   k  ) += jx * (1-fy) * (1-fz) * 0.5;
            Jx(i,   j+1, k  ) += jx * fy     * (1-fz) * 0.5;
            Jx(i+1, j+1, k  ) += jx * fy     * (1-fz) * 0.5;
            Jx(i,   j,   k+1) += jx * (1-fy) * fz     * 0.5;
            Jx(i+1, j,   k+1) += jx * (1-fy) * fz     * 0.5;
            Jx(i,   j+1, k+1) += jx * fy     * fz     * 0.5;
            Jx(i+1, j+1, k+1) += jx * fy     * fz     * 0.5;
            
            // Similar for Jy and Jz
            Jy(i,   j,   k  ) += jy * (1-fx) * (1-fz) * 0.5;
            Jy(i+1, j,   k  ) += jy * fx     * (1-fz) * 0.5;
            Jy(i,   j+1, k  ) += jy * (1-fx) * (1-fz) * 0.5;
            Jy(i+1, j+1, k  ) += jy * fx     * (1-fz) * 0.5;
            Jy(i,   j,   k+1) += jy * (1-fx) * fz     * 0.5;
            Jy(i+1, j,   k+1) += jy * fx     * fz     * 0.5;
            Jy(i,   j+1, k+1) += jy * (1-fx) * fz     * 0.5;
            Jy(i+1, j+1, k+1) += jy * fx     * fz     * 0.5;
            
            Jz(i,   j,   k  ) += jz * (1-fx) * (1-fy) * 0.5;
            Jz(i+1, j,   k  ) += jz * fx     * (1-fy) * 0.5;
            Jz(i,   j+1, k  ) += jz * (1-fx) * fy     * 0.5;
            Jz(i+1, j+1, k  ) += jz * fx     * fy     * 0.5;
            Jz(i,   j,   k+1) += jz * (1-fx) * (1-fy) * 0.5;
            Jz(i+1, j,   k+1) += jz * fx     * (1-fy) * 0.5;
            Jz(i,   j+1, k+1) += jz * (1-fx) * fy     * 0.5;
            Jz(i+1, j+1, k+1) += jz * fx     * fy     * 0.5;
        }
    }
    
    // Update E field using Ampere's law (Yee's method)
    void updateEField() {
        double c2dt = C * C * params.dt;
        
        // Update Ex
        for (int i = 0; i <= params.nx; ++i) {
            for (int j = 0; j < params.ny; ++j) {
                for (int k = 0; k < params.nz; ++k) {
                    Ex(i,j,k) += c2dt * (
                        (Bz(i,j+1,k) - Bz(i,j,k)) / params.dy -
                        (By(i,j,k+1) - By(i,j,k)) / params.dz -
                        MU0 * Jx(i,j,k)
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
                        MU0 * Jy(i,j,k)
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
                        MU0 * Jz(i,j,k)
                    );
                }
            }
        }
    }
    
    // Update B field using Faraday's law (Yee's method)
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
    
    // Interpolate fields to particle position
    void interpolateFields(const Particle& p, double& Ex_p, double& Ey_p, double& Ez_p,
                          double& Bx_p, double& By_p, double& Bz_p) {
        // Get normalized positions
        double x_norm = p.x / params.dx;
        double y_norm = p.y / params.dy;
        double z_norm = p.z / params.dz;
        
        // Electric field interpolation (staggered grid)
        // Ex is on (i+1/2, j, k) grid
        int ix = static_cast<int>(x_norm - 0.5);
        int jy = static_cast<int>(y_norm);
        int kz = static_cast<int>(z_norm);
        double fx = x_norm - 0.5 - ix;
        double fy = y_norm - jy;
        double fz = z_norm - kz;
        
        Ex_p = Ex(ix,  jy,  kz  ) * (1-fx) * (1-fy) * (1-fz) +
               Ex(ix+1,jy,  kz  ) * fx     * (1-fy) * (1-fz) +
               Ex(ix,  jy+1,kz  ) * (1-fx) * fy     * (1-fz) +
               Ex(ix+1,jy+1,kz  ) * fx     * fy     * (1-fz) +
               Ex(ix,  jy,  kz+1) * (1-fx) * (1-fy) * fz     +
               Ex(ix+1,jy,  kz+1) * fx     * (1-fy) * fz     +
               Ex(ix,  jy+1,kz+1) * (1-fx) * fy     * fz     +
               Ex(ix+1,jy+1,kz+1) * fx     * fy     * fz;
        
        // Ey is on (i, j+1/2, k) grid
        ix = static_cast<int>(x_norm);
        jy = static_cast<int>(y_norm - 0.5);
        kz = static_cast<int>(z_norm);
        fx = x_norm - ix;
        fy = y_norm - 0.5 - jy;
        fz = z_norm - kz;
        
        Ey_p = Ey(ix,  jy,  kz  ) * (1-fx) * (1-fy) * (1-fz) +
               Ey(ix+1,jy,  kz  ) * fx     * (1-fy) * (1-fz) +
               Ey(ix,  jy+1,kz  ) * (1-fx) * fy     * (1-fz) +
               Ey(ix+1,jy+1,kz  ) * fx     * fy     * (1-fz) +
               Ey(ix,  jy,  kz+1) * (1-fx) * (1-fy) * fz     +
               Ey(ix+1,jy,  kz+1) * fx     * (1-fy) * fz     +
               Ey(ix,  jy+1,kz+1) * (1-fx) * fy     * fz     +
               Ey(ix+1,jy+1,kz+1) * fx     * fy     * fz;
        
        // Ez is on (i, j, k+1/2) grid
        ix = static_cast<int>(x_norm);
        jy = static_cast<int>(y_norm);
        kz = static_cast<int>(z_norm - 0.5);
        fx = x_norm - ix;
        fy = y_norm - jy;
        fz = z_norm - 0.5 - kz;
        
        Ez_p = Ez(ix,  jy,  kz  ) * (1-fx) * (1-fy) * (1-fz) +
               Ez(ix+1,jy,  kz  ) * fx     * (1-fy) * (1-fz) +
               Ez(ix,  jy+1,kz  ) * (1-fx) * fy     * (1-fz) +
               Ez(ix+1,jy+1,kz  ) * fx     * fy     * (1-fz) +
               Ez(ix,  jy,  kz+1) * (1-fx) * (1-fy) * fz     +
               Ez(ix+1,jy,  kz+1) * fx     * (1-fy) * fz     +
               Ez(ix,  jy+1,kz+1) * (1-fx) * fy     * fz     +
               Ez(ix+1,jy+1,kz+1) * fx     * fy     * fz;
        
        // Magnetic field interpolation (staggered grid)
        // Bx is on (i, j+1/2, k+1/2) grid
        ix = static_cast<int>(x_norm);
        jy = static_cast<int>(y_norm - 0.5);
        kz = static_cast<int>(z_norm - 0.5);
        fx = x_norm - ix;
        fy = y_norm - 0.5 - jy;
        fz = z_norm - 0.5 - kz;
        
        Bx_p = Bx(ix,  jy,  kz  ) * (1-fx) * (1-fy) * (1-fz) +
               Bx(ix+1,jy,  kz  ) * fx     * (1-fy) * (1-fz) +
               Bx(ix,  jy+1,kz  ) * (1-fx) * fy     * (1-fz) +
               Bx(ix+1,jy+1,kz  ) * fx     * fy     * (1-fz) +
               Bx(ix,  jy,  kz+1) * (1-fx) * (1-fy) * fz     +
               Bx(ix+1,jy,  kz+1) * fx     * (1-fy) * fz     +
               Bx(ix,  jy+1,kz+1) * (1-fx) * fy     * fz     +
               Bx(ix+1,jy+1,kz+1) * fx     * fy     * fz;
        
        // By is on (i+1/2, j, k+1/2) grid
        ix = static_cast<int>(x_norm - 0.5);
        jy = static_cast<int>(y_norm);
        kz = static_cast<int>(z_norm - 0.5);
        fx = x_norm - 0.5 - ix;
        fy = y_norm - jy;
        fz = z_norm - 0.5 - kz;
        
        By_p = By(ix,  jy,  kz  ) * (1-fx) * (1-fy) * (1-fz) +
               By(ix+1,jy,  kz  ) * fx     * (1-fy) * (1-fz) +
               By(ix,  jy+1,kz  ) * (1-fx) * fy     * (1-fz) +
               By(ix+1,jy+1,kz  ) * fx     * fy     * (1-fz) +
               By(ix,  jy,  kz+1) * (1-fx) * (1-fy) * fz     +
               By(ix+1,jy,  kz+1) * fx     * (1-fy) * fz     +
               By(ix,  jy+1,kz+1) * (1-fx) * fy     * fz     +
               By(ix+1,jy+1,kz+1) * fx     * fy     * fz;
        
        // Bz is on (i+1/2, j+1/2, k) grid
        ix = static_cast<int>(x_norm - 0.5);
        jy = static_cast<int>(y_norm - 0.5);
        kz = static_cast<int>(z_norm);
        fx = x_norm - 0.5 - ix;
        fy = y_norm - 0.5 - jy;
        fz = z_norm - kz;
        
        Bz_p = Bz(ix,  jy,  kz  ) * (1-fx) * (1-fy) * (1-fz) +
               Bz(ix+1,jy,  kz  ) * fx     * (1-fy) * (1-fz) +
               Bz(ix,  jy+1,kz  ) * (1-fx) * fy     * (1-fz) +
               Bz(ix+1,jy+1,kz  ) * fx     * fy     * (1-fz) +
               Bz(ix,  jy,  kz+1) * (1-fx) * (1-fy) * fz     +
               Bz(ix+1,jy,  kz+1) * fx     * (1-fy) * fz     +
               Bz(ix,  jy+1,kz+1) * (1-fx) * fy     * fz     +
               Bz(ix+1,jy+1,kz+1) * fx     * fy     * fz;
    }
    
    // Boris particle pusher
    void pushParticles() {
        double Lx = params.nx * params.dx;
        double Ly = params.ny * params.dy;
        double Lz = params.nz * params.dz;
        
        for (auto& p : particles) {
            // Interpolate fields to particle position
            double Ex_p, Ey_p, Ez_p, Bx_p, By_p, Bz_p;
            interpolateFields(p, Ex_p, Ey_p, Ez_p, Bx_p, By_p, Bz_p);
            
            // Boris algorithm
            double qdt_2m = p.q * params.dt / (2.0 * p.m);
            
            // Half acceleration from E field
            double vx_minus = p.vx + qdt_2m * Ex_p;
            double vy_minus = p.vy + qdt_2m * Ey_p;
            double vz_minus = p.vz + qdt_2m * Ez_p;
            
            // Rotation from B field
            double tx = qdt_2m * Bx_p;
            double ty = qdt_2m * By_p;
            double tz = qdt_2m * Bz_p;
            
            double t_mag2 = tx*tx + ty*ty + tz*tz;
            double sx = 2.0 * tx / (1.0 + t_mag2);
            double sy = 2.0 * ty / (1.0 + t_mag2);
            double sz = 2.0 * tz / (1.0 + t_mag2);
            
            // v' = v- + v- x t
            double vx_prime = vx_minus + vy_minus * tz - vz_minus * ty;
            double vy_prime = vy_minus + vz_minus * tx - vx_minus * tz;
            double vz_prime = vz_minus + vx_minus * ty - vy_minus * tx;
            
            // v+ = v- + v' x s
            double vx_plus = vx_minus + vy_prime * sz - vz_prime * sy;
            double vy_plus = vy_minus + vz_prime * sx - vx_prime * sz;
            double vz_plus = vz_minus + vx_prime * sy - vy_prime * sx;
            
            // Half acceleration from E field
            p.vx = vx_plus + qdt_2m * Ex_p;
            p.vy = vy_plus + qdt_2m * Ey_p;
            p.vz = vz_plus + qdt_2m * Ez_p;
            
            // Update position
            p.x += p.vx * params.dt;
            p.y += p.vy * params.dt;
            p.z += p.vz * params.dt;
            
            // Apply periodic boundary conditions
            while (p.x < 0) p.x += Lx;
            while (p.x >= Lx) p.x -= Lx;
            while (p.y < 0) p.y += Ly;
            while (p.y >= Ly) p.y -= Ly;
            while (p.z < 0) p.z += Lz;
            while (p.z >= Lz) p.z -= Lz;
        }
    }
    
    // Calculate field energy
    double calculateFieldEnergy() {
        double energy = 0.0;
        double dV = params.dx * params.dy * params.dz;
        
        // Electric field energy
        for (int i = 0; i <= params.nx; ++i) {
            for (int j = 0; j < params.ny; ++j) {
                for (int k = 0; k < params.nz; ++k) {
                    energy += 0.5 * EPS0 * Ex(i,j,k) * Ex(i,j,k) * dV;
                }
            }
        }
        
        for (int i = 0; i < params.nx; ++i) {
            for (int j = 0; j <= params.ny; ++j) {
                for (int k = 0; k < params.nz; ++k) {
                    energy += 0.5 * EPS0 * Ey(i,j,k) * Ey(i,j,k) * dV;
                }
            }
        }
        
        for (int i = 0; i < params.nx; ++i) {
            for (int j = 0; j < params.ny; ++j) {
                for (int k = 0; k <= params.nz; ++k) {
                    energy += 0.5 * EPS0 * Ez(i,j,k) * Ez(i,j,k) * dV;
                }
            }
        }
        
        // Magnetic field energy
        for (int i = 0; i < params.nx; ++i) {
            for (int j = 0; j <= params.ny; ++j) {
                for (int k = 0; k <= params.nz; ++k) {
                    energy += 0.5 * Bx(i,j,k) * Bx(i,j,k) / MU0 * dV;
                }
            }
        }
        
        for (int i = 0; i <= params.nx; ++i) {
            for (int j = 0; j < params.ny; ++j) {
                for (int k = 0; k <= params.nz; ++k) {
                    energy += 0.5 * By(i,j,k) * By(i,j,k) / MU0 * dV;
                }
            }
        }
        
        for (int i = 0; i <= params.nx; ++i) {
            for (int j = 0; j <= params.ny; ++j) {
                for (int k = 0; k < params.nz; ++k) {
                    energy += 0.5 * Bz(i,j,k) * Bz(i,j,k) / MU0 * dV;
                }
            }
        }
        
        return energy;
    }
    
    // Calculate particle kinetic energy
    double calculateParticleEnergy() {
        double energy = 0.0;
        
        for (const auto& p : particles) {
            double v2 = p.vx*p.vx + p.vy*p.vy + p.vz*p.vz;
            energy += 0.5 * p.m * v2;
        }
        
        return energy;
    }
    
    // Main simulation loop
    void run() {
        std::cout << "Starting 3D PIC simulation\n";
        std::cout << "Grid: " << params.nx << "x" << params.ny << "x" << params.nz << "\n";
        std::cout << "Particles: " << params.num_particles << "\n";
        std::cout << "Time steps: " << params.num_timesteps << "\n";
        std::cout << "dt = " << params.dt << " s\n\n";
        
        // Output file for diagnostics
        std::ofstream outfile("pic3d_diagnostics.txt");
        outfile << "# Time Step, Time (s), Field Energy (J), Particle Energy (J), Total Energy (J)\n";
        
        for (int step = 0; step < params.num_timesteps; ++step) {
            // Deposit current
            depositCurrent();
            
            // Update fields (Yee's method)
            updateBField();    // B^n -> B^(n+1/2)
            updateEField();    // E^n -> E^(n+1)
            updateBField();    // B^(n+1/2) -> B^(n+1)
            
            // Push particles (Boris method)
            pushParticles();
            
            // Diagnostics
            if (step % 10 == 0) {
                double field_energy = calculateFieldEnergy();
                double particle_energy = calculateParticleEnergy();
                double total_energy = field_energy + particle_energy;
                
                std::cout << "Step " << step << "/" << params.num_timesteps 
                         << " | Field Energy: " << std::scientific << field_energy 
                         << " J | Particle Energy: " << particle_energy 
                         << " J | Total: " << total_energy << " J\n";
                
                outfile << step << " " 
                       << step * params.dt << " "
                       << field_energy << " "
                       << particle_energy << " "
                       << total_energy << "\n";
            }
        }
        
        outfile.close();
        std::cout << "\nSimulation complete. Diagnostics saved to pic3d_diagnostics.txt\n";
    }
    
    // Save particle positions for visualization
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

// Main function
int main() {
    // Set up simulation parameters
    SimulationParams params;
    
    // Optional: Customize parameters
    params.nx = 32;
    params.ny = 32;
    params.nz = 32;
    params.num_particles = 10000;
    params.num_timesteps = 100;
    
    // Create and run simulation
    PIC3D simulation(params);
    
    // Save initial particle positions
    simulation.saveParticlePositions("particles_initial.txt");
    
    // Run simulation
    simulation.run();
    
    // Save final particle positions
    simulation.saveParticlePositions("particles_final.txt");
    
    return 0;
}