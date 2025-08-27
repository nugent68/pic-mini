#!/usr/bin/env python3
"""
Visualization script for 3D PIC simulation results
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
import os

def load_particles(filename):
    """Load particle data from file"""
    if not os.path.exists(filename):
        print(f"Error: File {filename} not found")
        return None
    
    data = np.loadtxt(filename, comments='#')
    if len(data) == 0:
        print(f"Error: No data in {filename}")
        return None
    
    return {
        'x': data[:, 0],
        'y': data[:, 1],
        'z': data[:, 2],
        'vx': data[:, 3],
        'vy': data[:, 4],
        'vz': data[:, 5],
        'species': data[:, 6].astype(int)
    }

def load_diagnostics(filename):
    """Load diagnostic data from file"""
    if not os.path.exists(filename):
        print(f"Error: File {filename} not found")
        return None
    
    data = np.loadtxt(filename, comments='#')
    if len(data) == 0:
        print(f"Error: No data in {filename}")
        return None
    
    if data.shape[1] == 5:  # Full version
        return {
            'step': data[:, 0],
            'time': data[:, 1],
            'field_energy': data[:, 2],
            'particle_energy': data[:, 3],
            'total_energy': data[:, 4]
        }
    elif data.shape[1] == 3:  # Stable version
        return {
            'step': data[:, 0],
            'time': data[:, 1],
            'total_energy': data[:, 2]
        }
    else:
        print(f"Error: Unexpected data format in {filename}")
        return None

def plot_particles_3d(particles, title="Particle Distribution"):
    """Create 3D scatter plot of particles"""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Separate by species
    electrons = particles['species'] == 0
    ions = particles['species'] == 1
    
    # Plot electrons
    if np.any(electrons):
        ax.scatter(particles['x'][electrons], 
                  particles['y'][electrons], 
                  particles['z'][electrons],
                  c='blue', s=1, alpha=0.5, label='Electrons')
    
    # Plot ions
    if np.any(ions):
        ax.scatter(particles['x'][ions], 
                  particles['y'][ions], 
                  particles['z'][ions],
                  c='red', s=1, alpha=0.5, label='Ions')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    ax.legend()
    
    return fig

def plot_phase_space(particles, title="Phase Space"):
    """Create phase space plots"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Separate by species
    electrons = particles['species'] == 0
    ions = particles['species'] == 1
    
    # X-Vx phase space
    axes[0, 0].scatter(particles['x'][electrons], particles['vx'][electrons], 
                      c='blue', s=1, alpha=0.5, label='Electrons')
    axes[0, 0].scatter(particles['x'][ions], particles['vx'][ions], 
                      c='red', s=1, alpha=0.5, label='Ions')
    axes[0, 0].set_xlabel('X')
    axes[0, 0].set_ylabel('Vx')
    axes[0, 0].set_title('X-Vx Phase Space')
    axes[0, 0].legend()
    
    # Y-Vy phase space
    axes[0, 1].scatter(particles['y'][electrons], particles['vy'][electrons], 
                      c='blue', s=1, alpha=0.5)
    axes[0, 1].scatter(particles['y'][ions], particles['vy'][ions], 
                      c='red', s=1, alpha=0.5)
    axes[0, 1].set_xlabel('Y')
    axes[0, 1].set_ylabel('Vy')
    axes[0, 1].set_title('Y-Vy Phase Space')
    
    # Z-Vz phase space
    axes[0, 2].scatter(particles['z'][electrons], particles['vz'][electrons], 
                      c='blue', s=1, alpha=0.5)
    axes[0, 2].scatter(particles['z'][ions], particles['vz'][ions], 
                      c='red', s=1, alpha=0.5)
    axes[0, 2].set_xlabel('Z')
    axes[0, 2].set_ylabel('Vz')
    axes[0, 2].set_title('Z-Vz Phase Space')
    
    # Velocity distributions
    v_electrons = np.sqrt(particles['vx'][electrons]**2 + 
                         particles['vy'][electrons]**2 + 
                         particles['vz'][electrons]**2)
    v_ions = np.sqrt(particles['vx'][ions]**2 + 
                    particles['vy'][ions]**2 + 
                    particles['vz'][ions]**2)
    
    axes[1, 0].hist(particles['vx'][electrons], bins=50, alpha=0.5, 
                   color='blue', label='Electrons', density=True)
    axes[1, 0].hist(particles['vx'][ions], bins=50, alpha=0.5, 
                   color='red', label='Ions', density=True)
    axes[1, 0].set_xlabel('Vx')
    axes[1, 0].set_ylabel('Probability Density')
    axes[1, 0].set_title('Vx Distribution')
    axes[1, 0].legend()
    
    axes[1, 1].hist(particles['vy'][electrons], bins=50, alpha=0.5, 
                   color='blue', density=True)
    axes[1, 1].hist(particles['vy'][ions], bins=50, alpha=0.5, 
                   color='red', density=True)
    axes[1, 1].set_xlabel('Vy')
    axes[1, 1].set_ylabel('Probability Density')
    axes[1, 1].set_title('Vy Distribution')
    
    axes[1, 2].hist(v_electrons, bins=50, alpha=0.5, 
                   color='blue', label='Electrons', density=True)
    axes[1, 2].hist(v_ions, bins=50, alpha=0.5, 
                   color='red', label='Ions', density=True)
    axes[1, 2].set_xlabel('|V|')
    axes[1, 2].set_ylabel('Probability Density')
    axes[1, 2].set_title('Speed Distribution')
    axes[1, 2].legend()
    
    plt.suptitle(title)
    plt.tight_layout()
    
    return fig

def plot_energy_evolution(diagnostics, title="Energy Evolution"):
    """Plot energy evolution over time"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if 'field_energy' in diagnostics and 'particle_energy' in diagnostics:
        ax.plot(diagnostics['time'], diagnostics['field_energy'], 
               label='Field Energy', linewidth=2)
        ax.plot(diagnostics['time'], diagnostics['particle_energy'], 
               label='Particle Energy', linewidth=2)
        ax.plot(diagnostics['time'], diagnostics['total_energy'], 
               label='Total Energy', linewidth=2, linestyle='--')
    else:
        ax.plot(diagnostics['time'], diagnostics['total_energy'], 
               label='Total Energy', linewidth=2)
    
    ax.set_xlabel('Time')
    ax.set_ylabel('Energy')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add energy conservation check
    if len(diagnostics['total_energy']) > 1:
        initial_energy = diagnostics['total_energy'][0]
        final_energy = diagnostics['total_energy'][-1]
        error = abs(final_energy - initial_energy) / abs(initial_energy) * 100
        ax.text(0.02, 0.98, f'Energy Conservation Error: {error:.2e}%',
               transform=ax.transAxes, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    return fig

def main():
    """Main visualization function"""
    print("PIC3D Simulation Visualization")
    print("=" * 40)
    
    # Check for available data files
    files_to_check = [
        ('particles_initial.txt', 'Initial particle distribution'),
        ('particles_final.txt', 'Final particle distribution'),
        ('particles_stable_final.txt', 'Stable simulation final particles'),
        ('pic3d_diagnostics.txt', 'Energy diagnostics'),
        ('pic3d_stable_diagnostics.txt', 'Stable simulation diagnostics')
    ]
    
    available_files = []
    for filename, description in files_to_check:
        if os.path.exists(filename):
            available_files.append((filename, description))
            print(f"✓ Found: {filename} - {description}")
        else:
            print(f"✗ Missing: {filename}")
    
    if not available_files:
        print("\nNo data files found. Please run the simulation first.")
        return
    
    print("\n" + "=" * 40)
    
    # Plot available data
    figs = []
    
    # Plot particle distributions
    for filename, description in available_files:
        if 'particles' in filename:
            print(f"\nLoading {filename}...")
            particles = load_particles(filename)
            if particles is not None:
                # 3D distribution
                fig = plot_particles_3d(particles, f"3D {description}")
                figs.append(fig)
                
                # Phase space
                fig = plot_phase_space(particles, f"Phase Space - {description}")
                figs.append(fig)
                
                # Print statistics
                n_electrons = np.sum(particles['species'] == 0)
                n_ions = np.sum(particles['species'] == 1)
                print(f"  Electrons: {n_electrons}")
                print(f"  Ions: {n_ions}")
                
                if n_electrons > 0:
                    v_rms_e = np.sqrt(np.mean(particles['vx'][particles['species']==0]**2 + 
                                             particles['vy'][particles['species']==0]**2 + 
                                             particles['vz'][particles['species']==0]**2))
                    print(f"  Electron v_rms: {v_rms_e:.3e}")
                
                if n_ions > 0:
                    v_rms_i = np.sqrt(np.mean(particles['vx'][particles['species']==1]**2 + 
                                             particles['vy'][particles['species']==1]**2 + 
                                             particles['vz'][particles['species']==1]**2))
                    print(f"  Ion v_rms: {v_rms_i:.3e}")
    
    # Plot diagnostics
    for filename, description in available_files:
        if 'diagnostics' in filename:
            print(f"\nLoading {filename}...")
            diagnostics = load_diagnostics(filename)
            if diagnostics is not None:
                fig = plot_energy_evolution(diagnostics, f"Energy Evolution - {description}")
                figs.append(fig)
                
                # Print energy statistics
                print(f"  Initial total energy: {diagnostics['total_energy'][0]:.3e}")
                print(f"  Final total energy: {diagnostics['total_energy'][-1]:.3e}")
                error = abs(diagnostics['total_energy'][-1] - diagnostics['total_energy'][0]) / abs(diagnostics['total_energy'][0])
                print(f"  Energy conservation error: {error*100:.2e}%")
    
    # Show all plots
    if figs:
        print("\nDisplaying plots...")
        plt.show()
    else:
        print("\nNo valid data to plot.")

if __name__ == "__main__":
    main()