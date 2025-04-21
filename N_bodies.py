import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle
from scipy.fft import fftn, ifftn  # Using scipy's FFTW interface for faster FFTs
import time  # For performance monitoring

# Constants
G = 6.67430e-11  # Gravitational constant (m^3 kg^-1 s^-2)
# Scale G for simulation
G_SCALED = 1.0

class NBodySimulation:
    def __init__(self, n_bodies=9, box_size=10.0, dt=0.01, random_seed=None, 
                 edge_distance=0.2, initial_velocity=0, layout="grid",
                 mass_distribution='uniform', mass_range=(0.5, 3.0),
                 mesh_size=64, pm_threshold=200, performance_mode='auto'):
        """Initialize the n-body simulation."""
        if random_seed is not None:
            np.random.seed(random_seed)
            
        self.n_bodies = n_bodies
        self.box_size = box_size
        self.base_dt = dt  # Store the base time step
        self.dt = dt
        self.edge_distance = edge_distance
        self.layout = layout
        self.performance_mode = performance_mode
        
        # Performance settings
        self.use_adaptive_timestep = (performance_mode == 'adaptive' or 
                                     (performance_mode == 'auto' and n_bodies > 100))
        self.use_vectorized = (performance_mode != 'direct')
        
        # Symmetrization for direct calculations (compute i-j force once, apply to both)
        self.use_symmetrization = True
        
        # Counters for performance monitoring
        self.update_count = 0
        self.calculation_time = 0
        
        # Particle-Mesh method parameters
        self.mesh_size = mesh_size  # Size of mesh grid (mesh_size × mesh_size)
        self.pm_threshold = pm_threshold  # Switch to PM method when n_bodies > threshold
        # Prepare mesh grid for PM method
        self.mesh_cell_size = box_size / mesh_size
        self.use_pm_method = n_bodies > pm_threshold
        
        if self.use_pm_method:
            print(f"Using Particle-Mesh method for {n_bodies} bodies.")
            # Create the mesh grid for density and potential
            self.grid = np.zeros((mesh_size, mesh_size))
            # Calculate Green's function for the Poisson equation in Fourier space
            self.green_function = self._initialize_green_function()
        
        # For solar system, we need to handle masses differently
        if layout == "solar":
            # Generate masses for planets
            planet_masses = self.generate_masses(mass_distribution, mass_range, n_bodies - 1)
            # Create sun with mass 10x the maximum planet mass
            sun_mass = 10.0 * np.max(planet_masses)
            # Combine sun and planet masses
            self.masses = np.concatenate(([sun_mass], planet_masses))
        else:
            # Generate masses based on selected distribution for all bodies
            self.masses = self.generate_masses(mass_distribution, mass_range, n_bodies)
        
        # Set initial positions
        self.positions = np.zeros((n_bodies, 2))
        
        if layout == "grid":
            # Determine grid dimensions - try to make it as square as possible
            grid_size = int(np.ceil(np.sqrt(n_bodies)))
            
            # Calculate spacing for grid layout
            usable_width = (1 - 2 * edge_distance) * box_size
            spacing = usable_width / max(grid_size - 1, 1)
            
            # Calculate the offset from origin to place grid centered
            offset = -usable_width / 2
            
            # Place bodies in a grid pattern
            index = 0
            for i in range(grid_size):
                for j in range(grid_size):
                    if index < n_bodies:  # Only place up to n_bodies
                        self.positions[index] = [
                            offset + j * spacing,
                            offset + i * spacing
                        ]
                        index += 1
        
        elif layout == "circular":
            # Original circular pattern
            radius = (1 - edge_distance) * (box_size / 2)
            angles = np.linspace(0, 2 * np.pi, n_bodies, endpoint=False)
            
            for i in range(n_bodies):
                self.positions[i] = [radius * np.cos(angles[i]), radius * np.sin(angles[i])]
        
        elif layout == "solar":
            # Place sun at center
            self.positions[0] = [0, 0]
            
            # Place planets in orbits around the sun
            # Calculate minimum and maximum orbit radius
            min_radius = 0.1 * box_size
            max_radius = (1 - edge_distance) * (box_size / 2)
            
            # Create orbits with logarithmic spacing (more inner planets than outer)
            radii = np.logspace(np.log10(min_radius), np.log10(max_radius), n_bodies)
            
            # Distribute planets evenly around orbits
            angles = np.linspace(0, 2 * np.pi, n_bodies, endpoint=False)
            # Randomize angles slightly for more natural appearance
            angles += np.random.uniform(-0.1, 0.1, n_bodies)
            
            # Place planets
            for i in range(1, n_bodies):
                self.positions[i] = [
                    radii[i] * np.cos(angles[i]),
                    radii[i] * np.sin(angles[i])
                ]
                
        elif layout == "random":
            # Calculate the usable area accounting for edge distance
            usable_half_width = (1 - edge_distance) * (box_size / 2)
            
            # Generate random positions within the usable area
            for i in range(n_bodies):
                self.positions[i] = np.random.uniform(
                    -usable_half_width, 
                    usable_half_width, 
                    2  # 2 dimensions (x, y)
                )
        
        # Set initial velocities - scale with box size
        self.velocities = np.zeros((n_bodies, 2))
        
        if layout == "solar":
            # For solar system, set orbital velocities for planets
            sun_mass = self.masses[0]
            
            # Sun has no initial velocity
            self.velocities[0] = [0, 0]
            
            # Calculate orbital velocities for planets
            for i in range(1, n_bodies):
                # Vector from sun to planet
                r_vector = self.positions[i] - self.positions[0]
                r_norm = np.linalg.norm(r_vector)
                
                # Orbital velocity magnitude (v = sqrt(G * M_sun / r))
                orbital_velocity = np.sqrt(G_SCALED * sun_mass / r_norm)
                
                # Perpendicular direction for orbital motion
                perp_vector = np.array([-r_vector[1], r_vector[0]]) / r_norm
                
                # Set velocity
                self.velocities[i] = perp_vector * orbital_velocity
            
            # Add any additional initial velocity if specified
            if isinstance(initial_velocity, (int, float)) and initial_velocity != 0:
                scaled_velocity = initial_velocity * (box_size / 5.0)
                
                # Add velocity to planets (not sun)
                for i in range(1, n_bodies):
                    r_vector = self.positions[i] - self.positions[0]
                    r_norm = np.linalg.norm(r_vector)
                    perp_vector = np.array([-r_vector[1], r_vector[0]]) / r_norm
                    self.velocities[i] += perp_vector * scaled_velocity
                    
            elif initial_velocity == 'random':
                # Add random component to orbital velocities (for planets only)
                max_random_vel = box_size / 20.0  # Less randomness than default
                random_vels = np.random.uniform(-max_random_vel, max_random_vel, (n_bodies-1, 2))
                self.velocities[1:] += random_vels
                
        else:
            # For non-solar layouts, use the original velocity initialization
            if isinstance(initial_velocity, (int, float)):
                # Convert 0-1 scale to actual velocity scaled with box size
                scaled_velocity = initial_velocity * (box_size / 5.0)
                
                if scaled_velocity != 0:
                    for i in range(n_bodies):
                        # Direction vector from center to body
                        r_norm = np.linalg.norm(self.positions[i])
                        if r_norm > 1e-10:  # Avoid division by zero
                            r_vector = self.positions[i] / r_norm
                            # Perpendicular vector (rotate 90 degrees)
                            perp_vector = np.array([-r_vector[1], r_vector[0]])
                            # Set velocity magnitude and direction
                            self.velocities[i] = perp_vector * scaled_velocity
            elif initial_velocity == 'random':
                # Random velocities scaled with box size
                max_vel = box_size / 10.0
                self.velocities = np.random.uniform(-max_vel, max_vel, (n_bodies, 2))
        
        # For tracking trajectories
        self.trajectories = [[] for _ in range(n_bodies)]
        
        # Store the last acceleration for use in adaptive time steps
        self.last_accelerations = np.zeros((n_bodies, 2))
        
        # Colors for bodies
        if layout == "solar":
            # Special coloring for solar system: yellow sun, different colors for planets
            self.colors = np.zeros((n_bodies, 4))
            # Yellow sun
            self.colors[0] = [1, 0.9, 0, 1]  # RGBA: Yellow
            # Use a color map for planets
            planet_colors = plt.cm.jet(np.linspace(0, 1, n_bodies-1))
            self.colors[1:] = planet_colors
        else:
            # Standard coloring for other layouts
            self.colors = plt.cm.jet(np.linspace(0, 1, n_bodies))
            
        # Print performance mode information
        if self.use_pm_method:
            perf_method = "Particle-Mesh"
        elif self.use_vectorized:
            perf_method = "Vectorized"
        else:
            perf_method = "Direct"
            
        timestep_method = "Adaptive" if self.use_adaptive_timestep else "Fixed"
        
        print(f"Performance settings: {perf_method} calculation with {timestep_method} time step")
    
    def _initialize_green_function(self):
        """Initialize Green's function for the Poisson equation in Fourier space."""
        # Create meshgrid for wave numbers
        kx = 2 * np.pi * np.fft.fftfreq(self.mesh_size)
        ky = 2 * np.pi * np.fft.fftfreq(self.mesh_size)
        kx_grid, ky_grid = np.meshgrid(kx, ky)
        
        # Calculate squared wave number (k^2)
        k_squared = kx_grid**2 + ky_grid**2
        
        # The Green's function in Fourier space is -1/k^2
        # Handle k=0 case (DC component)
        k_squared[0, 0] = 1.0  # Avoid division by zero
        green_function = -1.0 / k_squared
        green_function[0, 0] = 0.0  # Set DC component to zero
        
        return green_function
    
    def generate_masses(self, distribution, mass_range, count):
        """Generate masses according to the specified distribution."""
        min_mass, max_mass = mass_range
        
        if distribution == 'equal':
            # All masses equal to the average of min and max
            return np.ones(count) * np.mean(mass_range)
        
        elif distribution == 'uniform':
            # Uniform distribution between min and max
            return np.random.uniform(min_mass, max_mass, count)
        
        elif distribution == 'gaussian':
            # Normal distribution with mean at center of range, std of 1/4 the range
            mean = np.mean(mass_range)
            std = (max_mass - min_mass) / 4
            masses = np.random.normal(mean, std, count)
            # Clip to ensure masses are within range
            return np.clip(masses, min_mass, max_mass)
        
        elif distribution == 'long-tail':
            # Power law distribution (more small masses, few large ones)
            # Using Pareto distribution shifted to start at min_mass
            alpha = 1.5  # Shape parameter for Pareto distribution
            # Generate raw values from Pareto
            raw_values = np.random.pareto(alpha, count)
            # Scale to fit within desired range
            scale_factor = (max_mass - min_mass) / max(np.max(raw_values), 1e-10)
            masses = min_mass + raw_values * scale_factor
            # Clip to ensure no values exceed max_mass
            return np.clip(masses, min_mass, max_mass)
        
        else:
            # Default to uniform if unknown distribution specified
            print(f"Unknown mass distribution '{distribution}', using uniform")
            return np.random.uniform(min_mass, max_mass, count)
    
    def calculate_acceleration_direct(self, positions, masses):
        """Calculate acceleration using direct N-body method (O(N²))."""
        n = len(positions)
        acceleration = np.zeros_like(positions)
        
        # Use symmetrization to reduce computations
        if self.use_symmetrization:
            for i in range(n):
                for j in range(i+1, n):  # Only calculate upper triangle
                    # Vector from body i to body j
                    r_ij = positions[j] - positions[i]
                    # Distance between bodies
                    r = np.linalg.norm(r_ij)
                    # Avoid division by zero or very small distances
                    min_distance = 0.01 * self.box_size / 10
                    if r < min_distance:
                        r = min_distance
                    # Gravitational force direction
                    r_hat = r_ij / r
                    # Acceleration magnitudes (F = G*m1*m2/r^2)
                    a_i_mag = G_SCALED * masses[j] / (r**2)
                    a_j_mag = G_SCALED * masses[i] / (r**2)
                    # Acceleration vectors (opposite directions)
                    acceleration[i] += a_i_mag * r_hat
                    acceleration[j] -= a_j_mag * r_hat  # Note the minus sign
        else:
            # Original direct calculation
            for i in range(n):
                for j in range(n):
                    if i != j:
                        # Vector from body i to body j
                        r_ij = positions[j] - positions[i]
                        # Distance between bodies
                        r = np.linalg.norm(r_ij)
                        # Avoid division by zero or very small distances
                        min_distance = 0.01 * self.box_size / 10
                        if r < min_distance:
                            r = min_distance
                        # Gravitational force direction
                        r_hat = r_ij / r
                        # Acceleration magnitude
                        a_mag = G_SCALED * masses[j] / (r**2)
                        # Acceleration vector
                        acceleration[i] += a_mag * r_hat
        
        return acceleration
    
    def calculate_acceleration_vectorized(self, positions, masses):
        """Calculate acceleration using vectorized operations for better performance."""
        n = len(positions)
        acceleration = np.zeros_like(positions)
        
        # Use numpy broadcasting to calculate all pairwise distances at once
        for i in range(n):
            # Compute vectors from body i to all other bodies (including itself)
            r_vectors = positions - positions[i]
            
            # Compute distances
            distances = np.sqrt(np.sum(r_vectors**2, axis=1))
            
            # Avoid self-interaction and small distances
            min_distance = 0.01 * self.box_size / 10
            mask = (distances > min_distance)
            mask[i] = False  # Exclude self-interaction
            
            # Compute accelerations only for valid interactions
            if np.any(mask):
                # Unit vectors
                r_hat = r_vectors[mask] / distances[mask][:, np.newaxis]
                
                # Acceleration magnitudes
                a_mags = G_SCALED * masses[mask] / (distances[mask]**2)
                
                # Sum all accelerations
                acceleration[i] = np.sum(r_hat * a_mags[:, np.newaxis], axis=0)
        
        return acceleration
    
    def calculate_acceleration_pm(self, positions, masses):
        """Calculate acceleration using Particle-Mesh method (O(N + M log M))."""
        # Clear density grid
        self.grid.fill(0.0)
        
        # Assign masses to grid points using Cloud-in-Cell (CIC) interpolation
        for i in range(self.n_bodies):
            # Convert position to grid coordinates
            x_pos = (positions[i, 0] + self.box_size/2) / self.mesh_cell_size
            y_pos = (positions[i, 1] + self.box_size/2) / self.mesh_cell_size
            
            # Get the grid cell indices for the four nearest cells
            x0 = int(np.floor(x_pos))
            y0 = int(np.floor(y_pos))
            
            # Calculate weights for CIC interpolation
            dx = x_pos - x0
            dy = y_pos - y0
            
            # Ensure indices are within bounds using periodic boundary conditions
            x0 = x0 % self.mesh_size
            y0 = y0 % self.mesh_size
            x1 = (x0 + 1) % self.mesh_size
            y1 = (y0 + 1) % self.mesh_size
            
            # Distribute mass to the four nearest grid points
            self.grid[y0, x0] += masses[i] * (1 - dx) * (1 - dy)
            self.grid[y0, x1] += masses[i] * dx * (1 - dy)
            self.grid[y1, x0] += masses[i] * (1 - dx) * dy
            self.grid[y1, x1] += masses[i] * dx * dy
        
        # Normalize density by cell area
        self.grid /= self.mesh_cell_size**2
        
        # Solve Poisson equation using FFT
        # This gives us the gravitational potential from the density distribution
        density_fft = fftn(self.grid)
        potential_fft = density_fft * self.green_function
        potential = np.real(ifftn(potential_fft))
        
        # Calculate the gradient of the potential to get the gravitational field
        # Using central differences
        field_x = np.zeros_like(potential)
        field_y = np.zeros_like(potential)
        
        # Calculate x-component of the field
        field_x[:, 1:-1] = (potential[:, 2:] - potential[:, :-2]) / (2 * self.mesh_cell_size)
        field_x[:, 0] = (potential[:, 1] - potential[:, -1]) / (2 * self.mesh_cell_size)
        field_x[:, -1] = (potential[:, 0] - potential[:, -2]) / (2 * self.mesh_cell_size)
        
        # Calculate y-component of the field
        field_y[1:-1, :] = (potential[2:, :] - potential[:-2, :]) / (2 * self.mesh_cell_size)
        field_y[0, :] = (potential[1, :] - potential[-1, :]) / (2 * self.mesh_cell_size)
        field_y[-1, :] = (potential[0, :] - potential[-2, :]) / (2 * self.mesh_cell_size)
        
        # Interpolate the gravitational field back to particle positions
        accelerations = np.zeros_like(positions)
        
        for i in range(self.n_bodies):
            # Convert position to grid coordinates
            x_pos = (positions[i, 0] + self.box_size/2) / self.mesh_cell_size
            y_pos = (positions[i, 1] + self.box_size/2) / self.mesh_cell_size
            
            # Get the grid cell indices for the four nearest cells
            x0 = int(np.floor(x_pos))
            y0 = int(np.floor(y_pos))
            
            # Calculate weights for CIC interpolation
            dx = x_pos - x0
            dy = y_pos - y0
            
            # Ensure indices are within bounds using periodic boundary conditions
            x0 = x0 % self.mesh_size
            y0 = y0 % self.mesh_size
            x1 = (x0 + 1) % self.mesh_size
            y1 = (y0 + 1) % self.mesh_size
            
            # Interpolate field values (note: we flip sign for acceleration)
            accelerations[i, 0] = -(field_x[y0, x0] * (1 - dx) * (1 - dy) +
                                  field_x[y0, x1] * dx * (1 - dy) +
                                  field_x[y1, x0] * (1 - dx) * dy +
                                  field_x[y1, x1] * dx * dy)
            
            accelerations[i, 1] = -(field_y[y0, x0] * (1 - dx) * (1 - dy) +
                                  field_y[y0, x1] * dx * (1 - dy) +
                                  field_y[y1, x0] * (1 - dx) * dy +
                                  field_y[y1, x1] * dx * dy)
        
        # Scale by G (already incorporated in the Green's function)
        accelerations *= G_SCALED
        
        return accelerations
    
    def calculate_acceleration(self, positions, masses):
        """Choose appropriate method to calculate accelerations based on system size."""
        start_time = time.time()
        
        if self.use_pm_method:
            accelerations = self.calculate_acceleration_pm(positions, masses)
        elif self.use_vectorized:
            accelerations = self.calculate_acceleration_vectorized(positions, masses)
        else:
            accelerations = self.calculate_acceleration_direct(positions, masses)
        
        # Record calculation time
        self.calculation_time += time.time() - start_time
        self.update_count += 1
        
        # Return calculated accelerations
        return accelerations
    
    def calculate_adaptive_timestep(self, accelerations):
        """Calculate an appropriate time step based on current accelerations."""
        # Get maximum acceleration magnitude
        a_max = np.max(np.linalg.norm(accelerations, axis=1))
        
        # If there's no significant acceleration, use base time step
        if a_max < 1e-10:
            return self.base_dt
        
        # Calculate time step based on acceleration
        # The idea is to limit how much the velocity can change in one step
        # and how far particles can move relative to their acceleration
        
        # Characteristic velocity (approximated from acceleration * base_dt)
        v_char = a_max * self.base_dt
        
        if v_char < 1e-10:
            return self.base_dt
        
        # Base the time step on the characteristic distance particles would move
        # We want particles to move at most ~1% of the box size in one step
        max_distance = 0.01 * self.box_size
        
        # Time to move max_distance at current acceleration
        t_acc = np.sqrt(2 * max_distance / a_max)
        
        # Limit the timestep to stay between 0.1 and 10 times the base timestep
        dt = np.clip(t_acc, 0.1 * self.base_dt, 10.0 * self.base_dt)
        
        return dt
    
    def update(self):
        """Update positions and velocities using Velocity Verlet integration."""
        # Calculate acceleration at current positions
        accelerations = self.calculate_acceleration(self.positions, self.masses)
        
        # Store accelerations for next adaptive time step calculation
        self.last_accelerations = accelerations.copy()
        
        # Adaptive time stepping if enabled
        if self.use_adaptive_timestep:
            self.dt = self.calculate_adaptive_timestep(accelerations)
        
        # Update positions and half-step velocities
        self.positions += self.velocities * self.dt + 0.5 * accelerations * self.dt**2
        self.velocities += 0.5 * accelerations * self.dt
        
        # Calculate acceleration at new positions
        new_accelerations = self.calculate_acceleration(self.positions, self.masses)
        
        # Complete velocity update
        self.velocities += 0.5 * new_accelerations * self.dt
        
        # Update trajectories (save a position copy every few steps to save memory)
        # Save more frequently for small simulations, less frequently for large ones
        trajectory_interval = max(1, min(10, self.n_bodies // 50))
        if self.update_count % trajectory_interval == 0:
            for i in range(self.n_bodies):
                self.trajectories[i].append(self.positions[i].copy())
    
    def animate(self, num_steps=1000, skip_steps=1):
        """Create and display the animation of the n-body system."""
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xlim(-self.box_size/2, self.box_size/2)
        ax.set_ylim(-self.box_size/2, self.box_size/2)
        ax.set_aspect('equal')
        
        # Show method and layout in title
        method = "Particle-Mesh" if self.use_pm_method else "Direct N-Body"
        if self.use_adaptive_timestep:
            method += " (Adaptive)"
        ax.set_title(f'N-Body Simulation (N={self.n_bodies}, {self.layout}, {method})')
        
        # Add FPS counter and performance meter
        fps_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, 
                          verticalalignment='top', fontsize=8)
        
        # Initialize circles for bodies
        circles = []
        trajectory_lines = []
        
        # Scale body size with box_size, with special handling for solar system
        if self.layout == "solar":
            # Sun size proportional to mass but not too large
            sun_radius = 0.04 * self.box_size
            
            # Sun
            circle = Circle((0, 0), sun_radius, color=self.colors[0], alpha=0.9)
            circles.append(circle)
            ax.add_patch(circle)
            
            # Sun trajectory
            line, = ax.plot([], [], '-', lw=1, alpha=0.3, color=self.colors[0])
            trajectory_lines.append(line)
            
            # Planets with size scaled by mass
            base_radius = 0.015 * self.box_size
            max_planet_mass = np.max(self.masses[1:])
            
            for i in range(1, self.n_bodies):
                radius = base_radius * (self.masses[i] / max_planet_mass)**0.3
                circle = Circle((0, 0), radius, color=self.colors[i], alpha=0.7)
                circles.append(circle)
                ax.add_patch(circle)
                
                # Initialize trajectory line
                line, = ax.plot([], [], '-', lw=1, alpha=0.3, color=self.colors[i])
                trajectory_lines.append(line)
        else:
            # Regular sizing for grid and circular layouts
            base_radius = 0.02 * self.box_size
            
            for i in range(self.n_bodies):
                radius = base_radius * (self.masses[i] / np.max(self.masses))**0.5
                circle = Circle((0, 0), radius, color=self.colors[i], alpha=0.7)
                circles.append(circle)
                ax.add_patch(circle)
                
                # Initialize trajectory line
                line, = ax.plot([], [], '-', lw=1, alpha=0.3, color=self.colors[i])
                trajectory_lines.append(line)
        
        # In PM mode, optionally show a mesh reference grid
        if self.use_pm_method and self.mesh_size <= 64:  # Only show grid if not too dense
            # Draw mesh grid lines (faintly)
            grid_lines = []
            for i in range(self.mesh_size + 1):
                pos = -self.box_size/2 + i * self.mesh_cell_size
                # Horizontal line
                line, = ax.plot([-self.box_size/2, self.box_size/2], [pos, pos], 
                              'k-', alpha=0.1, lw=0.5)
                grid_lines.append(line)
                # Vertical line
                line, = ax.plot([pos, pos], [-self.box_size/2, self.box_size/2], 
                              'k-', alpha=0.1, lw=0.5)
                grid_lines.append(line)
        
        def init():
            """Initialize animation."""
            for i, circle in enumerate(circles):
                circle.center = self.positions[i]
            
            fps_text.set_text('')
            
            return circles + trajectory_lines + [fps_text]
        
        # For FPS calculation
        frame_times = []
        last_frame_time = time.time()
        
        def animate(frame):
            """Perform animation step."""
            nonlocal last_frame_time
            
            # Calculate FPS
            current_time = time.time()
            frame_times.append(current_time - last_frame_time)
            last_frame_time = current_time
            
            # Keep only the last 20 frames for FPS calculation
            if len(frame_times) > 20:
                frame_times.pop(0)
            
            # Average FPS and calculation time per step
            if len(frame_times) > 0:
                avg_fps = 1.0 / (sum(frame_times) / len(frame_times))
                avg_calc_time = self.calculation_time / max(1, self.update_count) * 1000  # ms
                if self.use_adaptive_timestep:
                    dt_text = f", dt={self.dt:.4f}"
                else:
                    dt_text = ""
                fps_text.set_text(f'FPS: {avg_fps:.1f}, Calc: {avg_calc_time:.1f} ms{dt_text}')
            
            # Run multiple physics steps per frame for smoother animation
            for _ in range(skip_steps):
                self.update()
            
            # Update circle positions
            for i, circle in enumerate(circles):
                circle.center = self.positions[i]
                
                # Adjust trail length based on number of bodies and layout
                if self.layout == "solar":
                    # Longer trails for sun and planets
                    if i == 0:  # Sun
                        trail_length = 10  # Short trail for sun
                    else:  # Planets
                        trail_length = int(100 * (self.box_size / 10))
                elif self.n_bodies <= 100:
                    trail_length = int(100 * (self.box_size / 10))
                else:
                    # Shorter trails for many bodies to maintain performance
                    trail_length = int(20 * (self.box_size / 10))
                
                # Update trajectory lines (limited to last trail_length points)
                if len(self.trajectories[i]) > trail_length:
                    traj = np.array(self.trajectories[i][-trail_length:])
                else:
                    traj = np.array(self.trajectories[i])
                
                if len(traj) > 1:
                    trajectory_lines[i].set_data(traj[:, 0], traj[:, 1])
            
            return circles + trajectory_lines + [fps_text]
        
        anim = FuncAnimation(fig, animate, frames=num_steps, 
                             init_func=init, blit=True, interval=20)
        
        plt.tight_layout()
        plt.show()
        
        # Print performance statistics
        if self.update_count > 0:
            avg_calc_time = self.calculation_time / self.update_count * 1000  # ms
            print(f"\nPerformance statistics:")
            print(f"- Average calculation time: {avg_calc_time:.2f} ms per step")
            print(f"- Total update steps: {self.update_count}")
        
        return anim

def get_parameter(prompt, default, convert_func=None):
    """Ask for parameter input with default value."""
    user_input = input(f"{prompt} [{default}]: ").strip()
    if not user_input:
        return default
    try:
        if convert_func:
            return convert_func(user_input)
        return user_input
    except ValueError:
        print(f"Invalid input, using default: {default}")
        return default

def get_yes_no(prompt, default=True):
    """Ask for yes/no input with default value."""
    default_str = "Y/n" if default else "y/N"
    user_input = input(f"{prompt} [{default_str}]: ").strip().lower()
    if not user_input:
        return default
    return user_input[0] == 'y'

def get_choice(prompt, options, default):
    """Ask user to choose from a list of options."""
    options_str = '/'.join(options)
    user_input = input(f"{prompt} [{options_str}, default={default}]: ").strip().lower()
    if not user_input:
        return default
    if user_input in options:
        return user_input
    print(f"Invalid choice, using default: {default}")
    return default

# Run the simulation
if __name__ == "__main__":
    print("N-Body Gravitational Simulation")
    print("--------------------------------")
    print("Enter parameters (or press Enter for defaults)")
    
    # First ask for number of bodies
    n_bodies = get_parameter("Number of bodies", 3, int)
    if n_bodies >= 1000:
        print('Number too high, scaled down to 1000')
        n_bodies = 1000
    
    # PM method settings
    pm_threshold = 200
    if n_bodies > pm_threshold:
        print(f"Using Particle-Mesh method for {n_bodies} bodies (faster for large systems)")
        # For larger simulations, let user customize mesh size
        default_mesh = 64 if n_bodies < 500 else 128
        mesh_size = get_parameter("Mesh grid size (power of 2 recommended)", default_mesh, int)
    else:
        print(f"Using direct N-body method for {n_bodies} bodies")
        mesh_size = 64  # Default, won't be used
    
    # Performance mode selection
    if n_bodies > 20:
        perf_mode = get_choice(
            "Performance mode", 
            ["auto", "direct", "vectorized", "adaptive"], 
            "auto"
        )
    else:
        perf_mode = "direct"  # For few bodies, direct is accurate and fast enough
    
    box_size = get_parameter("Box size", 20.0, float)
    
    # Ask for layout type first
    layout_choice = get_choice("Initial arrangement", ["grid", "circular", "solar", "random"], "random")
    
    # Only ask for edge_distance for certain layouts
    if layout_choice in ["circular", "solar", "random"]:
        edge_distance = get_parameter("Edge distance from box boundary (fraction, 0-1)", 0.2, float)
    else:
        edge_distance = 0.2  # Default value, will be used for grid spacing calculation
    
    # If solar layout is chosen, let user know about the special mass handling
    if layout_choice == "solar":
        print("\nSolar system configuration:")
        print("- A massive 'sun' will be placed at the center")
        print("- Planets will orbit with stable initial velocities")
        print("- The sun's mass will be 10x the maximum planet mass")
    
    # Mass distribution options
    mass_distribution = get_choice(
        "Mass distribution", 
        ["equal", "uniform", "gaussian", "long-tail"], 
        "uniform"
    )
    
    # Mass range (only if not using equal masses)
    if mass_distribution != "equal":
        min_mass = get_parameter("Minimum mass", 0.1, float)
        max_mass = get_parameter("Maximum mass", 1.0, float)
        if min_mass >= max_mass:
            print("Minimum mass must be less than maximum mass. Using defaults.")
            min_mass, max_mass = 1.0, 1.0
        mass_range = (min_mass, max_mass)
    else:
        mass_value = get_parameter("Mass value", 1.0, float)
        mass_range = (mass_value, mass_value)
    
    # Velocity input (now in range 0-1, will be scaled with box size)
    if layout_choice == "solar":
        vel_prompt = "Additional velocity on top of orbital velocity (0-1 or 'random')"
    else:
        vel_prompt = "Initial velocity (0-1 or 'random')"
        
    vel_input = input(f"{vel_prompt} [0]: ").strip().lower()
    if not vel_input:
        initial_velocity = 0
    elif vel_input == 'random':
        initial_velocity = 'random'
    else:
        try:
            initial_velocity = float(vel_input)
            # Ensure velocity is in 0-1 range
            if not 0 <= initial_velocity <= 1:
                print("Velocity must be between 0 and 1. Adjusting to valid range.")
                initial_velocity = max(0, min(1, initial_velocity))
        except ValueError:
            print("Invalid input for velocity, using default: 0")
            initial_velocity = 0
    
    # Create simulation with user parameters
    sim = NBodySimulation(
        n_bodies=n_bodies, 
        box_size=box_size, 
        edge_distance=edge_distance,
        initial_velocity=initial_velocity,
        layout=layout_choice,
        mass_distribution=mass_distribution,
        mass_range=mass_range,
        mesh_size=mesh_size,
        pm_threshold=pm_threshold,
        performance_mode=perf_mode
    )
    
    # Run animation
    print("\nStarting simulation...")
    anim = sim.animate()