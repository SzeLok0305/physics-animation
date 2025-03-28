import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle
from scipy.fft import fftn, ifftn  # Using scipy's FFTW interface for faster FFTs

# Constants
G = 6.67430e-11  # Gravitational constant (m^3 kg^-1 s^-2)
# Scale G for simulation
G_SCALED = 1.0

class NBodySimulation:
    def __init__(self, n_bodies=9, box_size=10.0, dt=0.01, random_seed=None, 
                 edge_distance=0.2, initial_velocity=0, grid_layout=True,
                 mass_distribution='uniform', mass_range=(0.5, 3.0),
                 mesh_size=64, pm_threshold=200):
        """Initialize the n-body simulation."""
        if random_seed is not None:
            np.random.seed(random_seed)
            
        self.n_bodies = n_bodies
        self.box_size = box_size
        self.dt = dt
        self.edge_distance = edge_distance
        
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
        
        # Generate masses based on selected distribution
        self.masses = self.generate_masses(mass_distribution, mass_range)
        
        # Set initial positions
        self.positions = np.zeros((n_bodies, 2))
        
        if grid_layout:
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
        else:
            # Original circular pattern
            radius = (1 - edge_distance) * (box_size / 2)
            angles = np.linspace(0, 2 * np.pi, n_bodies, endpoint=False)
            
            for i in range(n_bodies):
                self.positions[i] = [radius * np.cos(angles[i]), radius * np.sin(angles[i])]
        
        # Set initial velocities - scale with box size
        self.velocities = np.zeros((n_bodies, 2))
        
        # Scale velocity with box size (0-1 range input becomes 0 to box_size/5 actual velocity)
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
        
        # Colors for bodies
        self.colors = plt.cm.jet(np.linspace(0, 1, n_bodies))
    
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
    
    def generate_masses(self, distribution, mass_range):
        """Generate masses according to the specified distribution."""
        min_mass, max_mass = mass_range
        
        if distribution == 'equal':
            # All masses equal to the average of min and max
            return np.ones(self.n_bodies) * np.mean(mass_range)
        
        elif distribution == 'uniform':
            # Uniform distribution between min and max
            return np.random.uniform(min_mass, max_mass, self.n_bodies)
        
        elif distribution == 'gaussian':
            # Normal distribution with mean at center of range, std of 1/4 the range
            mean = np.mean(mass_range)
            std = (max_mass - min_mass) / 4
            masses = np.random.normal(mean, std, self.n_bodies)
            # Clip to ensure masses are within range
            return np.clip(masses, min_mass, max_mass)
        
        elif distribution == 'long-tail':
            # Power law distribution (more small masses, few large ones)
            # Using Pareto distribution shifted to start at min_mass
            alpha = 1.5  # Shape parameter for Pareto distribution
            # Generate raw values from Pareto
            raw_values = np.random.pareto(alpha, self.n_bodies)
            # Scale to fit within desired range
            scale_factor = (max_mass - min_mass) / max(np.max(raw_values), 1e-10)
            masses = min_mass + raw_values * scale_factor
            # Clip to ensure no values exceed max_mass
            return np.clip(masses, min_mass, max_mass)
        
        else:
            # Default to uniform if unknown distribution specified
            print(f"Unknown mass distribution '{distribution}', using uniform")
            return np.random.uniform(min_mass, max_mass, self.n_bodies)
    
    def calculate_acceleration_direct(self, positions, masses):
        """Calculate acceleration using direct N-body method (O(N²))."""
        n = len(positions)
        acceleration = np.zeros_like(positions)
        
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
                    # Acceleration magnitude (F = G*m1*m2/r^2, a = F/m1 = G*m2/r^2)
                    a_mag = G_SCALED * masses[j] / (r**2)
                    # Acceleration vector
                    acceleration[i] += a_mag * r_hat
        
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
        if self.use_pm_method:
            return self.calculate_acceleration_pm(positions, masses)
        else:
            return self.calculate_acceleration_direct(positions, masses)
    
    def update(self):
        """Update positions and velocities using Velocity Verlet integration."""
        # Calculate acceleration at current positions
        accelerations = self.calculate_acceleration(self.positions, self.masses)
        
        # Update positions and half-step velocities
        self.positions += self.velocities * self.dt + 0.5 * accelerations * self.dt**2
        self.velocities += 0.5 * accelerations * self.dt
        
        # Calculate acceleration at new positions
        new_accelerations = self.calculate_acceleration(self.positions, self.masses)
        
        # Complete velocity update
        self.velocities += 0.5 * new_accelerations * self.dt
        
        # Update trajectories
        for i in range(self.n_bodies):
            self.trajectories[i].append(self.positions[i].copy())
    
    def animate(self, num_steps=1000, skip_steps=1):
        """Create and display the animation of the n-body system."""
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xlim(-self.box_size/2, self.box_size/2)
        ax.set_ylim(-self.box_size/2, self.box_size/2)
        ax.set_aspect('equal')
        
        # Show method used in title
        method = "Particle-Mesh" if self.use_pm_method else "Direct N-Body"
        ax.set_title(f'N-Body Simulation (N={self.n_bodies}, {method})')
        
        # Initialize circles for bodies
        circles = []
        trajectory_lines = []
        
        # Scale body size with box_size
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
            
            return circles + trajectory_lines
        
        def animate(frame):
            """Perform animation step."""
            # Run multiple physics steps per frame for smoother animation
            for _ in range(skip_steps):
                self.update()
            
            # Update circle positions
            for i, circle in enumerate(circles):
                circle.center = self.positions[i]
                
                # Adjust trail length based on number of bodies
                if self.n_bodies <= 100:
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
            
            return circles + trajectory_lines
        
        anim = FuncAnimation(fig, animate, frames=num_steps, 
                             init_func=init, blit=True, interval=20)
        
        plt.tight_layout()
        plt.show()
        
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
    n_bodies = get_parameter("Number of bodies", 9, int)
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
    
    box_size = get_parameter("Box size", 10.0, float)
    
    # Ask for layout type first
    layout_choice = get_choice("Initial arrangement", ["grid", "circular"], "grid")
    grid_layout = layout_choice == "grid"
    
    # Only ask for edge_distance if circular layout is chosen
    if not grid_layout:
        edge_distance = get_parameter("Edge distance from box boundary (fraction, 0-1)", 0.2, float)
    else:
        edge_distance = 0.2  # Default value, will be used for grid spacing calculation
    
    # Mass distribution options
    mass_distribution = get_choice(
        "Mass distribution", 
        ["equal", "uniform", "gaussian", "long-tail"], 
        "uniform"
    )
    
    # Mass range (only if not using equal masses)
    if mass_distribution != "equal":
        min_mass = get_parameter("Minimum mass", 0.5, float)
        max_mass = get_parameter("Maximum mass", 3.0, float)
        if min_mass >= max_mass:
            print("Minimum mass must be less than maximum mass. Using defaults.")
            min_mass, max_mass = 0.5, 3.0
        mass_range = (min_mass, max_mass)
    else:
        mass_value = get_parameter("Mass value", 1.0, float)
        mass_range = (mass_value, mass_value)
    
    # Velocity input (now in range 0-1, will be scaled with box size)
    vel_input = input("Initial velocity (0-1 or 'random') [0]: ").strip().lower()
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
        grid_layout=grid_layout,
        mass_distribution=mass_distribution,
        mass_range=mass_range,
        mesh_size=mesh_size,
        pm_threshold=pm_threshold
    )
    
    # Run animation
    print("\nStarting simulation...")
    anim = sim.animate()