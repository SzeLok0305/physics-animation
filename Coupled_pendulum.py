import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle
from matplotlib.lines import Line2D

class CoupledPendulumSystem:
    def __init__(self, n_pendulums=3, lengths=None, masses=None, 
                 initial_angles=None, g=9.81, dt=0.01):
        """Initialize the coupled pendulum system."""
        self.n = n_pendulums
        self.g = g
        self.dt = dt
        
        # Set lengths for each pendulum
        if lengths is None:
            self.lengths = np.ones(n_pendulums)
        else:
            self.lengths = np.array(lengths)
            
        # Set masses for each pendulum
        if masses is None:
            self.masses = np.ones(n_pendulums)
        else:
            self.masses = np.array(masses)
        
        # Set initial angles (in degrees, convert to radians)
        # In this system, 0 degrees means pendulum pointing down (-y direction)
        if initial_angles is None:
            self.theta = np.zeros(n_pendulums)
        else:
            self.theta = np.array(initial_angles) * (np.pi / 180.0) - np.pi/2
            
        # Initialize angular velocities
        self.omega = np.zeros(n_pendulums)
        
        # For storing position history
        self.history = []
        self.save_positions()
        
        # Small damping coefficient
        self.damping = 0.005
    
    def save_positions(self):
        """Calculate and save the positions of all pendulum bobs."""
        positions = self.get_positions()
        self.history.append(positions)
    
    def get_positions(self):
        """
        Calculate the positions of all pendulum bobs.
        In this coordinate system:
          - The origin is at the fixed pivot point
          - Positive y is upward
          - 0 angle corresponds to pendulum pointing down (-y direction)
        """
        x = np.zeros(self.n+1)  # +1 for the fixed pivot point
        y = np.zeros(self.n+1)
        
        for i in range(1, self.n+1):
            # For each pendulum segment, calculate its endpoint
            # sin(theta) gives x component, -cos(theta) gives y component
            # (for 0 degrees = pointing down)
            x[i] = x[i-1] + self.lengths[i-1] * np.sin(self.theta[i-1])
            y[i] = y[i-1] - self.lengths[i-1] * np.cos(self.theta[i-1])
            
        return np.column_stack((x, y))
    
    def equations_of_motion(self, theta, omega):
        """
        Compute accelerations for the n-pendulum system.
        Uses a simplified and more stable approach for multi-pendulum dynamics.
        """
        n = self.n
        g = self.g
        
        # Initialize acceleration array
        alpha = np.zeros(n)
        
        # For a single pendulum, use a simpler and more stable formula
        if n == 1:
            alpha[0] = -(g / self.lengths[0]) * np.sin(theta[0]) - self.damping * omega[0]
            return alpha
        
        # For multi-pendulum systems, use a more careful approach
        for i in range(n):
            # Gravity term (primary driver of motion)
            alpha[i] = -(g / self.lengths[i]) * np.sin(theta[i])
            
            # Damping term
            alpha[i] -= self.damping * omega[i]
            
            # Coupling effect from previous pendulum (if not the first)
            if i > 0:
                # Calculate angle difference, ensure it's in [-π, π]
                angle_diff = theta[i-1] - theta[i]
                while angle_diff > np.pi:
                    angle_diff -= 2 * np.pi
                while angle_diff < -np.pi:
                    angle_diff += 2 * np.pi
                
                # Coupling terms with numeric safeguards
                sin_term = np.sin(angle_diff)
                cos_term = np.cos(angle_diff)
                
                # Centripetal term
                alpha[i] += (self.lengths[i-1] / self.lengths[i]) * (omega[i-1]**2) * sin_term
                
                # Motion transfer term
                if i > 0 and np.isfinite(alpha[i-1]):  # Check if previous acceleration is valid
                    alpha[i] += (self.masses[i-1] / self.masses[i]) * cos_term * alpha[i-1]
        
        # Apply limits to avoid numeric explosion
        max_alpha = 100.0  # Reasonable limit for angular acceleration
        alpha = np.clip(alpha, -max_alpha, max_alpha)
        
        return alpha
    
    def update(self):
        """Update the pendulum system using a numerically stable integration method."""
        dt = self.dt
        
        # Store current state
        theta_current = self.theta.copy()
        omega_current = self.omega.copy()
        
        # For numerical stability, use a simpler but more stable method for large angles
        if np.any(np.abs(omega_current) > 10):
            # Simple Euler with damping for stabilization
            alpha = self.equations_of_motion(theta_current, omega_current)
            self.omega = omega_current + alpha * dt
            self.theta = theta_current + self.omega * dt
            
            # Additional damping when things get too energetic
            if np.any(np.abs(self.omega) > 20):
                self.omega *= 0.95  # Stronger damping when needed
        else:
            # RK4 integration when the system is well-behaved
            # First step
            k1_theta = omega_current * dt
            k1_omega = self.equations_of_motion(theta_current, omega_current) * dt
            
            # Second step
            theta_temp = theta_current + 0.5 * k1_theta
            omega_temp = omega_current + 0.5 * k1_omega
            k2_theta = omega_temp * dt
            k2_omega = self.equations_of_motion(theta_temp, omega_temp) * dt
            
            # Third step
            theta_temp = theta_current + 0.5 * k2_theta
            omega_temp = omega_current + 0.5 * k2_omega
            k3_theta = omega_temp * dt
            k3_omega = self.equations_of_motion(theta_temp, omega_temp) * dt
            
            # Fourth step
            theta_temp = theta_current + k3_theta
            omega_temp = omega_current + k3_omega
            k4_theta = omega_temp * dt
            k4_omega = self.equations_of_motion(theta_temp, omega_temp) * dt
            
            # Final update
            self.theta = theta_current + (k1_theta + 2*k2_theta + 2*k3_theta + k4_theta) / 6.0
            self.omega = omega_current + (k1_omega + 2*k2_omega + 2*k3_omega + k4_omega) / 6.0
        
        # Ensure angles stay within reasonable range
        for i in range(self.n):
            # Keep angles in [-2π, 2π] range
            while self.theta[i] > 2*np.pi:
                self.theta[i] -= 2*np.pi
            while self.theta[i] < -2*np.pi:
                self.theta[i] += 2*np.pi
        
        # Save the new positions
        self.save_positions()
    
    def animate(self, num_steps=1000, skip_steps=1):
        """Create and display the animation of the pendulum system."""
        # Determine figure size based on pendulum lengths
        total_length = np.sum(self.lengths)
        fig_size = max(8, min(12, total_length * 1.2))
        
        fig, ax = plt.subplots(figsize=(fig_size, fig_size))
        
        # Set reasonable limits based on pendulum lengths
        margin = total_length * 0.1
        limit = total_length + margin
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)
        
        ax.set_aspect('equal')
        ax.grid(True)
        ax.set_title(f'Coupled {self.n}-Pendulum System')
        
        # Create line objects for pendulum rods
        lines = [Line2D([], [], lw=2, color='black') for _ in range(self.n)]
        for line in lines:
            ax.add_line(line)
        
        # Create circles for pendulum bobs
        circles = []
        colors = plt.cm.viridis(np.linspace(0, 0.8, self.n))
        
        for i in range(self.n):
            # Scale radius with mass
            radius = 0.05 * total_length * (self.masses[i] / np.max(self.masses))**0.5
            circle = Circle((0, 0), radius, color=colors[i], zorder=3)
            circles.append(circle)
            ax.add_patch(circle)
        
        # Add trail for the last pendulum
        trail_line, = ax.plot([], [], 'r-', alpha=0.3, lw=1)
        
        def init():
            """Initialize animation."""
            positions = self.get_positions()
            
            for i in range(self.n):
                lines[i].set_data([positions[i, 0], positions[i+1, 0]], 
                                 [positions[i, 1], positions[i+1, 1]])
                circles[i].center = positions[i+1, 0], positions[i+1, 1]
            
            trail_line.set_data([], [])
            
            return lines + circles + [trail_line]
        
        # Run simulation for initial history
        for _ in range(20):  # Pre-run to build up history
            self.update()
        
        def animate(frame):
            """Perform animation step."""
            # Update pendulum multiple times per frame
            for _ in range(skip_steps):
                self.update()
            
            positions = self.history[-1]
            
            # Update lines and circles
            for i in range(self.n):
                lines[i].set_data([positions[i, 0], positions[i+1, 0]], 
                                 [positions[i, 1], positions[i+1, 1]])
                circles[i].center = (positions[i+1, 0], positions[i+1, 1])
            
            # Update trail (last 100 positions of the last pendulum)
            trail_length = min(100, len(self.history))
            x_trail = [pos[-1, 0] for pos in self.history[-trail_length:]]
            y_trail = [pos[-1, 1] for pos in self.history[-trail_length:]]
            trail_line.set_data(x_trail, y_trail)
            
            return lines + circles + [trail_line]
        
        anim = FuncAnimation(fig, animate, frames=num_steps, init_func=init, 
                             blit=True, interval=20)
        
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

# Run the simulation
if __name__ == "__main__":
    print("N-Link Coupled Pendulum Simulation")
    print("---------------------------------")
    print("Enter parameters (or press Enter for defaults)")
    
    n_pendulums = get_parameter("Number of pendulums", 3, int)
    
    # Get lengths
    lengths_input = input(f"Pendulum lengths (comma-separated) [{'1.0, ' * (n_pendulums-1) + '1.0'}]: ").strip()
    if lengths_input:
        try:
            lengths = [float(x.strip()) for x in lengths_input.split(',')]
            if len(lengths) != n_pendulums:
                print(f"Expected {n_pendulums} values, using default lengths of 1.0")
                lengths = np.ones(n_pendulums)
        except ValueError:
            print("Invalid input for lengths, using default of 1.0 for all")
            lengths = np.ones(n_pendulums)
    else:
        lengths = np.ones(n_pendulums)
    
    # Get masses
    masses_input = input(f"Pendulum masses (comma-separated) [{'1.0, ' * (n_pendulums-1) + '1.0'}]: ").strip()
    if masses_input:
        try:
            masses = [float(x.strip()) for x in masses_input.split(',')]
            if len(masses) != n_pendulums:
                print(f"Expected {n_pendulums} values, using default masses of 1.0")
                masses = np.ones(n_pendulums)
        except ValueError:
            print("Invalid input for masses, using default of 1.0 for all")
            masses = np.ones(n_pendulums)
    else:
        masses = np.ones(n_pendulums)
    
    # Get initial angles
    angles_input = input(f"Initial angles in degrees (comma-separated) [{'0.0, ' * (n_pendulums-1) + '0.0'}]: ").strip()
    if angles_input:
        try:
            angles = [float(x.strip()) for x in angles_input.split(',')]
            if len(angles) != n_pendulums:
                print(f"Expected {n_pendulums} values, using default angles of 0.0")
                angles = np.zeros(n_pendulums)
        except ValueError:
            print("Invalid input for angles, using default of 0.0 for all")
            angles = np.zeros(n_pendulums)
    else:
        angles = np.zeros(n_pendulums)
    
    # Create system with user parameters
    pendulum_system = CoupledPendulumSystem(
        n_pendulums=n_pendulums,
        lengths=lengths,
        masses=masses,
        initial_angles=angles,
    )
    
    # Run animation
    print("\nStarting simulation...")
    anim = pendulum_system.animate()