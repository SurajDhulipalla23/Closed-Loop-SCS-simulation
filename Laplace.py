import numpy as np
import matplotlib.pyplot as plt

def solve_laplaces_equation(grid_size=100, source_position=(50, 50), point_of_interest=(57, 50), source_strength=1):
    """
    Solves Laplace's equation with specified conditions and calculates the potential at a given point.
    
    Parameters:
    - grid_size: The size of the grid (assuming a square grid).
    - source_position: The (x, y) position of the current source on the grid.
    - point_of_interest: The (x, y) position of the point where we want to calculate the potential.
    - source_strength: The strength of the source at the electrode position.
    
    Returns:
    - voltage_grid: A 2D array of voltage values across the grid.
    - potential_at_point: The electric potential at the specified point of interest.
    """
    # Initialize the grid
    voltage_grid = np.zeros((grid_size, grid_size))
    
    # Set boundaries to ground (0V), simulate with edges of the grid
    # Note: In a more complex model, you might have different shapes or regions set to ground
    
    # Apply source strength at electrode position
    x_source, y_source = source_position
    voltage_grid[x_source, y_source] = source_strength
    
    # Placeholder for SCS lead shaft and inactive electrodes
    # In a full implementation, you would define these areas and apply specific conditions
    
    # Iterative solver parameters
    tolerance = 1e-4  # Convergence tolerance
    max_iterations = 10000  # Max iterations to prevent infinite loops
    iteration = 0
    delta = 1  # Initial delta for convergence check
    
    while delta > tolerance and iteration < max_iterations:
        iteration += 1
        delta = 0  # Reset delta for this iteration
        new_grid = voltage_grid.copy()  # Create a copy for simultaneous updates
        
        for i in range(1, grid_size-1):
            for j in range(1, grid_size-1):
                # Check for insulating areas and skip updating
                #if (i, j) in insulating_areas:
                #    continue
                
                # Calculate the new value using the average of neighbors
                value = 0.25 * (voltage_grid[i+1, j] + voltage_grid[i-1, j] +
                                voltage_grid[i, j+1] + voltage_grid[i, j-1])
                new_grid[i, j] = value
                
                # Update delta
                delta = max(delta, abs(value - voltage_grid[i, j]))
        
        voltage_grid = new_grid  # Update the grid
    
    # Calculate the potential at the point of interest
    x_interest, y_interest = point_of_interest
    potential_at_point = voltage_grid[x_interest, y_interest]
    
    if iteration == max_iterations:
        print("Solution did not converge.")
    else:
        print(f"Converged in {iteration} iterations.")
    
    return voltage_grid, potential_at_point

# Grid size in centimeters (assuming each unit is 1 cm for simplicity)
grid_size = 100
# Source position, assuming center for simplicity
source_position = (50, 50)
# Point of interest, 6.3 cm away from the source, directly along the x-axis for simplicity
point_of_interest = (int(50 + 7), 50)

# Solve Laplace's equation and calculate potential at a point of interest
voltage_grid, potential_at_point = solve_laplaces_equation(grid_size, source_position, point_of_interest)

print(f"Electric potential at point 6.3 cm from the source: {potential_at_point} V (arbitrary units)")

# Plot the results
plt.imshow(voltage_grid)
plt.show()
