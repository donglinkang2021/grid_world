from typing import Tuple
import matplotlib.patches as patches
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline

# RGB colors
COLOR_FORBID = (0.9290,0.6940,0.125)
COLOR_TARGET = (0.3010,0.7450,0.9330)
COLOR_POLICY = (0.4660,0.6740,0.1880)
COLOR_TRAJECTORY = (0, 1, 0)
COLOR_AGENT = (0,0,1)

def state2rect(state:Tuple[int, int], color:Tuple[float, float, float]):
    return patches.Rectangle(
        xy=(
            state[0]-0.5, 
            state[1]-0.5
        ),
        width=1, height=1, 
        linewidth=1, 
        edgecolor=color, facecolor=color
    )

def draw_matrix2d(matrix2d:np.ndarray, title:str):
    m, n = matrix2d.shape
    x = np.arange(m)
    y = np.arange(n)
    X, Y = np.meshgrid(x, y)
    Z = matrix2d
    
    # Draw 3D surface plot
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(Y, X, Z, cmap='viridis', edgecolor='none')
    
    # Set aspect ratio
    ax.view_init(elev=30, azim=345)
    ax.set_box_aspect([2.5, 2.5, 1])
    
    # Set axis labels
    ax.set_xlabel('X (rows)', fontsize=12)
    ax.set_ylabel('Y (columns)', fontsize=12)
    ax.set_zlabel('Z (values)', fontsize=12)
    ax.set_title(title, fontsize=14)

    # Add color bar
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    plt.tight_layout()

def draw_matrix2d_smooth(matrix2d:np.ndarray, title:str, k:int=1):

    m, n = matrix2d.shape  # Get number of rows and columns

    # Generate original row and column coordinates (x for columns, y for rows)
    x_orig = np.arange(m)
    y_orig = np.arange(n)

    # Create bilinear interpolation function
    interp_func = RectBivariateSpline(y_orig, x_orig, matrix2d, kx=k, ky=k)

    # Generate dense interpolation points (using 100 points in this example)
    x_new = np.linspace(0, n-1, 100)
    y_new = np.linspace(0, m-1, 100)

    # Calculate interpolated Z values
    Z_new = interp_func(y_new, x_new, grid=True)

    # Generate grid point coordinates
    X_new, Y_new = np.meshgrid(x_new, y_new)

    # Draw 3D surface plot
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(Y_new, X_new, Z_new, cmap='viridis', edgecolor='none')

    # Set aspect ratio
    ax.view_init(elev=30, azim=345)
    ax.set_box_aspect([2.5, 2.5, 1])

    # Set axis labels
    ax.set_xlabel('X (columns)', fontsize=12)
    ax.set_ylabel('Y (rows)', fontsize=12)
    ax.set_zlabel('Z (values)', fontsize=12)
    ax.set_title(title, fontsize=14)

    # Add color bar
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    plt.tight_layout()

if __name__ == '__main__':
    # Create a 5x5 matrix (example)
    Z = np.random.rand(5, 5)

    # Draw the 3D surface plot
    draw_matrix2d(Z, 'Original 3D Surface Plot')
    draw_matrix2d_smooth(Z, 'Smoothed 3D Surface Plot (k=1)')
    draw_matrix2d_smooth(Z, 'Smoothed 3D Surface Plot (k=2)', k=2)
    plt.show()