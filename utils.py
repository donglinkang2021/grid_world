from typing import Tuple, List, Callable
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
    ax.view_init(elev=30, azim=330)
    ax.set_box_aspect([2.5, 2.5, 1])
    
    # Set axis labels
    ax.set_xlabel('X (rows)', fontsize=12)
    ax.set_ylabel('Y (columns)', fontsize=12)
    ax.set_zlabel('Z (values)', fontsize=12)
    ax.set_title(title, fontsize=14)

    # Add color bar
    # fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
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
    fig = plt.figure(title, figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(
        Y_new, X_new, Z_new,
        cmap='viridis',
        alpha=0.8,
        edgecolor='none'
    )

    # Add gridlines on the interpolated surface
    for x_row in x_orig:
        # Get interpolated values along this row
        z_points = interp_func(y_new, [x_row] * 100, grid=False)
        ax.plot(y_new, [x_row] * 100, z_points.flatten(),
                color='black', linestyle='-', linewidth=1)

    for y_col in y_orig:
        # Get interpolated values along this column
        z_points = interp_func([y_col] * 100, x_new, grid=False)
        ax.plot([y_col] * 100, x_new, z_points.flatten(),
                color='black', linestyle='-', linewidth=1)

    # Set aspect ratio
    ax.view_init(elev=30, azim=330)
    ax.set_box_aspect([2.5, 2.5, 1])

    # Set axis labels
    ax.set_xlabel('X (rows)', fontsize=12)
    ax.set_ylabel('Y (columns)', fontsize=12)
    ax.set_zlabel('Z (values)', fontsize=12)
    ax.set_title(title, fontsize=14)

    # Add color bar
    # fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    plt.tight_layout()

def draw_prediction(weights:np.ndarray, m:int, n:int, title:str):
    # Generate original row and column coordinates (x for rows, y for columns)
    x_orig = np.arange(m)
    y_orig = np.arange(n)

    # Generate dense interpolation points (using 100 points in this example)
    x_new = np.linspace(0, n-1, 100)
    y_new = np.linspace(0, m-1, 100)


    # Generate grid point coordinates
    X_new, Y_new = np.meshgrid(x_new, y_new)

    phi = np.stack([X_new, Y_new, np.ones((100,100))], axis=-1)
    Z_new = (phi @ weights).squeeze()

    # Draw 3D surface plot
    fig = plt.figure(title, figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(
        Y_new, X_new, Z_new,
        cmap='viridis',
        alpha=0.8,
        edgecolor='none'
    )

    for x_row in x_orig:
        phi = np.stack([[x_row] * 100, y_new, np.ones(100)], axis=-1)
        z_points = (phi @ weights).squeeze()
        ax.plot(y_new, [x_row] * 100, z_points,
                color='black', linestyle='-', linewidth=1)

    for y_col in y_orig:
        phi = np.stack([x_new, [y_col]*100, np.ones(100)], axis=-1)
        z_points = (phi @ weights).squeeze()
        ax.plot([y_col]*100, x_new, z_points,
                color='black', linestyle='-', linewidth=1)

    # Set aspect ratio
    ax.view_init(elev=30, azim=330)
    ax.set_box_aspect([2.5, 2.5, 1])

    # Set axis labels
    ax.set_xlabel('X (rows)', fontsize=12)
    ax.set_ylabel('Y (columns)', fontsize=12)
    ax.set_zlabel('Z (values)', fontsize=12)
    ax.set_title(title, fontsize=14)

    # Add color bar
    # fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    plt.tight_layout()


def draw_curve(
        data1d_list:List[np.ndarray], 
        label_list:List[str],
        title:str = "State Value Error (RMSE) vs Episodes",
        xlabel:str = "Episode index",
        ylabel:str = "State Value Error (RMSE)"
    ):
    plt.figure(title, figsize=(8, 6))
    for data1d, label in zip(data1d_list, label_list):
        plt.plot(data1d, marker='.', linestyle='-', label=label)
    
    plt.title(title, fontsize=14)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.legend()
    
    plt.grid(alpha=0.3)
    plt.tight_layout()

def _test_draw_matrix():
    # Create a 5x5 matrix (example)
    Z = np.random.rand(5, 5)

    # Draw the 3D surface plot
    draw_matrix2d(Z, 'Original 3D Surface Plot')
    draw_matrix2d_smooth(Z, 'Smoothed 3D Surface Plot (k=1)')
    draw_matrix2d_smooth(Z, 'Smoothed 3D Surface Plot (k=2)', k=2)
    plt.show()

def _test_draw_curve():
    loss1 = np.random.randn(500) * 0.1
    loss2 = np.cumsum(loss1)  # Cumulative sum for a more realistic learning curve
    draw_curve([loss1, loss2], ["randn*0.1", "cumsum(randn*0.1)"], "Random Value", "x", "y")
    plt.show()

if __name__ == '__main__':
    _test_draw_matrix()
    # _test_draw_curve()