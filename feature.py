import numpy as np

def pos2poly(state, p=2):
    r"""
    state: (x,y) or [(x1,y1), (x2,y2), ...]
    p: the maximum degree of the polynomial
    
    return: return: poly_basis: [N, (p+1)(p+2)/2]
    - [1, x, y, x^2, xy, y^2, ...]
    """
    state = np.array(state)
    if state.ndim == 1:
        state = state.reshape(1, -1)
    
    n_samples = state.shape[0]
    x = state[:, 0]
    y = state[:, 1]
    
    features = [np.ones(n_samples)]
    
    for i in range(1, p + 1):
        for j in range(i + 1):
            # i \in [1, p] j \in [0, i]
            # {x^(i-j) * y^j}
            features.append((x ** (i - j)) * (y ** j))
            
    return np.column_stack(features)

def pos2fourier(state, p=2):
    r"""
    state: (x,y) or [(x1,y1), (x2,y2), ...]
    p: the maximum degree of the fourier series
    
    return: fourier_basis: [N, (p+1)(p+2)/2]
    - [1, cos(\pi x), cos(\pi y), cos(2\pi x), cos(\pi (x+y)), cos(2\pi y), ...]
    """
    state = np.array(state)
    if state.ndim == 1:
        state = state.reshape(1, -1)
    
    n_samples = state.shape[0]
    x = state[:, 0]
    y = state[:, 1]
    
    features = [np.ones(n_samples)]
    
    for i in range(1, p + 1):
        for j in range(i + 1):
            # i \in [1, p] j \in [0, i]
            # {cos(\pi (x*(i-j) + y*j))}
            features.append(np.cos(np.pi * (x * (i - j) + y * j)))
            
    return np.column_stack(features)

def pos2fourierq(state, q=2):
    r"""
    state: (x,y) or [(x1,y1), (x2,y2), ...]
    q: cos(\pi(c_1 x + c_2 y)), and c_1, c_2 \in {0,...,q}
    
    return: fourier_basis: [N, (q+1)^2]
    - [1, cos(\pi x), cos(\pi y), cos(2\pi x), cos(\pi (x+y)), cos(2\pi y), ...]
    """
    state = np.array(state)
    if state.ndim == 1:
        state = state.reshape(1, -1)
    
    x = state[:, 0]
    y = state[:, 1]
    
    features = []
    
    for c1 in range(0, q + 1):
        for c2 in range(0, q + 1):
            features.append(np.cos(np.pi * (c1 * x + c2 * y)))
            
    return np.column_stack(features)

def _test():
    # [1, x, y]
    print(pos2poly((1, 2), p=1))

    # [1, x, y, x^2, xy, y^2]
    print(pos2poly((1, 2), p=2))
    print(pos2poly([(1, 2), (3, 4)], p=2))

    # [1, cos(\pi x), cos(\pi y)]
    print(pos2fourier((1, 2), p=1))

    # [1, cos(\pi x), cos(\pi y), cos(2\pi x), cos(\pi (x+y)), cos(2\pi y)]
    print(pos2fourier((1, 2), p=2))
    print(pos2fourier([(1, 2), (3, 4)], p=2))

    # [1, cos(\pi x), cos(\pi y), cos(\pi (x+y))]
    print(pos2fourierq((1, 2), q=1))

def _test_fit_curve():
    """
    Test function to demonstrate fitting a curve using different basis functions
    """
    # Generate some sample data
    n_samples = 100
    x = np.linspace(0, 1, n_samples)
    y = np.linspace(0, 1, n_samples)
    grid_x, grid_y = np.meshgrid(x, y)
    points = np.column_stack((grid_x.flatten(), grid_y.flatten()))
    
    # Create a target function: z = sin(2πx) * cos(2πy)
    z = np.sin(2 * np.pi * grid_x) * np.cos(2 * np.pi * grid_y)
    z_flat = z.flatten()
    
    # Fit using polynomial basis
    poly_features = pos2poly(points, p=4)
    poly_weights = np.linalg.lstsq(poly_features, z_flat, rcond=None)[0]
    poly_pred = poly_features @ poly_weights
    poly_error = np.mean((poly_pred - z_flat) ** 2)
    
    # Fit using Fourier basis
    fourier_features = pos2fourier(points, p=4)
    fourier_weights = np.linalg.lstsq(fourier_features, z_flat, rcond=None)[0]
    fourier_pred = fourier_features @ fourier_weights
    fourier_error = np.mean((fourier_pred - z_flat) ** 2)
    
    # Fit using Fourier Q basis
    fourierq_features = pos2fourierq(points, q=4)
    fourierq_weights = np.linalg.lstsq(fourierq_features, z_flat, rcond=None)[0]
    fourierq_pred = fourierq_features @ fourierq_weights
    fourierq_error = np.mean((fourierq_pred - z_flat) ** 2)
    
    print(f"Polynomial basis MSE: {poly_error:.6f}")
    print(f"Fourier basis MSE: {fourier_error:.6f}")
    print(f"Fourier Q basis MSE: {fourierq_error:.6f}")
    
    # You can add visualization code here if matplotlib is available
    try:
        import matplotlib.pyplot as plt
        title = 'Comparison of Different Basis Functions for Function Approximation'
        fig = plt.figure(title, figsize=(15, 5))
        fig.suptitle(title, fontsize=16)
        
        # Original function
        ax1 = fig.add_subplot(141, projection='3d')
        ax1.plot_surface(grid_x, grid_y, z, cmap='viridis', alpha=0.7)
        ax1.set_title('Original')
        
        # Polynomial approximation
        ax2 = fig.add_subplot(142, projection='3d')
        ax2.plot_surface(grid_x, grid_y, poly_pred.reshape(n_samples, n_samples), cmap='viridis', alpha=0.7)
        ax2.set_title(f'Polynomial (p=4)\nMSE: {poly_error:.6f}')
        
        # Fourier approximation
        ax3 = fig.add_subplot(143, projection='3d')
        ax3.plot_surface(grid_x, grid_y, fourier_pred.reshape(n_samples, n_samples), cmap='viridis', alpha=0.7)
        ax3.set_title(f'Fourier (p=4)\nMSE: {fourier_error:.6f}')
        
        # Fourier Q approximation
        ax4 = fig.add_subplot(144, projection='3d')
        ax4.plot_surface(grid_x, grid_y, fourierq_pred.reshape(n_samples, n_samples), cmap='viridis', alpha=0.7)
        ax4.set_title(f'Fourier Q (q=4)\nMSE: {fourierq_error:.6f}')
        
        plt.tight_layout()
        plt.show()
    except ImportError:
        print("Matplotlib not available for visualization")

if __name__ == "__main__":
    _test()
    _test_fit_curve()
