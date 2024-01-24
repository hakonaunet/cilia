import time

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter
from numba import njit, prange

# Constants
N = 10  # Size of lattice nxn
K = 1.0  # Average coupling constant

# Initialize the angles and frequencies
angles = 2 * np.pi * np.random.rand(N, N)
frequencies = 0.95 + 0.1 * np.random.rand(N, N)

@njit
def get_coupling(r, coupling_type):
    if coupling_type == 1:  # uniform
        return K
    elif coupling_type == 2:  # one_over_r
        return K / (r + 1e-10)
    elif coupling_type == 3:  # one_over_r_squared
        return K / (r**2 + 1e-10)
    elif coupling_type == 4:  # nearest_neighbor
        return K if r < 1.1 else 0
    else:
        return 0

@njit(parallel=True)
def kuramoto_dtheta_dt(angles, frequencies, coupling_type):
    N, _ = angles.shape
    dtheta = np.zeros_like(angles)
    for i in prange(N):  # Replace this loop with prange
        for j in prange(N):  # And this loop as well
            sum_term = 0.0
            if coupling_type != 5:
                for x in range(N):
                    for y in range(N):
                        r = np.sqrt((x - i)**2 + (y - j)**2)
                        sum_term += get_coupling(r, coupling_type) * np.sin(angles[x, y] - angles[i, j])
                dtheta[i, j] = frequencies[i, j] + sum_term
            else:
                # Only nearest neighbors
                for x in range(i - 1, i + 2):
                    for y in range(j - 1, j + 2):
                        if x == i and y == j:
                            continue
                        r = np.sqrt((x - i)**2 + (y - j)**2)
                        sum_term += get_coupling(r, coupling_type) * np.sin(angles[x, y] - angles[i, j])
                dtheta[i, j] = frequencies[i, j] + sum_term
    return dtheta

def format_time(seconds):
    """Convert seconds to a string format: H:M:S"""
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{int(hours)}:{int(minutes):02}:{int(seconds):02}"

# Pre-compute the simulation data
NUM_FRAMES = 10000
angles_list = [angles.copy()]

start_time = time.time()  # Mark the start time

for i in range(NUM_FRAMES):
    dt = 0.01
    dtheta = kuramoto_dtheta_dt(angles, frequencies, 4)  # 4 corresponds to "uniform" coupling
    angles += dt * dtheta
    mod_angles = angles % (2 * np.pi)
    if i % 50 == 0:
        angles_list.append(mod_angles.copy())
    
    elapsed_time = time.time() - start_time  # Calculate elapsed time
    #print(f"Calculating frame {i+1}/{NUM_FRAMES}... Elapsed Time: {format_time(elapsed_time)}", end="\r", flush=True)

# Now, animate the stored angles
def update(frame):
    im.set_array(angles_list[frame])
    return im,

fig, ax = plt.subplots()
im = ax.imshow(angles_list[0], cmap='hsv', animated=True)

# Add a colorbar
cbar = fig.colorbar(im, ax=ax)
cbar.set_label('Cosine Value')

ani = animation.FuncAnimation(fig, update, frames=200, interval=50, blit=True)
ani.save("animation.gif", writer=PillowWriter(fps=100))