import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

# Constants
EARTH_RADIUS = 6371  # in km
MARS_RADIUS = 3389  # in km
SUN_DISTANCE_EARTH = 149.6e6  # km
SUN_DISTANCE_MARS = 227.9e6  # km
TRANSFER_TIME = 8  # Approximate months for Hohmann transfer
RETURN_WAIT = 26  # Approximate months until next launch window
RETURN_TIME = 8  # Approximate months for return transfer

# Time steps (months)
t = np.linspace(0, TRANSFER_TIME + RETURN_WAIT + RETURN_TIME, 500)

# Circular orbits around the Sun (simplified)
theta_earth = np.linspace(0, 2 * np.pi, 500)
theta_mars = np.linspace(0, 2 * np.pi * (SUN_DISTANCE_EARTH / SUN_DISTANCE_MARS)**1.5, 500)
earth_orbit_x = SUN_DISTANCE_EARTH * np.cos(theta_earth)
earth_orbit_y = SUN_DISTANCE_EARTH * np.sin(theta_earth)
mars_orbit_x = SUN_DISTANCE_MARS * np.cos(theta_mars)
mars_orbit_y = SUN_DISTANCE_MARS * np.sin(theta_mars)

# Approximate Hohmann Transfer Paths
transfer_x = np.linspace(SUN_DISTANCE_EARTH, SUN_DISTANCE_MARS, len(t) // 3)
transfer_y = np.sqrt(SUN_DISTANCE_MARS**2 - transfer_x**2)
return_x = np.linspace(SUN_DISTANCE_MARS, SUN_DISTANCE_EARTH, len(t) // 3)
return_y = np.sqrt(SUN_DISTANCE_EARTH**2 - return_x**2)

# Create 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel("X (km)")
ax.set_ylabel("Y (km)")
ax.set_zlabel("Z (km)")
ax.set_title("Mars Mission: Earth to Mars and Back")

# Plot the orbits
ax.plot(earth_orbit_x, earth_orbit_y, np.zeros_like(earth_orbit_x), label='Earth Orbit', color='blue')
ax.plot(mars_orbit_x, mars_orbit_y, np.zeros_like(mars_orbit_x), label='Mars Orbit', color='red')
ax.scatter([0], [0], [0], color='yellow', s=500, label='Sun')

# Plot spacecraft path
spacecraft, = ax.plot([], [], [], 'o-', color='white', markersize=5, label='Spacecraft')


def update(frame):
    if frame < len(transfer_x):
        x, y = transfer_x[frame], transfer_y[frame]
    elif frame < len(transfer_x) + len(t) // 3:
        x, y = SUN_DISTANCE_MARS, 0  # Stay on Mars
    else:
        x, y = return_x[frame - len(transfer_x) - len(t) // 3], return_y[frame - len(transfer_x) - len(t) // 3]

    spacecraft.set_data([x], [y])
    spacecraft.set_3d_properties([0])
    return spacecraft,


# Animate
ani = animation.FuncAnimation(fig, update, frames=len(t), interval=50, blit=False)
plt.legend()
plt.show()
