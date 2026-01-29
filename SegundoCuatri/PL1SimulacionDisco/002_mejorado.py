"""
Simular discos (puntos) dentro de una caja rebotando en las paredes
u.a.
"""

import numpy as np
import matplotlib.pyplot as plt

# ------------------ Parámetros ------------------
number_particles = 100
L = 10.0
dt = 0.01
n = 10000
v0 = 5
# Velocidad inicial 
v0 = np.ones(number_particles)*v0

# ------------------ Inicialización ------------------
angle = 2 * np.pi * np.random.rand(number_particles)
pos = L * np.random.rand(number_particles, 2)  

vel = np.zeros((number_particles, 2))
vel[:, 0] = v0 * np.cos(angle)
vel[:, 1] = v0 * np.sin(angle)

for i in range(number_particles):
    print(f"Partícula {i}")
    print(f"  ángulo = {angle[i]:.3f} rad")
    print(f"  posición = (x={pos[i,0]:.3f}, y={pos[i,1]:.3f})")
    print(f"  velocidad = (vx={vel[i,0]:.3f}, vy={vel[i,1]:.3f})")
    print(f"  |v| = {np.linalg.norm(vel[i]):.3f}")
    print("-" * 30)

# ------------------ Plot (scatter) ------------------
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_title("Simulación de partículas (scatter)")
ax.set_xlim(0, L)
ax.set_ylim(0, L)
ax.set_aspect("equal", adjustable="box")

plt.ion()

sc = ax.scatter(pos[:, 0], pos[:, 1], s=80)  

fig.canvas.draw()
fig.canvas.flush_events()

# ------------------ Simulación ------------------
for step in range(n):
    pos += vel * dt

    # Rebote en X con clamp
    left = pos[:, 0] < 0
    right = pos[:, 0] > L
    if np.any(left):
        pos[left, 0] = 0
        vel[left, 0] *= -1
    if np.any(right):
        pos[right, 0] = L
        vel[right, 0] *= -1

    # Rebote en Y con clamp
    bottom = pos[:, 1] < 0
    top = pos[:, 1] > L
    if np.any(bottom):
        pos[bottom, 1] = 0
        vel[bottom, 1] *= -1
    if np.any(top):
        pos[top, 1] = L
        vel[top, 1] *= -1

    if step % 10 == 0:
        sc.set_offsets(pos)     
        fig.canvas.draw_idle()
        plt.pause(0.01)

plt.ioff()
plt.show()

print("Pos final:\n", pos)
print("Vel final:\n", vel)
