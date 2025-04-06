import numpy as np

import matplotlib.pyplot as plt
import matplotlib.animation as animation

from scipy.integrate import solve_ivp

x0 = np.array([1.0, 0.0, -35 * (np.pi / 180), 0.0])
m = 1
M = 2
g = 9.81
l = 1

X = 0.5
Y = 1.5
Z = 2
W = 1
t_span = (0, 10)

a = m * g / M
b = (M + m) * g / (M * l)

A = np.array([
    [0, 1, 0, 0],
    [0, 0, m*g / M, 0],
    [0, 0, 0, 1],
    [0, 0, (M + m)*g/ M * l, 0]
])

B = np.array([
    0,
    1 / M,
    0,
    -1 / (M * l)
])

print(f"Rank of B is {np.linalg.matrix_rank(B)}")

qs = np.zeros(4, dtype = object)
qs[3] = B
qs[2] = A @ B
qs[1] = (A @ A) @ B - b * B
qs[0] = (A @ A @ A) @ B - b * (A @ B)

# Convert from column major to row major.
Tinv = np.transpose(np.array([qs[0], qs[1], qs[2], qs[3]]))
T = np.linalg.inv(Tinv)

Acanon = T @ A @ Tinv
Bcanon = T @ B

print(Acanon)

C3 = X + Y + Z + W
C2 = X * (Y + Z + W) + Y * (Z + W) + Z * W
C1 = X * (Y * Z + Y * W + Z * W) + Y * Z * W
C0 = X * Y * Z * W

# s^4 + 10s^3 + 35s^2 + 50s + 24
fs = np.zeros(4, dtype = float)
# fs[3] = -Acanon[3, 3] - 10
# fs[2] = -Acanon[3, 2] - 35
# fs[1] = -Acanon[3, 1] - 50
# fs[0] = -Acanon[3, 0] - 24
fs[3] = -Acanon[3, 3] - C3
fs[2] = -Acanon[3, 2] - C2
fs[1] = -Acanon[3, 1] - C1
fs[0] = -Acanon[3, 0] - C0

Fcanon = np.array([fs[0], fs[1], fs[2], fs[3]])
F = Fcanon @ T

Acontrolled = Acanon + np.outer(Bcanon, Fcanon)
print(Acontrolled)
print(np.linalg.eigvals(Acontrolled))

def dynamics(t, x):
    return Acontrolled @ x

t_eval = np.linspace(*t_span, 500)
sol = solve_ivp(dynamics, t_span, x0, t_eval = t_eval)

plt.figure(figsize = (10, 6))
plt.plot(sol.t, sol.y.T)
plt.xlabel("Time (s)")
plt.ylabel("States")
plt.legend(["x (cart pos)", "$\\dot{x}$ (cart velocity)", "$\\theta$ (angle)", "$\\dot{\\theta}}$ (angular velocity)"])
# plt.show()

cart_width = 0.5
cart_height = 0.3

# fig, ax = plt.subplots(figsize = (15, 10))
fig, ax = plt.subplots()

x_max = np.abs(sol.y[0]).max() + cart_width

ax.grid()
ax.set_xlim(-x_max, x_max)
ax.set_ylim(-0.1, l + cart_height + 0.1)
ax.set_aspect("equal", adjustable = "box")

plt.title("Inverted Pendulum on a Cart")

line, = ax.plot([], [], 'k-', lw = 2)
cart = plt.Rectangle((0, 0), cart_width, cart_height, fc = "blue")

ax.add_patch(cart)

def init_state():
    line.set_data([], [])
    cart.set_xy((-cart_width / 2, -cart_height / 2))

    return line, cart

def update_state(i):
    cart_x, cart_vel, theta, dtheta = sol.y[:, i]

    pendulum_x = cart_x + l * np.sin(theta)
    pendulum_y = l * np.cos(theta)

    line.set_data([cart_x, pendulum_x], [cart_height / 2, pendulum_y])
    cart.set_xy((cart_x - cart_width / 2, -cart_height / 2))

    return line, cart

print("AAAA", len(t_span), 1000 * (t_span[1] - t_span[0]))

anim = animation.FuncAnimation(
    fig, update_state, frames = len(t_eval), #init_func = init_state,
    interval = (t_span[1] - t_span[0]), blit = True
)

plt.show()