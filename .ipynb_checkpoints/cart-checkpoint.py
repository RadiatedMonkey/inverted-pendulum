import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from scipy.integrate import solve_ivp

# The initial condition.
x0 = np.array([0.0, -1.0, 15.0 * (np.pi / 180), 0.0], dtype = float)

# Parameters.
m = 1
M = 1
g = 9.81
l = 1

# These are the eigenvalues we want the system to have (multiplied by -1).
e1 = 4
e2 = 3 
e3 = 2 
e4 = 1
t_span = (0, 5)

# The a and b constants defined in the article.
a = m * g / M
b = (M + m) * g / (M * l)

# Linearized matrix of the system
A = np.array([
    [0, 1, 0, 0],
    [0, 0, m*g / M, 0],
    [0, 0, 0, 1],
    [0, 0, (M + m)*g/ M * l, 0]
])

# The linearized input matrix.
B = np.array([
    0,
    1 / M,
    0,
    -1 / (M * l)
])

# These are the q_n vectors used to create the similarity transformation.
qs = np.zeros(4, dtype = object)
qs[3] = B
qs[2] = A @ B
qs[1] = (A @ A) @ B - b * B
qs[0] = (A @ A @ A) @ B - b * (A @ B)

# Create the matrices for the similarity transformation.
T_inv = np.transpose(np.array([qs[0], qs[1], qs[2], qs[3]]))
T = np.linalg.inv(T_inv)

# The controllable canonical forms of A and B.
A_canon = T @ A @ T_inv
B_canon = T @ B

print(f"The controllable canonical form of A is \n{A_canon}\nand B is {B_canon}")

# Computes the coefficients of the characteristic polynomial that gives the desired eigenvalues
# (p1, p2, p3, p4) as roots. These are just the coefficients of (s + p1)(s + p2)(s + p3)(s + p4).
c3 = e1 + e2 + e3 + e4
c2 = e1 * (e2 + e3 + e4) + e2 * (e3 + e4) + e3 * e4
c1 = e1 * (e2 * e3 + e2 * e4 + e3 * e4) + e2 * e3 * e4
c0 = e1 * e2 * e3 * e4

# Determines the feedback matrix, using the previously computed coefficients.
fs = np.zeros(4, dtype = float)
fs[3] = -A_canon[3, 3] - c3
fs[2] = -A_canon[3, 2] - c2
fs[1] = -A_canon[3, 1] - c1
fs[0] = -A_canon[3, 0] - c0

# The initial feedback matrix is in canonical form, we need to
# transform it back to regular space.
F_canon = np.array([fs[0], fs[1], fs[2], fs[3]])

# Determines the full stabilized system by adding the feedback matrix as input.
A_controlled = T_inv @ (A_canon + np.outer(B_canon, F_canon)) @ T
print(f"The stabilised system matrix is \n{A_controlled}")
print(f"with eigenvalues {np.linalg.eigvals(A_controlled)}")

# Evaluates the system
def dynamics(t, x):
    return A_controlled @ x

# Solve the system using Scipy.
t_eval = np.linspace(*t_span, 500)
sol = solve_ivp(dynamics, t_span, x0, t_eval = t_eval)

# input_force = (F_canon @ T) @ sol.y[:, i]
# forces = (F_canon @ T) @ sol.y

plt.figure(figsize = (10, 6))
plt.plot(sol.t, sol.y.T)
# plt.plot(forces)
plt.xlabel("Time (s)")
plt.ylabel("States")
plt.legend(["x (cart position)", "$\\frac{dx}{dt}$ (cart velocity)", "$\\theta$ (angle)", "$\\frac{d\\theta}{dt}$ (angular velocity)"])

cart_width = 0.5
cart_height = 0.3

fig, ax = plt.subplots()

# Ensure the cart stays on screen by setting the x limits accordingly.
x_max = np.abs(sol.y[0]).max() + cart_width

ax.grid()
ax.set_xlim(-x_max, x_max)
ax.set_ylim(-0.1, l + cart_height + 0.1)
ax.set_aspect("equal", adjustable = "box")

plt.title("Stabilised Inverted Pendulum on a Cart")

pendulum, = ax.plot([], [], 'k-', lw = 2)
cart = plt.Rectangle((0, 0), cart_width, cart_height, fc = "blue")
text = ax.text(0.05, 0.9, "", transform = ax.transAxes)

ax.add_patch(cart)

# Initialises the cart and pendulum
def init_state():
    pendulum.set_data([], [])
    cart.set_xy((-cart_width / 2, -cart_height / 2))

    return pendulum, cart

# Called each frame to update the animation to the next state.
def update_state(i):
    cart_x, cart_vel, theta, dtheta = sol.y[:, i]

    input_force = (F_canon @ T) @ sol.y[:, i]
    print(f"Force = {input_force}")
    text.set_text(f"Force: {input_force:.2f}")

    pendulum_x = cart_x + l * np.sin(theta)
    pendulum_y = l * np.cos(theta)

    pendulum.set_data([cart_x, pendulum_x], [cart_height / 2, pendulum_y])
    cart.set_xy((cart_x - cart_width / 2, -cart_height / 2))

    return pendulum, cart, text

anim = animation.FuncAnimation(
    fig, update_state, frames = len(t_eval), #init_func = init_state,
    interval = (t_span[1] - t_span[0]), blit = True
)

plt.show()