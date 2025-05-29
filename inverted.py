import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from scipy.integrate import solve_ivp
from scipy.optimize._numdiff import approx_derivative

M = 5
m = 1
l = 1
F = 0
g = 9.81
x0 = [0, 0, np.pi / 2, 0]                   # Initial conditions
equi_x0 = np.array([0, 0, 0, 0])    # Equilibrium point to linearize around

# These are the eigenvalues we want the system to have (multiplied by -1).
e1 = 4
e2 = 3 
e3 = 2 
e4 = 1

t_span = (0, 5)
t_eval = np.linspace(*t_span, 500)

def system(t, X, F = 0):
    x, v, theta, omega = X

    A = np.array([
        [(M + m) / (m * l), -np.cos(theta)],
        [-np.cos(theta), l]
    ])
    
    b = np.array([
        F + (omega ** 2) * np.sin(theta),
        g * np.sin(theta)
    ])
    
    dt = np.linalg.solve(A, b)
    
    x_dot = v
    theta_dot = omega
    v_dot, omega_dot = dt

    return x_dot, v_dot, theta_dot, omega_dot

def A_wrapper(X):
    return system(0, X, 0)

def B_wrapper(u):
    F, = u
    return system(0, equi_x0, F)

A_jac = approx_derivative(A_wrapper, equi_x0)
B_jac = approx_derivative(B_wrapper, 0)

# Determine the characteristic polynomial of A_jac
A_coeffs = np.poly(A_jac)

print(f"A_coeffs = {A_coeffs}")

# These are the q_n vectors used to create the similarity transformation.
qs = np.zeros(4, dtype = object)
qs[3] = B_jac
qs[2] = A_jac @ B_jac + A_coeffs[1] * B_jac
qs[1] = (A_jac @ A_jac) @ B_jac - A_coeffs[1] * (A_jac @ B_jac) + A_coeffs[2] * B_jac
qs[0] = (A_jac @ A_jac @ A_jac) @ B_jac + A_coeffs[1] * (A_jac @ A_jac) @ B_jac + A_coeffs[2] * A_jac @ B_jac + A_coeffs[3] * B_jac

# Create the matrices for the similarity transformation.
T_inv = np.transpose(np.array([qs[0], qs[1], qs[2], qs[3]]))
T = np.linalg.inv(T_inv)

# The controllable canonical forms of A and B.
A_canon = (T @ A_jac @ T_inv)[0]
B_canon = np.ravel((T @ B_jac)[0])

print(f"The controllable canonical form of A is \n{A_canon}\nand B is {B_canon}")

# Computes the coefficients of the characteristic polynomial that gives the desired eigenvalues
# (p1, p2, p3, p4) as roots. These are just the coefficients of (s + p1)(s + p2)(s + p3)(s + p4).
c3 = e1 + e2 + e3 + e4
c2 = e1 * (e2 + e3 + e4) + e2 * (e3 + e4) + e3 * e4
c1 = e1 * (e2 * e3 + e2 * e4 + e3 * e4) + e2 * e3 * e4
c0 = e1 * e2 * e3 * e4

# Determines the feedback matrix, using the previously computed coefficients.
fs = np.empty(4, dtype = float)
fs[3] = -A_canon[3, 3] - c3
fs[2] = -A_canon[3, 2] - c2
fs[1] = -A_canon[3, 1] - c1
fs[0] = -A_canon[3, 0] - c0

# The initial feedback matrix is in canonical form, we need to
# transform it back to regular space.
F_canon = np.array([fs[0], fs[1], fs[2], fs[3]])

# Determines the full stabilized system by adding the feedback matrix as input.
A_controlled = (T_inv @ (A_canon + np.outer(B_canon, F_canon)) @ T)[0]

def linear_system(t, X):
    return A_jac @ X 

def controlled_system(t, X):
    return A_controlled @ X

sol = solve_ivp(system, t_span, x0, t_eval = t_eval)
# linear_sol = solve_ivp(linear_system, t_span, x0, t_eval = t_eval)
linear_sol = solve_ivp(controlled_system, t_span, x0, t_eval = t_eval)
# sol = solve_ivp(controlled_system, t_span, x0, t_eval = t_eval)

errors = np.abs(sol.y - linear_sol.y)

plt.figure(figsize = (10, 6))
plt.plot(sol.t, np.array([errors[0], errors[2]]).T)

plt.xlabel("Time (s)")
plt.ylabel("Error")
plt.legend([
    "$\\Delta x$",
    # "$\\Delta v$",
    "$\\Delta \\theta$",
    # "$\\Delta \\omega$"
])

plt.figure(figsize = (10, 6))

plt.plot(sol.t, sol.y[0], '-.', color = 'red')
plt.plot(linear_sol.t, linear_sol.y[0], color = 'red')

plt.plot(sol.t, sol.y[2], '-.', color = 'green')
plt.plot(linear_sol.t, linear_sol.y[2], color = 'green')

# plt.plot(forces)
plt.xlabel("Time (s)")
plt.ylabel("States")
plt.legend([
    "x (cart position)", 
    "linear x (cart position)",
    # "$\\frac{dx}{dt}$ (cart velocity)", 
    "$\\theta$ (angle)", 
    # "$\\frac{d\\theta}{dt}$ (angular velocity)",
    # "linear $\\frac{dx}{dt}$ (cart velocity)", 
    "linear $\\theta$ (angle)", 
    # "linear $\\frac{d\\theta}{dt}$ (angular velocity)"
])

cart_width = 0.5
cart_height = 0.3

fig, ax = plt.subplots()

# Ensure the cart stays on screen by setting the x limits accordingly.
# x_max = np.abs(sol.y[0]).max() + cart_width
x_max = 5

ax.grid()
ax.set_xlim(-3, 3)
ax.set_ylim(-(l + cart_height + 0.1), l + cart_height + 0.1)
ax.set_aspect("equal", adjustable = "box")

plt.title("Pendulum on a Cart")

pendulum1, = ax.plot([], [], 'k-', lw = 2, alpha = 0.5)
cart1 = plt.Rectangle((0, 0), cart_width, cart_height, fc = "red", alpha = 0.5, label = "Nonlinear")
# text2 = ax.text(0.05, 0.9, "", transform = ax.transAxes)
bob1 = plt.Circle((0, 0), 0.05, alpha = 0.5, color = "red")
trace1, = ax.plot([], [], 'r--', lw = 1)

ax.add_patch(cart1)
ax.add_patch(bob1)

trace1_x, trace1_y = [], []

pendulum2, = ax.plot([], [], 'k-', lw = 2)
cart2 = plt.Rectangle((0, 0), cart_width, cart_height, fc = "blue", label = "Linear")
# text1 = ax.text(0.05, 0.9, "", transform = ax.transAxes)
bob2 = plt.Circle((0, 0), 0.05, color = "blue")
trace2, = ax.plot([], [], 'b--', lw = 1)

ax.add_patch(cart2)
ax.add_patch(bob2)

trace2_x, trace2_y = [], []

ax.plot([-x_max, x_max], [0, 0])

# Initialises the cart and pendulum
def init_state():
    trace1.set_data([], [])
    pendulum1.set_data([], [])
    cart1.set_center((-cart_width / 2, -cart_height / 2))

    trace2.set_data([], [])
    pendulum2.set_data([], [])
    cart2.set_center((-cart_width / 2, -cart_height / 2))

    return pendulum1, cart1, bob1, trace1, pendulum2, cart2, bob2, trace2

# Called each frame to update the animation to the next state.
def update_state(i):
    if i == 0:
        trace1_x.clear()
        trace1_y.clear()

        trace2_x.clear()
        trace2_y.clear()

    x1, v1, theta1, omega1 = sol.y[:, i]
    x2, v2, theta2, omega2 = linear_sol.y[:, i]

    print(theta1, theta2)

    # x-axis is flipped in the mathematical model
    x1 = -x1
    x2 = -x2

    pendulum1_x = x1 + l * np.sin(theta1)
    pendulum1_y = l * np.cos(theta1)

    trace1_x.append(pendulum1_x)
    trace1_y.append(pendulum1_y)
    trace1.set_data(trace1_x, trace1_y)

    pendulum1.set_data([x1, pendulum1_x], [0, pendulum1_y])
    cart1.set_xy((x1 - cart_width / 2, -cart_height / 2))
    bob1.set_center([pendulum1_x, pendulum1_y])

    #################################################################

    pendulum2_x = x2 + l * np.sin(theta2)
    pendulum2_y = l * np.cos(theta2)

    trace2_x.append(pendulum2_x)
    trace2_y.append(pendulum2_y)
    trace2.set_data(trace2_x, trace2_y)

    pendulum2.set_data([x2, pendulum2_x], [0, pendulum2_y])
    cart2.set_xy((x2 - cart_width / 2, -cart_height / 2))
    bob2.set_center([pendulum2_x, pendulum2_y])

    return pendulum1, cart1, bob1, trace1, pendulum2, cart2, bob2, trace2

legend_elements = [cart1, cart2]
plt.legend(
    handles = legend_elements,
    loc = 'upper left'
)

anim = animation.FuncAnimation(
    fig, update_state, frames = len(t_eval), #init_func = init_state,
    interval = (t_span[1] - t_span[0]), blit = True
)

plt.show()
