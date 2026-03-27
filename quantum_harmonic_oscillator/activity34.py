import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simpson
from matplotlib.animation import FuncAnimation, FFMpegWriter
from part1 import get_eigenstuff
from activity2 import psi


# Finding eigenvectors
num_eigenvectors = 30

L = 20/2 # nm
N = 1500 # grid points
h = 0.6852 # eV * fs
homega = 0.050 # eV
omega = homega/h # fs^-1
imag = 1j # imaginary

hsquaredover2m = 0.0381 # eV nm^2
momegasquared = 0.0328 # ev / nm^2

energies, eigenvectors, xrange = get_eigenstuff(N, hsquaredover2m ,momegasquared, L, [0,num_eigenvectors])
dx = xrange[1] - xrange[0]


c_vals = np.zeros(num_eigenvectors, dtype=complex)

# discrete normalization
eigenvectors = eigenvectors / np.sqrt(dx) 

for i in range(num_eigenvectors):

    # overlap
    overlap = np.array([np.conjugate(eigenvectors[i][index])*psi(x,0) for index, x in enumerate(xrange[1:-1])])
    c_vals[i] = simpson(overlap, x=xrange[1:-1])


psi_vals = np.array([psi(x,0) for x in xrange[1:-1]])


x = xrange[1:-1]

# ----------------------------------------
# Choose time range
# ----------------------------------------
T = 2 * np.pi / omega
t_max = 4 * T
num_frames = 300
t_vals = np.linspace(0, t_max, num_frames)

num_states_used = len(c_vals)

# ----------------------------------------
# Store wavefunction and probability density
# ----------------------------------------
psi_frames = []
probability_frames = []
x_expectation_vals = []

for t in t_vals:
    psi_t = np.zeros_like(x, dtype=complex)

    for n in range(num_states_used):
       #************ ADD YOUR CODE HERE ***************
       # This section should find psi(x,t) by adding up the
       # contributions from each c_n * phi_n(x) and store it
       # as a np.array named psi_t
       #***********************************************

       psi_t += c_vals[n] * eigenvectors[n] * np.exp(-imag*energies[n]*t/h)



    psi_frames.append(psi_t)

    prob_density = np.abs(psi_t)**2
    probability_frames.append(prob_density)

    #******** PUT YOUR CODE HERE ************
    # calculate x_exp
    # append x_exp into x_expectation_vals
    #****************************************
    x_exp = simpson(x*prob_density, x=x)
    x_expectation_vals.append(x_exp)

psi_frames = np.array(psi_frames)
probability_frames = np.array(probability_frames)

# ----------------------------------------
# Animation of probability density
# ----------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))
line, = ax.plot(x, probability_frames[0], lw=2)

ax.set_xlabel("x (nm)")
ax.set_ylabel(r"$|\psi(x,t)|^2$")
ax.set_title("Time Evolution of Probability Density")

ax.set_xlim(x.min(), x.max())
ax.set_ylim(0, 1.1 * np.max(probability_frames))

time_text = ax.text(
    0.02, 0.95, "", transform=ax.transAxes,
    verticalalignment="top"
)

def update(frame):
    line.set_ydata(probability_frames[frame])
    time_text.set_text(f"t = {t_vals[frame]:.2f} fs")
    return line, time_text

anim = FuncAnimation(
    fig,
    update,
    frames=num_frames,
    interval=40,
    blit=True
)

from matplotlib.animation import PillowWriter
anim.save("./part2plots/sho_time_evolution.gif", writer=PillowWriter(fps=20))

plt.show()




# Activity 4

from scipy.optimize import curve_fit

# ----------------------------------------
# Define fitting function
# ----------------------------------------
def sinusoid(t, A, omega, phi, C):
    return A * np.cos(omega * t + phi) + C

# ----------------------------------------
# Initial guesses
# ----------------------------------------
A_guess = (np.max(x_expectation_vals) - np.min(x_expectation_vals)) / 2
C_guess = np.mean(x_expectation_vals)
omega_guess = omega   # use known oscillator frequency as a starting guess
phi_guess = 0

initial_guess = [A_guess, omega_guess, phi_guess, C_guess]

# ----------------------------------------
# Perform fit
# ----------------------------------------
params, covariance = curve_fit(
    sinusoid,
    t_vals,
    x_expectation_vals,
    p0=initial_guess
)

A_fit, omega_fit, phi_fit, C_fit = params

# ----------------------------------------
# Print results
# ----------------------------------------
print(f"Fitted amplitude A = {A_fit:.4f} nm")
print(f"Fitted angular frequency omega = {omega_fit:.4f} fs^-1")
print(f"Fitted phase phi = {phi_fit:.4f} rad")
print(f"Fitted offset C = {C_fit:.4f} nm")

print("\nExpected omega =", omega)
print("Percent error =", abs((omega_fit - omega)/omega) * 100, "%")

# ----------------------------------------
# Plot data and fit
# ----------------------------------------
t_fine = np.linspace(t_vals.min(), t_vals.max(), 1000)
fit_curve = sinusoid(t_fine, A_fit, omega_fit, phi_fit, C_fit)

plt.figure(figsize=(8,5))
plt.plot(t_vals, x_expectation_vals, 'o', label="Data")
plt.plot(t_fine, fit_curve, '-', label="Fit")
plt.xlabel("time (fs)")
plt.ylabel(r"$\langle x \rangle (t)$ (nm)")
plt.title("Sinusoidal Fit to Expectation Value")
plt.legend()
plt.grid(True)
plt.savefig("./part2plots/expectationvalue.png")