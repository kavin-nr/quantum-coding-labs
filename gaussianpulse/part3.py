import numpy as np
import matplotlib.pyplot as plt
import random

from qutip import (
    expect, basis, sigmax, sigmay, sigmaz,
    sesolve, Bloch, Options)

# ============================================================
# 1. Parameters
#    Work numerically with ħ = 1; H has units of rad/s.
# ============================================================

Omega0 = 2 * np.pi * 10e3   # 2π × 10 kHz ~ 6.283e4 rad/s

# Sigma in microseconds:
sigma_us = 10      # 10 microseconds
sigma    = sigma_us * 1e-6  # convert to seconds

#Number of pulses
NumPulse = 50

#initialize arrays  *** BE CAREFULE WITH DATATYPES IN QuTip
pulseNum = []           # list to store the index of pulses
times_all = None        # 1D numpy array
states_all = []         # list of Qobj (specal QuTip datatype)
expect_all = None       # list of many numpy arrays

seed = 31                                 #for reproducability
random.seed(seed)
np.random.seed(seed)

# Statistical Jitter
sigma_jitter_us = sigma_us * 0.2#.005         # .5% error units microsecond
sigma_jitter    = sigma_jitter_us * 1e-6  # convert to seconds

Omega0_jitter = Omega0 * .0#001             # 0.1% error units rad/s

# ============================================================
# 2. Time grid: positive-only times 0 → 10σ, pulse centered at 5σ
# ============================================================

t0 = 5 * sigma          # center the Gaussian at 5σ (in seconds)
t_start = 0.0
t_end   = 10 * sigma    # window [0, 10σ] in seconds
    
tlist    = np.linspace(t_start, t_end, 20)   # seconds
tlist_us = tlist * 1e6                        # microseconds for plotting

args_nominal = {"Omega0": Omega0, "sigma": sigma, "t0": t0}


# ============================================================
# 3. Time-dependent Hamiltonian H(t) = Omega_z(t) * Sz
# ============================================================

def Omega(t, args):
    """Z-rotation rate Omega_z(t) in rad/s (Gaussian pulse)."""
    Omega0 = args["Omega0"]
    sigma  = args["sigma"]
    t0     = args["t0"]
    return Omega0 * np.exp(-(t - t0)**2 / (2 * sigma**2))

# Spin operators
Sx = 0.5 * sigmax()
Sy = 0.5 * sigmay()
Sz = 0.5 * sigmaz()

# list possible directions applied field
spinoperators=[Sx,Sy,Sz]

# In QuTiP units, with ħ = 1, a Hamiltonian in rad/s is fine:
# H(t) = Omega(t) * Sz
H = [[Sz, Omega]]
print(H)

# ============================================================
# 4. Initial state and time evolution
#    Start in |+x> = (|0> + |1>)/sqrt(2).
# ============================================================

psi0 = (basis(2, 0) + basis(2, 1)).unit()

# # Start at +z
# psi0 = (basis(2,0))

# # Start at +y
# psi0 = (basis(2,0) + basis(2,1)*1j).unit()

# Track expectations of sigma_x, sigma_y, sigma_z
e_ops = [2*Sx, 2*Sy, 2*Sz]  # gives <σx>, <σy>, <σz>

# Concatinates time, state, and expectation value lists after each pulse
def concat_data(result, times_all, states_all, expect_all):
    times_k = np.array(result.times)      # convert result.times to numpy array for t_offset

    # check for first pulse empty data
    if times_all is None:
        # first pulse: take everything as-is
        times_all  = times_k
        states_all = list(result.states)
        expect_all = [np.array(e) for e in result.expect]

    # for all pulses after first pulse
    else:
        # shift times of this pulse so it follows previous pulse
        t_offset  = times_all[-1]
        new_times = times_k[1:] + t_offset  # skip first to avoid duplicate

        # concatenate times
        times_all = np.concatenate([times_all, new_times])

        # concatenate states (skip first to avoid duplicate)
        states_all.extend(result.states[1:])

        # concatenate expectation values (each expect[j] is already an array)
        for j in range(len(expect_all)):
            expect_all[j] = np.concatenate(
                [expect_all[j], np.array(result.expect[j][1:])]
            )
    return times_all, states_all, expect_all

def draw_jittered_args(args):
    return {"Omega0": args["Omega0"] + np.random.normal(scale=Omega0_jitter), "sigma": args["sigma"] + np.random.normal(scale=sigma_jitter), "t0": args["t0"]}

# Force QuTiP to store the states as well as expectations
opts = Options(store_states=True)
for k in range(NumPulse):
    pulseNum.append(k+1)                      #get index of loop

    def get_random_basis():
        return spinoperators[random.randint(0,2)]
    H_axis = get_random_basis()
    H = [[H_axis, Omega]]

    result = sesolve(H, psi0, tlist, e_ops=e_ops, args=draw_jittered_args(args_nominal), options=opts)


    psi0 = result.states[-1]

    times_all, states_all, expect_all = concat_data(result, times_all, states_all, expect_all)

sx_t, sy_t, sz_t = expect_all

# Optional: plot expectation values vs time in microseconds
plt.figure()
plt.plot(times_all, sx_t, label=r'$\langle \sigma_x \rangle$')
plt.plot(times_all, sy_t, label=r'$\langle \sigma_y \rangle$')
plt.plot(times_all, sz_t, label=r'$\langle \sigma_z \rangle$')
plt.xlabel('time (µs)')
plt.ylabel('expectation value')
plt.legend()
plt.grid(True)
plt.tight_layout()

# ============================================================
# 5. Bloch sphere trajectory
# ============================================================

b = Bloch()
b.add_states(states_all)
b.show()

plt.show()