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
sigma_us = 10            # 10 microseconds
sigma    = sigma_us * 1e-6  # convert to seconds

# ============================================================
# 2. Time grid: positive-only times 0 → 10σ, pulse centered at 5σ
# ============================================================

t0 = 5 * sigma          # center the Gaussian at 5σ (in seconds)
t_start = 0.0
t_end   = 10 * sigma    # window [0, 10σ] in seconds

tlist    = np.linspace(t_start, t_end, 20)   # seconds
tlist_us = tlist * 1e6                        # microseconds for plotting

args = {"Omega0": Omega0, "sigma": sigma, "t0": t0}

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

# Force QuTiP to store the states as well as expectations
opts = Options(store_states=True)

result = sesolve(H, psi0, tlist, e_ops=e_ops, args=args, options=opts)
sx_t, sy_t, sz_t = result.expect

# Optional: plot expectation values vs time in microseconds
plt.figure()
plt.plot(tlist_us, sx_t, label=r'$\langle \sigma_x \rangle$')
plt.plot(tlist_us, sy_t, label=r'$\langle \sigma_y \rangle$')
plt.plot(tlist_us, sz_t, label=r'$\langle \sigma_z \rangle$')
plt.xlabel('time (µs)')
plt.ylabel('expectation value')
plt.legend()
plt.grid(True)
plt.tight_layout()

# ============================================================
# 5. Bloch sphere trajectory
# ============================================================

b = Bloch()
b.add_states(result.states)
b.show()

plt.show()