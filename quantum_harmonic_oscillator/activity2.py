import numpy as np
from scipy.linalg import eigh_tridiagonal
from scipy.optimize import curve_fit
from scipy.integrate import simpson
import matplotlib.pyplot as plt
import time
from part1 import get_eigenstuff

def psi(x,t=0):
    sigma = 0.45 # nm
    x0 = 2.0 # nm
    k0 = 1.8 # nm^-1
    i = 1j
    return (1/(2*np.pi*sigma**2)**0.25)*np.exp(-(x-x0)**2/(4*sigma**2))*np.exp(i*k0*x)


def reconstructPsi(num_eigenvectors = 30):
    L = 20/2 # nm
    N = 1500 # grid points
    h = 0.6852 # eV * fs
    homega = 0.050 # eV
    hsquaredover2m = 0.0381 # eV nm^2
    momegasquared = 0.0328 # ev / nm^2

    eigenvals, eigenvectors, xrange = get_eigenstuff(N, hsquaredover2m ,momegasquared, L, [0,num_eigenvectors])
    dx = xrange[1] - xrange[0]

    
    coeffs = np.zeros(num_eigenvectors, dtype=complex)

    # discrete normalization
    eigenvectors = eigenvectors / np.sqrt(dx) 

    for i in range(num_eigenvectors):

        # overlap
        overlap = np.array([np.conjugate(eigenvectors[i][index])*psi(x,0) for index, x in enumerate(xrange[1:-1])])
        coeffs[i] = simpson(overlap, x=xrange[1:-1])
    
    # norm_constraint = np.sum(np.abs(coeffs)**2)
    # print(norm_constraint)
    print(coeffs)
    psi_vals = np.array([psi(x,0) for x in xrange[1:-1]])
    psi_reconstructed = coeffs @ eigenvectors[:num_eigenvectors] 

    return psi_vals, psi_reconstructed, eigenvals, xrange

def main():
    
    begin = time.time()

    num_eigenvectors = 30

    psi_vals, psi_reconstructed, eigenvals, xrange = reconstructPsi(num_eigenvectors)
    
    plt.plot(xrange[1:-1], np.power(np.abs(psi_vals),2), label="Real Psi")

    overlap = simpson(np.array(np.conjugate(psi_reconstructed)*psi_vals), x=xrange[1:-1])

    plt.plot(xrange[1:-1], np.power(np.abs(psi_reconstructed),2), label=f"Reconstructed with overlap {np.abs(overlap):.4f}")
    plt.title(f"Wavefunction probability reconstruction with {num_eigenvectors} eigenvectors")
    plt.xlabel("x (nm)")
    plt.ylabel("Probability")
    plt.ylim(0,1)
    plt.legend()
    plt.savefig("./part2plots/reconstruction.png")
    print(f"Part 2 done: {(time.time()-begin):.4f}")

    
        



if __name__ == "__main__":
    main()