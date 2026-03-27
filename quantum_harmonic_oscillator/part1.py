import numpy as np
from scipy.linalg import eigh_tridiagonal
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import time


def get_eigenstuff(N, hsquaredover2m, momegasquared, L, select_range):
    xrange = np.linspace(-L, L, N)

    dx = xrange[1] - xrange[0] 

    t = hsquaredover2m/(dx**2)


    d = 2*t*np.ones(N)
    e = -t*np.ones(N-1)

    for i in range(len(xrange)):
        d[i] += 0.5 * momegasquared * xrange[i]**2


    # SHOULD THERE BE A BOUND ON EIGENVALUES?
    eigenvals, eigenvectors = eigh_tridiagonal(d[1:-1], e[1:-1], select='i', select_range=select_range) 

    # print(eigenvals)
    # print(eigenvectors)
    # A = np.diag(d) + np.diag(e, k=1) + np.diag(e, k=-1)

    eigenvectors = [eigenvectors[:,i] for i in range(len(eigenvectors[0]))]
    return eigenvals, eigenvectors, xrange

def main():
    
    begin = time.time()

    L = 30/2 # nm
    N = 1500 # grid points
    h = 0.6852 # eV * fs
    homega = 0.050 # eV
    hsquaredover2m = 0.0381 # eV nm^2
    momegasquared = 0.0328 # ev / nm^2

    eigenvals, eigenvectors, xrange = get_eigenstuff(N, hsquaredover2m ,momegasquared, L, [0,60])
    dx = xrange[1] - xrange[0]

    for i, state in enumerate(eigenvectors):
        temp = np.zeros(N)
        temp[1:-1] = state
        state = temp 

        prob = state**2
        norm = np.sum(prob) * dx

        state = state / norm
        prob = prob / norm

        plt.plot(xrange, prob)
        plt.title(f"Quantum number {i}")
        plt.xlabel("x (nm)")
        plt.ylabel("Probability")
        plt.ylim(0,1)
        if i == 60:
            def springP(x, k):
                A = np.sqrt(2*eigenvals[i]/k)
                return 1/(np.pi*np.sqrt(A**2-x**2))
            k = momegasquared
            plt.plot(xrange, springP(xrange,k))
            plt.ylim(0,0.2)
        plt.savefig(f"./part1plots/q{i}.jpg")
        plt.close()

        print(f"Quantum number {i}")
        print(f"Associated eigenenergy: {eigenvals[i]:.4f}")
        realeigen = homega * (i + 1/2)
        print(f"Real eigenenergy: {realeigen:.4f}")
        print(f"Percent error: {(abs(realeigen - eigenvals[i])/realeigen)*100:.4f}%")
        print()

    print(f"Part 1 done: {(time.time()-begin):.4f}")

    
        



if __name__ == "__main__":
    main()