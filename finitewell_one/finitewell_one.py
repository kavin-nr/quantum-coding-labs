import numpy as np
from scipy.linalg import eigh_tridiagonal
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import time


def get_eigenstuff(N, m, h, a, V0, L):
    xrange = np.linspace(-L, L, N)

    dx = xrange[1] - xrange[0] 

    t = h**2/(2*m*dx**2)


    d = 2*t*np.ones(N)
    e = -t*np.ones(N-1)

    for i in range(len(xrange)):
        if abs(xrange[i]) > a:
            d[i] += V0

    eigenvals, eigenvectors = eigh_tridiagonal(d[1:-1], e[1:-1], select='v', select_range=(0,V0)) # the limit really killed computational time! 
                                                                                                # N doesn't really matter once you get here

    # print(eigenvals)
    # print(eigenvectors)
    # A = np.diag(d) + np.diag(e, k=1) + np.diag(e, k=-1)

    eigenvectors = [eigenvectors[:,i] for i in range(len(eigenvectors[0]))]
    return eigenvals, eigenvectors, xrange

def main():
    
    begin = time.time()

    N = 60000


    m = 511000 # eV
    h = 1240/(2*np.pi) # eV nm
    a = 1.5 / 2 # nm
    V0 = 5 # eV

    L = 20/2 # nm


    eigenvals, eigenvectors, xrange = get_eigenstuff(N, m ,h, a, V0, L)
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
        plt.title(f"Quantum number {i+1}")
        plt.xlabel("x (nm)")
        plt.ylabel("Probability")
        plt.savefig(f"./part1plots/q{i+1}.jpg")
        plt.close()

        integral = 0
        for j in range(len(xrange)):
            if abs(xrange[j]) < a:
                integral += prob[j]
        
        print(f"Quantum number {i+1}")
        print(f"Probability of being within well: {integral*dx}")
        print(f"Associated eigenenergy: {eigenvals[i]}")
        print()

    print(f"Part 1 done: {(time.time()-begin):.4f}")

    # Part 2 
    begin = time.time()
    plt.plot(range(1,len(eigenvectors) + 1), eigenvals, label="Calculated values")
    plt.title(f"Eigenenergy vs Quantum Number for V0 = {V0} eV")
    plt.xlabel("Quantum number (n)")
    plt.ylabel("Eigenenergy (eV)")

    def f(x, A, B):
        return A*x**B

    A,B = curve_fit(f, range(1,len(eigenvectors) + 1), eigenvals)[0]

    x = np.linspace(0, len(eigenvectors), 100)
    plt.plot(x, f(x,A,B), label=f'Curve fit: {A:.3f}x^{B:.3f}')
    plt.legend()
    plt.savefig(f"./part2plots/{V0}ev.jpg")
    plt.close()
    print(f"Part 2.1 done: {(time.time()-begin):.4f}")

    


    # Part 3
    begin = time.time()
    twoarange = np.linspace(1, 10, 20)
    grounds = []
    firsts = []
    for twoa in twoarange:
        eigenvals, eigenvectors, xrange = get_eigenstuff(N, m ,h, twoa/2, V0, L)
        grounds.append(eigenvals[0])
        firsts.append(eigenvals[1])

    plt.plot(twoarange, grounds, label="Calculated values")
    plt.title(f"Ground state energy vs well size for V0 = {V0} eV")
    plt.xlabel("Well Size (nm)")
    plt.ylabel("Eigenenergy (eV)")

    A,B = curve_fit(f, twoarange, grounds)[0]

    x = np.linspace(0, 10, 100)

    plt.plot(x, f(x,A,B), label=f'Curve fit: {A:.3f}x^{B:.3f}')
    plt.legend()
    plt.savefig(f"./part2plots/groundstates.jpg")
    plt.close()

    plt.plot(twoarange, firsts, label="Calculated values")
    plt.title(f"First excited state energy vs well size for V0 = {V0} eV")
    plt.xlabel("Well Size (nm)")
    plt.ylabel("Eigenenergy (eV)")

    A,B = curve_fit(f, twoarange, firsts)[0]

    plt.plot(x, f(x,A,B), label=f'Curve fit: {A:.3f}x^{B:.3f}')
    plt.legend()
    plt.savefig(f"./part2plots/firstexcitedstates.jpg")
    plt.close()
    print(f"Part 2.2 done: {(time.time()-begin):.4f}")

    begin = time.time()
    
    m_e = 511000        # eV/c^2
    m_e_star = 0.13 * m_e   
    m_h_star = 0.45 * m_e   
    V0_e = 4.0          # eV
    V0_h = 4.0          # eV
    E_g = 1.74          # eV 
    hc = 1239.84        # eV nm
    h = 1240 / (2 * np.pi)  # eV nm 
    
    N = 60000 # same as before            
    L = 10 # nm
    
    dot_sizes = np.arange(1.0, 10, 0.5)  
    wavelengths = []
    
    for dot_size in dot_sizes:
        a = dot_size / 2  
        
        e_vals, eigenvectors, xrange = get_eigenstuff(N, m_e_star, h, a, V0_e, L)
        E_e1 = e_vals[0]
        h_vals, eigenvectors, xrange = get_eigenstuff(N, m_h_star, h, a, V0_h, L)
        E_h1 = h_vals[0]
        
        E_photon = E_g + E_e1 + E_h1
        wavelength = hc / E_photon
        wavelengths.append(wavelength)
    
    plt.scatter(dot_sizes, wavelengths)
    plt.title("Photon Wavelength vs. Quantum Dot Size")
    plt.xlabel("Quantum Dot Width (nm)")
    plt.ylabel("Wavelength (nm)")
    plt.ylim(400, 700)

    plt.tight_layout()
    plt.savefig("./part2plots/quantum_dot_wavelength.jpg")
    plt.close()
    
    print(f"Activity 3 done: {(time.time()-begin):.4f}")
    
        



if __name__ == "__main__":
    main()