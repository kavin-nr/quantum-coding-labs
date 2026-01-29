import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations

from qutip import (
    basis, tensor, qeye, ket2dm, ptrace,
    entropy_vn
)

# QuTip puts purity as an attribute of the density matrix.
# You can use it as rho.purity()
# or you can define a specific function to extract the purity.

def purity(rho):    
    return rho.purity()

np.set_printoptions(precision=4, suppress=True)


zero = basis(2,0)
one = basis(2,1)

xup = (zero + one).unit()

def zero_state(N):
  return tensor([zero]*N)



def main():
    # for N in range(1,5):
    #     print(f"N: {N} and Zero State: {zero_state(N).shape}")

    print("N=4")
    print("tensor(zero, one, xup, one)")
    rho = ket2dm(tensor(zero, one, xup, one))
    print(rho.data_as("csr_matrix"))
    print(f"Global Purity: {rho.purity():.4f}")
    print(f"Global Entropy: {entropy_vn(rho, 2):.4f}")

    rho0 = ptrace(rho, [0])
    print()
    print("Subsystem 0")
    print(rho.data_as("csr_matrix"))
    print(f"Purity of Subsystem 0: {rho0.purity():.4f}")
    print(f"Entropy of Subsystem 0: {entropy_vn(rho0, 2):.4f}")

    print()

    print("N=5")
    print("tensor(zero, one, xup, one, one)")
    rho = ket2dm(tensor(zero, one, xup, one, one))
    print(rho.data_as("csr_matrix"))

    print(f"Global Purity: {rho.purity():.4f}")
    print(f"Global Entropy: {entropy_vn(rho, 2):.4f}")

    rho0 = ptrace(rho, [0])
    print()
    print("Subsystem 0")
    print(rho.data_as("csr_matrix"))
    print(f"Purity of Subsystem 0: {rho0.purity():.4f}")
    print(f"Entropy of Subsystem 0: {entropy_vn(rho0, 2):.4f}")

if __name__ == "__main__":
   main()