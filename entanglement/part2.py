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

# psi_bell = (tensor(zero, zero) + tensor(one, one)).unit()
# rho_bell = ket2dm(psi_bell)

# print("Bell: Global purity =", purity(rho_bell))
# print("Bell: Global entropy =", entropy_vn(rho_bell, base=2))

# rho_A = ptrace(rho_bell, [0])
# print("Bell: purity(A) =", purity(rho_A))
# print("Bell: entropy(A) =", entropy_vn(rho_A, base=2))

def ghz_state(N):
   return (tensor([zero]*N) + tensor([one]*N)).unit()

def w_state(N):
    return sum([tensor([zero]*i + [one] + [zero]*(N-i-1)) for i in range(N)]).unit()

def vn_entropy_subset(rho, keep):
    return entropy_vn(ptrace(rho, keep), base=2)

def main():
    psi_ghz3 = ghz_state(3)
    rho_ghz3 = ket2dm(psi_ghz3)

    print("GHZ3: Global purity =", purity(rho_ghz3))
    print("GHZ3: Global entropy =", entropy_vn(rho_ghz3, base=2))

    for i in range(3):
        print(f"S(qubit {i}) =", vn_entropy_subset(rho_ghz3, [i]))

    
if __name__ == "__main__":
   main()