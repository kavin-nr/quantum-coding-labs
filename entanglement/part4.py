import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from part1 import zero_state
from part2 import ghz_state, w_state, vn_entropy_subset
from part3 import entropy_map_2body, plot_entropy_heatmap
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




def remove_one_qubit(rho,N):
   keep = [i for i in range(N) if i != 0] # Only removes q0
   return ptrace(rho, keep)

def map_2body(rho,N):
    M = np.zeros((N,N))
    for i in range (N):
        for j in range (N):
            M[i, j] = rho[i,j]
    return M

def plot_heatmap(M, N, state):
    plt.figure(figsize=(5,5))
    im = plt.imshow(M, interpolation="nearest", vmin=0, vmax=1)
    plt.colorbar(im, label="Value")
    plt.xticks(range(M.shape[0]))
    plt.yticks(range(M.shape[0]))
    plt.title(f"N: {N} and state {state}:")
    plt.tight_layout()
    plt.colorbar()
    plt.savefig(f"graphs/activity7/{state}{N}.png")
    plt.close()

def main():
    for N in range(5,9):
        ghz = ghz_state(N)
        w = w_state(N)

        nameDict = {'ghz': ghz, 'w': w}

        for name, state in nameDict.items():
            rho = ket2dm(state)
            M = map_2body(remove_one_qubit(rho, N), remove_one_qubit(rho,N).shape[0])
            # M = map_2body(rho, N)
            plot_heatmap(M, N, name)

            



if __name__ == "__main__":
    main()