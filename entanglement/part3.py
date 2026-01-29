import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from part1 import zero_state
from part2 import ghz_state, w_state, vn_entropy_subset
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

def entropy_map_2body(rho,N):
    M = np.zeros((N,N))
    for i in range (N):
        for j in range (N):
            if i != j:
                M[i, j] = round(vn_entropy_subset(rho, [i, j]),3)
            else:
                M[i, j] = round(vn_entropy_subset(rho, [i]),3)
    return M

def plot_entropy_heatmap(M, N, state):
    plt.figure(figsize=(5,5))
    im = plt.imshow(M, interpolation="nearest", vmin=0, vmax=1)
    plt.colorbar(im, label="Entropy (bits)")
    plt.xticks(range(M.shape[0]))
    plt.yticks(range(M.shape[0]))
    plt.title(f"N: {N} and state {state}:")
    plt.tight_layout()
    plt.colorbar()
    plt.savefig(f"graphs/activity5/{state}{N}.png")
    plt.close()


def avg_entropy_for_k(rho, N, k):
    entropies = []
    for keep in combinations(range(N), k):
        entropies.append(vn_entropy_subset(rho, list(keep)))
    return np.mean(entropies)

def main():

    '''Activity 5'''
    for N in range(3,9):
        simple = zero_state(N)
        ghz = ghz_state(N)
        w = w_state(N)
        namesDict = {'simple': simple, 'ghz': ghz, 'w': w}
        for name, state in namesDict.items():
            M = entropy_map_2body(ket2dm(state), N)
            plot_entropy_heatmap(M, N, name)

    '''Activity 6'''
    N=8
    simple = zero_state(N)
    ghz = ghz_state(N)
    w = w_state(N)
    namesDict = {'simple': simple, 'ghz': ghz, 'w': w}
    y = {'simple': [], 'ghz': [], 'w': []}

    for name, state in namesDict.items():
        for k in range(1,8):
            rho = ket2dm(state)
            y[name].append(avg_entropy_for_k(rho, N, k))
        
        plt.plot(range(1,8), y[name])
        plt.xlabel("k")
        plt.ylabel("Average Entropy")
        plt.title(f"Average Entropy vs k for {name}{N}")
        plt.ylim(-0.1, 1.1)
        plt.savefig(f"graphs/activity6/{name}{N}.png")
        plt.close()
    






if __name__ == "__main__":
    main()