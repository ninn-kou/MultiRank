import numpy as np
from scipy.sparse import spdiags

def multirank(A, gamma, s, a, alpha=0.85):
    # M: Amount of Layers
    # N: Amount of Nodes
    M = A.shape[0]
    N = A.shape[1]

    # Error to control iteration times
    v_quadratic_error = 0.0001

    # Symmetric adjacency matrix A (shape: M * N * N)
    for layer in range(M):
        A[layer] = A[layer].T

    # Calculate matrix B (shape: M * N) at once
    B = np.zeros((M, N))
    for layer in range(M):
        for node in range (N):
            B[layer][node] = A[layer].sum(axis=0)[node]

    # Initialise the vector z (shape: M) as an unit vector
    z = np.ones(M)

    # Calculate the initial matrix G (shape: N * N) with the all 1 vector z
    G = np.zeros((N, N))
    for layer in range(M):
        G += A[layer] * z[layer]

    # Create the sparse matrix D (shape: N * N)
    # D[i][j] means the inverse of all links to one node in G
    D = np.sum(G, axis=0) + (np.sum(G, axis=0) == 0)
    D = np.ones(N) / D
    D = spdiags(D, 0, N, N)

    # Calculate the vector v (shape: N)
    # Every element in v is
    #     an equally constant if the node has neighbours,
    #     or a 0 if the node has no connection.
    v = (np.sum(G, axis=0) + np.sum(G, axis=1).T) > 0
    v = v.T / np.count_nonzero(v)

    # delta: apply the neutralize constant into all none-zero element in G
    l = np.sum(G, axis=0) > 0;
    delta = alpha * l.T

    # Calculate the Initial vector x (shape: N)
    x = v
    x = G * D @ (x * delta) + np.sum((1 - delta) * x, axis=0) * v

    # Calculate the Initial vector z (shape: M) based on initial x
    z = (np.sum(B, axis=1) ** a) * (B @ (x + (x == 0)) ** (s * gamma)) / ((np.sum(B, axis=1) + (np.sum(B, axis=1) == 0)) ** s)
    z /= sum(z)

    # Composition of iteration controller
    last_x = np.ones(N) * np.inf

    # Iterations here:
    #     Mostly as same as above calculations, but using the data from last iteration
    while np.linalg.norm(x - last_x, 2) > v_quadratic_error * np.linalg.norm(x):
        last_x = x;
        G = np.zeros((N, N))
        for layer in range(M):
            G += A[layer] * z[layer]

        D = np.sum(G, axis=0) + (np.sum(G, axis=0) == 0)
        D = np.ones(N) / D
        D = spdiags(D, 0, N, N)

        # No need to recalculate vector v again because even G is changing in every iteration,
        # the value of v is steady because it related to a boolean value of G.

        l = np.sum(G, axis=0) > 0;
        delta = alpha * l.T

        x = G * D @ (x * delta) + np.sum((1 - delta) * x, axis=0) * v

        z = (np.sum(B, axis=1) ** a) * (B @ (x + (x == 0)) ** (s * gamma)) / ((np.sum(B, axis=1) + (np.sum(B, axis=1) == 0)) ** s)
        z /= sum(z)

    # Returning the final vector x and z, based on the specific value of gamma, a, s and alpha.
    return x, z
