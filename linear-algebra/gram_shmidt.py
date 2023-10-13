import numpy as np
import numpy.linalg as la

verySmallNumber = 1e-14

def gsBasis4(A):
    B = np.array(A, dtype=np.float_)
    B[:, 0] = B[:, 0] / la.norm(B[:, 0])
    B[:, 1] = B[:, 1] - B[:, 1] @ B[:, 0] * B[:, 0]

    if la.norm(B[:, 1]) > verySmallNumber:
        B[:, 1] = B[:, 1] / la.norm(B[:, 1])
    else:
        B[:, 1] = np.zeros_like(B[:, 1])

    B[:, 2] = B[:, 2] - B[:, 2] @ B[:, 0] * B[:, 0] - B[:, 2] @ B[:, 1] * B[:, 1]

    if la.norm(B[:, 2]) > verySmallNumber:
        B[:, 2] = B[:, 2] / la.norm(B[:, 2])
    else:
        B[:, 2] = np.zeros_like(B[:, 2])

    B[:, 3] = B[:, 3] - B[:, 3] @ B[:, 0] * B[:, 0] - B[:, 3] @ B[:, 1] * B[:, 1] - B[:, 3] @ B[:, 2] * B[:, 2]

    if la.norm(B[:, 3]) > verySmallNumber:
        B[:, 3] = B[:, 3] / la.norm(B[:, 3])
    else:
        B[:, 3] = np.zeros_like(B[:, 3])

    return B

def gsBasis(A):

    B = np.array(A, dtype=np.float_)

    for i in range(B.shape[1]):
        for j in range(i):
            B[:, i] = B[:, i] - B[:, i] @ B[:, j] * B[:, j]

        if la.norm(B[:, i]) > verySmallNumber:
            B[:, i] = B[:, i] / la.norm(B[:, i])
        else:
            B[:, i] = np.zeros_like(B[:, i])
            
    return B

def dimensions(A) :
    return np.sum(la.norm(gsBasis(A), axis=0))
