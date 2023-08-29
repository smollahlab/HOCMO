import numpy as np
from numpy.linalg import norm

def compute_namda(A, B, C):
    '''
    Calculates namda? Found by multiplying the norm of the three dimensions of the tensor?
    INPUT:
    A:
    B:
    C:

    OUTPUT:
    namda:

    '''
    K = A.shape[1]
    namda = 1
    for r in range(K):
        A_r = A[:,r]
        B_r = B[:,r]
        C_r = C[:,r]
        namda *= norm(A_r)*norm(B_r)*norm(C_r)
    return namda


# Method1: rank the core consistency rigidly following Panisson's paper
def rank_(cc_values, namdas, k):
    '''
    Calculates namda? Found by multiplying the norm of the three dimensions of the tensor?
    INPUT:
    cc_values:
    namdas:
    k:

    OUTPUT:
    top core consistency values:
    
    '''
    top_50percent = int(len(cc_values)/2)
    top_idx = np.argsort(cc_values)[-top_50percent:]
    cc_values_selected = cc_values[top_idx]
    namdas_selected = namdas[top_idx]
    top_idx = np.argsort(namdas_selected)[-k:]
    return cc_values_selected[top_idx]


# Method 2: rank the core consistency with top-k returned
def rank_k(cc_values, k):
    '''
    Calculates namda? Found by multiplying the norm of the three dimensions of the tensor? Differs from Panissons's how?
    INPUT:
    cc_values:
    namdas:
    k:

    OUTPUT:
    top core consistency values:
    
    '''
    #top_50percent = int(len(cc_values)/2) This line isn't used?
    top_idx = np.argsort(cc_values)[-k:]
    return cc_values[top_idx]