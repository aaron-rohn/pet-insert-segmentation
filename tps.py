import numpy as np

def U(pi, pj):
    r2 = np.sum((pi - pj)**2, 1)
    return r2 * np.log(r2 + np.finfo(float).eps)

def K(p):
    n = len(p)
    idx = np.arange(n)
    idx_i = np.tile(idx, n)
    idx_j = idx.repeat(n)
    return U(p[idx_i], p[idx_j]).reshape(n,n)

def Z(p):
    b = np.append(np.ones((p.shape[0],1)), p, 1)
    pad = np.zeros((b.shape[1], b.shape[1]))
    return np.block([[K(p), b], [b.T, pad]])

def Y(v): return np.append(v, np.zeros((3, 2)), 0)

def f(p, v):
    *w, a1, ax, ay = np.linalg.inv(Z(p)) @ Y(v)
    return lambda x,y: a1+(x*ax)+(y*ay) + np.sum(w * U(p,[x,y])[:,None], 0)
