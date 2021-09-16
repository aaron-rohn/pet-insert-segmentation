import numpy as np
import matplotlib.pyplot as plt

from scipy import ndimage

def load_flood(f):
    return np.fromfile(f, 'int32').reshape((512,512))

def log_filter(fld, ksize = 2):
    fld = ndimage.gaussian_laplace(fld.astype('double'), ksize)
    fld = fld / np.min(fld)
    fld[fld < 0] = 0
    return fld

def sample_flood(flood, n):
    flat = flood.flatten()
    sample_idx = np.random.choice(flat.size, size = int(n), p = flat)
    idx = np.unravel_index(sample_idx, flood.shape)
    return np.array(idx)

def edges(fld, axis = 0):
    p = np.sum(fld, axis)
    thresh = np.max(p) / 10
    ledge = np.argmax(p > thresh)
    redge = len(p) - np.argmax(p[::-1] > thresh)
    return np.array([ledge, redge])

def flood_init_mask(fld, ncrystals = 19):
    ta_edges = edges(fld, 0)
    ax_edges = edges(fld, 1)
    ncry2 = ncrystals**2 

    ta = np.linspace(*ta_edges, ncrystals)
    ax = np.linspace(*ax_edges, ncrystals)
    mu0 = np.array(np.meshgrid(ta, ax)).reshape(2,ncry2)

    w0 = np.array(1.0/ncry2).repeat(ncry2)

    ta_diff = np.diff(ta_edges)[0] / ncrystals
    ax_diff = np.diff(ax_edges)[0] / ncrystals
    sigma0 = np.array([[ta_diff,0],[0,ax_diff]])[None,...].repeat(ncry2, axis = 0)

    return (w0, mu0.T, sigma0)

def gaussian(samples, mu, sigma, sigma_default = 100):
    try:
        sigma_i = np.linalg.inv(sigma)
        norm = 1.0 / np.sqrt(4 * np.linalg.det(sigma) * np.pi**2)
        if np.any(np.abs(sigma_i) > 1e10): raise np.linalg.LinAlgError
    except:
        sigma_i = np.eye(2) / sigma_default
        norm = 1.0 / (2 * np.pi * sigma_default**2)

    x = samples.T - mu
    x = np.sum((x @ sigma_i) * x, axis = 1)
    return norm * np.exp(-0.5 * x)

def fit(f, nsamples = 100_000, niter = 10):
    fld = load_flood(f)
    fld = log_filter(fld)
    fld /= np.sum(fld)

    samples = sample_flood(fld, nsamples)
    fld0,*_ = np.histogram2d(*samples, 512, [[0,511],[0,511]])
    w, mu, sigma = flood_init_mask(fld0)

    k = len(w)
    g = np.zeros((k, nsamples), 'float64')

    for i in range(20):
        print(i)

        for j in range(k):
            g[j] = gaussian(samples, mu[j], sigma[j])

        g *= w[:,None]
        g /= g.sum(0)
        q = g.sum(1)
        q[q == 0] = 1e-6

        w = q / nsamples

        mu = np.sum(samples[:,None,:] * g, 2) / q
        mu = mu.T

        sigma = samples.T[:,None,:] - mu
        sigma = sigma.T * g
        sigma = np.moveaxis(sigma, 0, 1)
        sigma = sigma @ np.moveaxis(sigma, 1, 2)
        sigma /= q[:,None,None]

    plt.imshow(fld0)
    plt.scatter(*mu.T, s = 4, color = 'red')
    plt.show()

    return w, mu, sigma
