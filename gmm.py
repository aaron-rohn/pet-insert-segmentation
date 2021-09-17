import tps
import numpy as np
import matplotlib.pyplot as plt

from scipy import ndimage

class Flood():
    def __init__(self, f):
        if isinstance(f, str):
            self.fld = np.fromfile(f, 'int32').reshape((512,512))
            self.fld = self.fld.astype('double')
            self.fld = self.log_filter()
        else:
            self.fld = f

    def log_filter(self, ksize = 2):
        f = ndimage.gaussian_laplace(self.fld, ksize)
        f = f / np.min(f)
        f[f < 0] = 0
        return f

    def sample(self, n):
        flat = self.fld.flatten() / np.sum(self.fld)
        sample_idx = np.random.choice(flat.size, size = int(n), p = flat)
        idx = np.unravel_index(sample_idx, self.fld.shape)
        return np.array(idx)

    def edges(self, axis = 0):
        p = np.sum(self.fld, axis)
        thresh = np.max(p) / 10
        ledge = np.argmax(p > thresh)
        redge = len(p) - np.argmax(p[::-1] > thresh)
        return np.array([ledge, redge])

    def init_mask(self, ncrystals = 19):
        ta_edges = edges(self.fld, 1)
        ax_edges = edges(self.fld, 0)
        ncry2 = ncrystals**2 

        ta = np.linspace(*ta_edges, ncrystals)
        ax = np.linspace(*ax_edges, ncrystals)
        mu0 = np.array(np.meshgrid(ta, ax)).reshape(2,ncry2)

        w0 = np.array(1.0/ncry2).repeat(ncry2)

        ta_diff = np.diff(ta_edges)[0] / ncrystals
        ax_diff = np.diff(ax_edges)[0] / ncrystals
        sigma0 = np.array([[ta_diff,0],[0,ax_diff]])[None,...].repeat(ncry2, axis = 0)

        return (w0, mu0.T, sigma0)

class Gmm():
    @staticmethod
    def gauss(c, mu, sigma):
        sigma_i = np.linalg.inv(sigma)
        norm = 1.0 / np.sqrt(4 * np.linalg.det(sigma) * np.pi**2)
        x = c.T - mu
        x = np.sum((x @ sigma_i) * x, axis = 1)
        return norm * np.exp(-0.5 * x)

    def __init__(self, f, nsamples = 100_000):
        self.fld = Flood(f)
        self.samples = self.fld.sample(nsamples)
        sz = self.fld.fld.shape[0]
        self.fld0, *_ = np.histogram2d(*self.samples, sz, [[0,sz-1],[0,sz-1]])

    def gaussian(self, mu, sigma, sigma_default = 100):
        """
        try:
            sigma_i = np.linalg.inv(sigma)
            norm = 1.0 / np.sqrt(4 * np.linalg.det(sigma) * np.pi**2)
            if np.any(np.abs(sigma_i) > 1e10): raise np.linalg.LinAlgError
        except:
            sigma_i = np.eye(2) / sigma_default
            norm = 1.0 / (2 * np.pi * sigma_default**2)
        """

        sigma_i = np.linalg.inv(sigma)
        norm = 1.0 / np.sqrt(4 * np.linalg.det(sigma) * np.pi**2)
        x = self.samples.T - mu
        x = np.sum((x @ sigma_i) * x, axis = 1)
        return norm * np.exp(-0.5 * x)

    def e_step(self, w, mu, sigma):
        g = np.array([self.gaussian(u,s) for u,s in zip(mu,sigma)])
        g *= w[:,None]
        g /= g.sum(0)
        q = g.sum(1)
        q[q == 0] = 1e-6
        return g, q

    def m_step(self, g, q):
        w = q / g.shape[1] 
        mu = (np.sum(self.samples[:,None,:] * g, 2) / q).T
        sigma = self.samples.T[:,None,:] - mu
        sigma = sigma.T * g
        sigma = np.moveaxis(sigma, 0, 1)
        sigma = sigma @ np.moveaxis(sigma, 1, 2)
        sigma /= q[:,None,None]
        return w, mu, sigma

    def fit(self, pars = None, niter = 10):
        w, mu, sigma = pars if pars else self.fld.init_mask()

        for i in range(niter):
            g, q = self.e_step(w, mu, sigma)
            w, mu_new, sigma = self.m_step(g, q)
            tps_map = tps.f(mu, mu_new)
            mu = np.array([tps_map(*ui) for ui in mu])

            plt.imshow(self.fld0.T)
            plt.scatter(*mu.T, s = 4, color = 'red')
            plt.show()

        return w, mu, sigma
