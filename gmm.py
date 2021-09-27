import tps
import numpy as np
import matplotlib.pyplot as plt

from scipy import ndimage, signal

class Peak():
    def __init__(self, *args, **kwds):
        self.update(*args, **kwds)

    def update(self, w, u, s):
        self.w = w
        self.u = u
        self.s = s

    def get(self):
        return self.w, self.u, self.s

class PeakArray():
    def __init__(self, w0 = [], u0 = [], s0 = []):
        self.peaks = [Peak(*theta) for theta in zip(w0,u0,s0)]
        self.w, self.u, self.s = self.get()

    def update(self):
        self.w, self.u, self.s = self.get()

    def get(self, idx = None):
        if idx is None:
            vals = [p.get() for p in self.peaks]
        else:
            vals = [self.peaks[i].get() for i in idx]
        return [np.array(i) for i in zip(*vals)]

    def in_region(self, r, cog):
        dst = np.sqrt(np.sum((self.u - cog)**2, 1))
        return np.nonzero(dst < r)[0]

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

    def find_1d_peaks(self, axis = 0, quantile = 0.5):
        s = np.sum(self.fld, axis)
        cog = np.average(np.arange(len(s)), weights = s**2)

        height = np.quantile(s, quantile)
        pks, other = signal.find_peaks(s, height, distance = 5)
        actual_height = other['peak_heights']
        order = actual_height.argsort()[::-1]

        pks = pks[order]
        actual_height = actual_height[order]

        center_pk_idx = pks[np.argmin(np.abs(pks - cog))]
        lpeak_idx = list(pks[pks < center_pk_idx])
        rpeak_idx = list(pks[pks > center_pk_idx])

        all_pks = lpeak_idx[:9] + [center_pk_idx] + rpeak_idx[:9]

        for pk in pks:
            if len(all_pks) >= 19:
                break
            elif pk not in all_pks:
                all_pks.append(pk)

        return all_pks

    def estimate_peaks(self):
        ta = self.find_1d_peaks(1)
        ax = self.find_1d_peaks(0)
        return np.array(np.meshgrid(ta, ax)).reshape(2,ncry2)

    def init_mask(self, ncrystals = 19):
        ncry2 = ncrystals**2 
        mu0 = self.estimate_peaks()
        w0 = np.array(1.0/ncry2).repeat(ncry2)

        ta_edges = self.edges(1)
        ax_edges = self.edges(0)
        ta_diff = np.diff(ta_edges)[0] / ncrystals
        ax_diff = np.diff(ax_edges)[0] / ncrystals
        sigma0 = np.array([[ta_diff,0],[0,ax_diff]])[None,...].repeat(ncry2, axis = 0)

        return PeakArray(w0, mu0.T, sigma0)

class Gmm():
    @staticmethod
    def gaussian(c, w, mu, s):
        sigma_i = np.linalg.inv(s)
        sigma_d = np.linalg.det(s)

        # i->events,j/k->dimension of gaussian(2)
        # exp_val = -1/2 * (x @ sigma^-1 @ x.T)
        # (i,) array of scalars to pass to exp
        x = c.T - mu
        x = np.einsum('ij,jk,ij->i', x, -sigma_i/2, x)
        return w * np.exp(x) / np.sqrt((2*np.pi)**2 * sigma_d)

    def __init__(self, f, nsamples = 100_000):
        self.fld = Flood(f)
        self.samples = self.fld.sample(nsamples)
        sz = self.fld.fld.shape[0]
        self.fld0, *_ = np.histogram2d(*self.samples, sz, [[0,sz-1],[0,sz-1]])

    def e_step(self, samp, w, mu, s):
        g = np.array([Gmm.gaussian(samp, *theta) for theta in zip(w,mu,s)])
        g /= g.sum(0)
        q = g.sum(1)
        return g, q

    def m_step(self, samp, g, q):
        w1  = q / g.shape[1] 
        mu1 = (g @ samp.T) / q[:,None]

        # k->events, l->features, i/j->coords
        # sigma_l = sum_k {(ck.T @ ck) * gkl} / ql
        # l x 2 x 2 matrix of sigmas
        c   = samp.T[:,None,:] - mu1
        s1  = np.einsum('kli,klj,lk,l->lij',c,c,g,1/q)
        return w1, mu1, s1

    def fit(self, niter = 10):
        pks = self.fld.init_mask()

        r = 0
        done = set()
        cog = np.array(ndimage.center_of_mass(self.fld0)[::-1])

        sz = self.fld.fld.shape[0]
        self.fld0, *_ = np.histogram2d(*self.samples, sz, [[0,sz-1],[0,sz-1]])

        plt.imshow(self.fld0.T)
        plt.scatter(*pks.u.T, s = 4, color = 'red')
        plt.show()

        for _ in range(10000):
            print(len(done))
            r += 1 
            current = pks.in_region(r, cog)
            new = list(set(current) - done)
            nnew = len(new)

            if nnew > 0:
                samples_dst = np.sqrt(np.sum((self.samples - cog[:,None])**2, 0))
                c = self.samples[:,samples_dst < r]
                self.fld0, *_ = np.histogram2d(*c, sz, [[0,sz-1],[0,sz-1]])

                converged = np.array([False] * nnew)
                ll_old = np.array([-np.inf] * nnew)

                while not np.all(converged):
                    #w_v, mu_v, s_v = w[cur,], mu[cur,], s[cur,]
                    w, u, s = pks.get(current)
                    g, q = self.e_step(c, w, u, s)
                    w_new, u_new, s_new = self.m_step(c, g, q)

                    ll_new = np.array([np.mean(np.log(gi + np.finfo('float').eps)) for gi in g])

                    """
                    if nnew > 1:
                        tps_map = tps.f(mu_v, mu_new)
                        mu[new] = np.array([tps_map(*ui) for ui in mu_new[:nnew,]])
                    else:
                        mu[new] = mu_new[:nnew,]
                    """

                    for i in range(len(current)):
                        idx = current[i]
                        if idx in new:
                            new_idx = new.index(idx)
                            converged[new_idx] = (ll_new[i] - ll_old[new_idx]) < 1e-6
                            ll_old[new_idx] = ll_new[i]

                            wi,ui,si = w_new[i,], u_new[i,], s_new[i,]
                            pks.peaks[idx].update(wi,ui,si)

                    pks.update()

                    """
                    mu[new] = mu_new[:nnew,]
                    w[new] = w_new[:nnew,]
                    s[new] = s_new[:nnew,]
                    """

                    plt.imshow(self.fld0.T)
                    plt.scatter(*pks.u.T, s = 4, color = 'red')
                    plt.show()

                done = set(current)


