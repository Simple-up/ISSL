import numpy as np
import random
import math
from numba import jit
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt

# assume x of shape [2, L]

@jit
def _normalize(x):
   return (x-np.mean(x))/np.std(x) 

class Normalize:
    def __init__(self):
        return
    
    def __call__(self, x):
        return _normalize(x)

class PhaseShift:
    def __init__(self, p):
        self.p = p

    def __call__(self, x):
        if random.uniform(0., 1.) < self.p:
            x = x[0,:] + 1j*x[1,:]
            x = x*np.exp(1j*np.random.uniform(0, math.pi*2.))
            x = np.array([np.real(x), np.imag(x)], dtype='float32')
        return x

class Permutation:
    def __init__(self, seg):
        self.seg = seg

    def __call__(self, x):
        # seg = random.randint(1, self.seg)
        seg = self.seg
        length = x.shape[-1]
        y = np.zeros(x.shape)
        splits = np.array([0, length], dtype='int')
        for _ in range(seg-1):
            splits = np.append(splits, random.randint(0, length))
        splits = np.sort(splits)
        segid = np.random.permutation(seg)
        yi = 0
        for i in range(seg):
            li = splits[segid[i]+1] - splits[segid[i]]
            y[:, yi:yi+li] = x[:, splits[segid[i]]:splits[segid[i]+1]]
            yi += li
        # plt.plot(y[0,:])
        # plt.plot(y[1,:])
        # plt.show()
        return y

class Jitter:
    def __init__(self, sigma):
        self.sigma = sigma

    def __call__(self, x):
        jitter = np.random.normal(loc=0., scale=self.sigma, size=x.shape)
        return x + jitter

class Jitter2:
    def __init__(self, sigma):
        self.sigma = sigma

    def __call__(self, x):
        jitter1 = np.random.normal(loc=0., scale=self.sigma, size=x.shape[-1])
        jitter2 = np.random.normal(loc=0., scale=self.sigma, size=x.shape[-1])
        jitter3 = np.vstack((jitter1,jitter2))
        # print(x.shape)
        # print(jitter1)
        # print('*********************************************')
        # print(jitter2)
        # print(x.shape[-1])
        return x + jitter3

class JitterWarp:
    def __init__(self, sigma, knots):
        self.sigma = sigma
        self.knots = knots
    
    def __call__(self, x):
        orig_steps = np.arange(x.shape[-1])
        # warp_steps = np.random.uniform(0, x.shape[-1], self.knots)
        # warp_steps = np.sort(np.append(warp_steps, [0, x.shape[-1]]))
        warp_steps = np.linspace(0, x.shape[-1]-1., num=self.knots+2)
        warp_values = np.random.normal(loc=0., scale=self.sigma, size=self.knots+2)
        # print(warp_steps, warp_values)
        warper = np.array(CubicSpline(warp_steps, warp_values)(orig_steps))
        # plt.plot(warper)
        # plt.show()
        x = x + warper
        return x

class Scaling:
    def __init__(self, sigma):
        self.sigma = sigma

    def __call__(self, x):
        factor = np.random.normal(loc=1., scale=self.sigma)
        # print(factor)
        return x*factor

class RandomScale:
    def __init__(self, sigma):
        self.sigma = sigma
    
    def __call__(self, x):
        factor = np.random.normal(loc=1., scale=self.sigma, size=x.shape[-1])
        return x*factor

class SegRandomScale:
    def __init__(self, sigma, seglen):
        self.sigma = sigma
        self.seglen = seglen

    def __call__(self, x):
        factor = np.random.normal(loc=1., scale=self.sigma, size=x.shape[-1]//self.seglen)
        factor = np.repeat(factor, self.seglen)
        return x*factor

class MagnitudeWarp:
    def __init__(self, sigma, knots):
        self.sigma = sigma
        self.knots = knots
    
    def __call__(self, x):
        orig_steps = np.arange(x.shape[-1])
        # warp_steps = np.random.uniform(0, x.shape[-1], self.knots)
        # warp_steps = np.sort(np.append(warp_steps, [0, x.shape[-1]]))
        warp_steps = np.linspace(0, x.shape[-1]-1., num=self.knots+2)
        warp_values = np.random.normal(loc=1., scale=self.sigma, size=self.knots+2)
        # print(warp_steps, warp_values)
        warper = np.array(CubicSpline(warp_steps, warp_values)(orig_steps))
        # plt.plot(warper)
        # plt.show()
        x = x * warper
        return x

class RandomReverse:
    def __init__(self, p):
        self.p = p

    def __call__(self, x):
        # plt.plot(x[0,:])
        if random.uniform(0., 1.) < self.p:
            x = np.ascontiguousarray(x[:,::-1])
        # plt.plot(x[0,:])
        # plt.show()
        return x

class RandomExchange:
    def __init__(self, p):
        self.p = p
    
    def __call__(self, x):
        if random.uniform(0., 1.) < self.p:
            x = x[[1,0],:]
        return x

class RandomErase:
    def __init__(self, p, scale):
        self.p = p
        self.scale = scale

    def __call__(self, x):
        if random.uniform(0., 1.) < self.p:
            xlen = x.shape[-1]
            len1 = int(random.uniform(self.scale[0], self.scale[1])*xlen)
            start1 = random.randint(0, xlen-len1)
            x[:, start1:start1+len1] = 0
        return x

class RandomEraseScatter:
    def __init__(self, p, scale):
        self.p = p
        self.scale = scale
    
    def __call__(self, x):
        if random.uniform(0., 1.) < self.p:
            xlen = x.shape[-1]
            len1 = int(random.uniform(self.scale[0], self.scale[1])*xlen)
            idx = np.random.choice(np.arange(xlen), len1, replace=False)
            x[:, idx] = 0
        return x

class RandomErase2:
    def __init__(self, p, scale):
        self.p = p
        self.scale = scale

    def __call__(self, x):
        if random.uniform(0., 1.) < self.p:
            xlen = x.shape[-1]
            len1 = int(random.uniform(self.scale[0], self.scale[1])*xlen)
            start1 = random.randint(0, xlen-len1)
            x[0, start1:start1+len1] = 0
        if random.uniform(0., 1.) < self.p:
            xlen = x.shape[-1]
            len2 = int(random.uniform(self.scale[0], self.scale[1])*xlen)
            start2 = random.randint(0, xlen-len2)
            x[1, start2:start2+len2] = 0
        return x

class RandomCrop:
    def __init__(self, length):
        self.length = length

    def __call__(self, x):
        xlen = x.shape[-1]
        start = random.randint(0, xlen-self.length)
        x = x[:, start:start+self.length]
        # plt.plot(x[0,:])
        # plt.plot(x[1,:])
        # plt.show()
        return x

class FFT:
    def __call__(self, x):
        x = x[0,:] + 1j*x[1,:]
        xf = np.fft.fft(x)
        xf = np.stack([np.real(xf), np.imag(xf)])
        return xf

class IFFT:
    def __call__(self, xf):
        xf = xf[0,:] + 1j*xf[1,:]
        x = np.fft.ifft(xf)
        x = np.stack([np.real(x), np.imag(x)])
        # plt.plot(x[0,:])
        # plt.plot(x[1,:])
        # plt.show()
        return x