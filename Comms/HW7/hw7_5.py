from deriv import deriv
import numpy as np
import matplotlib.pyplot as plt

r = []

n = 80
L = 6
N = 10
WcT = np.pi
w0 = int(n/L)
T = 1/n
Wc = WcT/T

for t in np.arange(0,N*w0,w0/n):
    r.append(np.sin(t))

deriv_r, delay = deriv(L, r, T, w0)