import math
import random
import numpy as np
from numpy.random import rand
from srrc1 import srrc1
import matplotlib.pyplot as plt

N = 121 # samples per symbol
Lp = 122 # SRRC truncation length
Ts = 1 # symbol time
T = Ts/N # sample time

MANN_0 = np.zeros(Lp*2+1)
MANN_0[62:122] = np.ones(60)
MANN_0[122:183] = np.negative(np.ones(61))
MANN = np.divide(MANN_0,np.sqrt(121))

MANNout = np.convolve(MANN,np.multiply(MANN, -1)) #convolve with time reversal of itself
# peak at 2Lp
plt.figure(1); plt.clf()
plt.plot(MANN)
plt.show()
a = 1 # PAM amplitude
LUT1d = np.array([-1,1])*a # 1-dimensional lookup table
Nsym = 100 # number of symbols

bits = (rand(Nsym)> 0.5).astype(int) # generate random bits {0,1}
print(bits)
ampa = LUT1d[bits] # map the bits to {+1,-1} values
# for i in range(0,Nsym): upsampled[N*i] = ampa[i]
breakpoint()
upsampled = np.zeros((N*Nsym,1))
upsampled[range(0,N*Nsym,N)] = ampa.reshape(Nsym,1)
plt.figure(2); plt.clf()
plt.stem(upsampled,linefmt='b',markerfmt='.',basefmt='b-')
s = np.convolve(upsampled.reshape((N*Nsym,)),MANN) # the transmitted signal
plt.figure(3); plt.clf()
plt.plot(s)
plt.show()
x = np.convolve(s,MANN) # the matched filter
plt.figure(4); plt.clf()
plt.plot(x[:(2*Lp+1) + N*20])
for i in range(20):
    plt.plot(
        np.array([2*Lp + i*N, 2*Lp + i*N]),
        np.array([-2,2]),color='gray',linewidth=0.25)
plt.show()
print('bits=',bits[:10])
      
# first peak at 2*Lp, then every N samples after that
offset = (2*Lp - np.floor(N/2)).astype(int)
# ˆ ˆ
# 1st correlation |
# peak |
# move to center
Nsymtoss = 2*np.ceil(Lp/N) # throw away symbols at the end
nc = (np.floor((len(x) - offset - Nsymtoss*N)/N)).astype(int) # number of points of signal to plot
xreshape = x[offset:offset + nc*N].reshape(nc,N)
plt.figure(5); plt.clf()

plt.plot(np.arange(-np.floor(N/2), np.floor(N/2)+1), xreshape.T,color='b', linewidth=0.25)
plt.show()

