import math
import random
import numpy as np
from numpy.random import rand
from srrc1 import srrc1
import matplotlib.pyplot as plt

alpha = 1.0 # excess bandwidth
N = 11 # samples per symbol
Lp = 60 # SRRC truncation length
Ts = 1 # symbol time
LOG2_M = 2
T = Ts/N # sample time
cos_smpl_rate = 5.0 #carrier frequency
carrier_freq = np.pi*2*Ts
srrcout = srrc1(alpha,N,Lp,Ts); # get the square-root raised cosine pulse

rcout = np.convolve(srrcout,srrcout)
# peak at 2Lp
plt.figure(1); plt.clf()
plt.plot(srrcout)
plt.show()
a = 1 # PAM amplitude
LUT1 = np.array([-1,-1,1,1])*a # LUT for x-array
LUT2 = np.array([-1,1,1,-1])*a
Nsym = 300 # number of symbols
bits = []
index_bits = []
bits = (rand(Nsym*LOG2_M)> 0.5).astype(int) # generate random bits {0,1}
for i in range(0, int(len(bits)/LOG2_M)):   # S/P converter (makes it log_2_M size)
    log2_bit_list = []
    res = 0
    for j in range(0, LOG2_M):
        log2_bit_list.append(bits[LOG2_M*i+j])
    for ele in log2_bit_list:
        res = (res << 1) | ele
    index_bits.append(res)

print(index_bits)
ampa_0 = LUT1[index_bits] # map the bits to {+1,-1} values
ampa_1 = LUT2[index_bits]
# for i in range(0,Nsym): upsampled[N*i] = ampa[i]
breakpoint()
upsampled_0 = np.zeros((N*Nsym,1))
upsampled_1 = np.zeros((N*Nsym,1))
upsampled_0[range(0,N*Nsym,N)] = ampa_0.reshape(Nsym,1)
upsampled_1[range(0,N*Nsym,N)] = ampa_1.reshape(Nsym,1)
plt.figure(2); plt.clf()
plt.stem(upsampled_0,linefmt='b',markerfmt='.',basefmt='b-')
plt.stem(upsampled_1,linefmt='r',markerfmt='.',basefmt='b-')
plt.show()
s_i = np.convolve(upsampled_0.reshape((N*Nsym,)),srrcout) # the transmitted signal
s_q = np.convolve(upsampled_1.reshape((N*Nsym,)),srrcout) # the transmitted signal
plt.figure(3); plt.clf()
plt.plot(s_i)
plt.figure(4); plt.clf()
plt.plot(s_q)
plt.show()

n = [x / cos_smpl_rate for x in range(-1*(len(s_i)), len(s_i))]

cos_s = []
for i in range(0,len(s_i)):
    cos_s.append(np.sqrt(2)*s_i[i]*np.cos(carrier_freq*n[i]))

sin_s = []
for i in range(0,len(s_q)):
    sin_s.append(-1*np.sqrt(2)*s_q[i]*np.sin(carrier_freq*n[i]))

plt.figure(5); plt.clf()
plt.plot(cos_s[:200])

plt.figure(6); plt.clf()
plt.plot(sin_s[:200])
plt.show()

s_t = []
for i in range(0, len(cos_s)):
    s_t.append(sin_s[i]+cos_s[i])

r_t = s_t

n = [x / cos_smpl_rate for x in range(-1*(len(r_t)), len(r_t))]
I_r = []
for i in range(0, len(r_t)):
    I_r.append(r_t[i]*np.sqrt(2)*np.cos(carrier_freq*n[i]))

Q_r = []
for i in range(0, len(r_t)):
    Q_r.append(-1*np.sqrt(2)*r_t[i]*np.sin(carrier_freq*n[i]))

plt.figure(13); plt.clf()
plt.plot(r_t)

plt.figure(7); plt.clf()
plt.plot(I_r)

plt.figure(8); plt.clf()
plt.plot(Q_r)
plt.show()

x = np.convolve(I_r,srrcout) # the matched filter
plt.figure(9); plt.clf()
plt.plot(x)

y = np.convolve(Q_r,srrcout) # the matched filter
plt.figure(10); plt.clf()
plt.plot(y)

plt.show()
print('bits=',index_bits[:10])

plt.figure(10); plt.clf()
plt.plot(x, y)

plt.figure(11); plt.clf()
plt.plot(s_i, s_q)
plt.show()
