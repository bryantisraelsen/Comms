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
LOG2_M = 3
T = Ts/N # sample time
Fs = 1/T
carrier_freq = 3
cos_smple_rate = N/carrier_freq
twoPiTs = np.pi*2*Ts
a = 1 # PAM amplitude
LUT1 = np.array([0,-1,-0.6235,0.2225,0.9010,0.9010,0.2225,-0.6235])*a # LUT for x-array
LUT2 = np.array([0,0,0.7818,0.9749,0.4339,-0.4339,-0.9749,-0.7818])*a
srrcout = srrc1(alpha,N,Lp,Ts); # get the square-root raised cosine pulse

rcout = np.convolve(srrcout,srrcout)
# peak at 2Lp
plt.figure(1); plt.clf()
plt.plot(srrcout)
plt.show()
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

n = [x / cos_smple_rate for x in range(-1*(len(s_i)), len(s_i))]

cos_s = []
for i in range(0,len(s_i)):
    cos_s.append(np.sqrt(2)*s_i[i]*np.cos(twoPiTs*n[i]))

sin_s = []
for i in range(0,len(s_q)):
    sin_s.append(-1*np.sqrt(2)*s_q[i]*np.sin(twoPiTs*n[i]))

plt.figure(5); plt.clf()
plt.plot(cos_s[:200])

plt.figure(6); plt.clf()
plt.plot(sin_s[:200])
plt.show()

s_t = []
for i in range(0, len(cos_s)):
    s_t.append(sin_s[i]+cos_s[i])

r_t = s_t

n = [x / cos_smple_rate for x in range(-1*(len(r_t)), len(r_t))]
I_r = []
for i in range(0, len(r_t)):
    I_r.append(r_t[i]*np.sqrt(2)*np.cos(twoPiTs*n[i]))

Q_r = []
for i in range(0, len(r_t)):
    Q_r.append(-1*np.sqrt(2)*r_t[i]*np.sin(twoPiTs*n[i]))

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


plt.figure(12); plt.clf()
plt.plot(x[:(2*Lp+1) + N*20])
for i in range(20):
    plt.plot(
        np.array([2*Lp + i*N, 2*Lp + i*N]),
        np.array([-2,2]),color='gray',linewidth=0.25)
plt.show()

plt.figure(12); plt.clf()
plt.plot(y[:(2*Lp+1) + N*20])
for i in range(20):
    plt.plot(
        np.array([2*Lp + i*N, 2*Lp + i*N]),
        np.array([-2,2]),color='gray',linewidth=0.25)
plt.show()
      
# first peak at 2*Lp, then every N samples after that
offset = (2*Lp - np.floor(N/2)).astype(int)
# ˆ ˆ
# 1st correlation |
# peak |
# move to center
Nsymtoss = 2*np.ceil(Lp/N) # throw away symbols at the end
nc = (np.floor((len(x) - offset - Nsymtoss*N)/N)).astype(int) # number of points of signal to plot
xreshape = x[offset:offset + nc*N].reshape(nc,N)
plt.figure(13); plt.clf()

plt.plot(np.arange(-np.floor(N/2), np.floor(N/2)+1), xreshape.T,color='b', linewidth=0.25)

Nsymtoss = 2*np.ceil(Lp/N) # throw away symbols at the end
nc = (np.floor((len(y) - offset - Nsymtoss*N)/N)).astype(int) # number of points of signal to plot
xreshape = y[offset:offset + nc*N].reshape(nc,N)
plt.figure(14); plt.clf()

plt.plot(np.arange(-np.floor(N/2), np.floor(N/2)+1), xreshape.T,color='b', linewidth=0.25)
plt.show()

first_sample = 2*Lp
nxt_smpl = first_sample
x_received = []
y_received = []
received_bits = []
#slice (this will change depending on the constellation)
for i in range(Nsym):
    x_received.append(x[nxt_smpl])
    y_received.append(y[nxt_smpl])
    nxt_smpl = nxt_smpl+N

plt.figure(15); plt.clf()
plt.scatter(x_received,y_received)
plt.show()

faxis = np.linspace((-1*Fs/2), Fs/2, len(s_i)*2-1)
plt.figure(16); plt.clf()
plt.plot(faxis, 10*np.log10(np.abs(np.fft.fftshift(np.fft.fft(np.correlate(s_i,s_i,'full'))))**2))
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude (dB)")
plt.show()

plt.figure(17); plt.clf()
plt.plot(faxis, 10*np.log10(np.abs(np.fft.fftshift(np.fft.fft(np.correlate(s_t,s_t,'full'))))**2))
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude (dB)")
plt.show()

faxis1 = np.linspace((-1*Fs/2), Fs/2, len(x)*2-1)
plt.figure(18); plt.clf()
plt.plot(faxis1, 10*np.log10(np.abs(np.fft.fftshift(np.fft.fft(np.correlate(x,x,'full'))))**2))
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude (dB)")
plt.show()

plt.figure(19); plt.clf()
plt.plot(faxis1, 10*np.log10(np.abs(np.fft.fftshift(np.fft.fft(np.correlate(y,y,'full'))))**2))
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude (dB)")
plt.show()