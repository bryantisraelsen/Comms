import math
import random
import numpy as np
from numpy.random import rand
from srrc1 import srrc1
import matplotlib.pyplot as plt

alpha = 1.0 # excess bandwidth
Nsym = 1000 # number of symbols
phase_offset_deg = 130
phase_offset_rad = phase_offset_deg*np.pi/180
NsymPlus1 = Nsym + 1
N = 7 # samples per symbol
Lp = 60 # SRRC truncation length
Ts = 1 # symbol time
LOG2_M = 2
M = 4
T = Ts/N # sample time
carrier_freq = 2.2
cos_smple_rate = N/carrier_freq
TwoPiTs = np.pi*2*Ts
a = 1 # PAM amplitude
LUT1 = np.array([-1,-1,1,1])*a # LUT for x-array
LUT2 = np.array([-1,1,1,-1])*a
srrcout = srrc1(alpha,N,Lp,Ts); # get the square-root raised cosine pulse

k0 = 1
kp = 1
zeta = 1/np.sqrt(2)
BnT = 0.065

den = zeta+(1/(4*zeta))
bigterm = (BnT/den)

k1top = 4*zeta*bigterm
k1bottom = 1+2*zeta*bigterm+bigterm**2
k1 = k1top/(k1bottom*kp*k0)

k2top = 4*((BnT/den)**2)
k2bottom = 1+2*zeta*bigterm+bigterm**2
k2 = k2top/(k2bottom*k0*kp)

print("k1 is", k1, "\nk2 is", k2)

rcout = np.convolve(srrcout,srrcout)
bits = []
data_bits = []
bits = (rand(Nsym*LOG2_M)> 0.5).astype(int) # generate random bits {0,1}
for i in range(0, int(len(bits)/LOG2_M)):   # S/P converter (makes it log_2_M size)
    log2_bit_list = []
    res = 0
    for j in range(0, LOG2_M):
        log2_bit_list.append(bits[LOG2_M*i+j])
    for ele in log2_bit_list:
        res = (res << 1) | ele
    data_bits.append(res)

#encoder
initsym = 0
sent_sym = []
sent_sym.append(initsym)
prev_sym = initsym
for i in range(0, len(data_bits)):
    curr_sym = data_bits[i]
    send_sym = prev_sym + curr_sym
    if send_sym > 3:
        send_sym -= 4
    # print("send_sym for iteration ", i, " is ", send_sym, " because prev sym was ", prev_sym, " and curr sym is ", curr_sym)
    sent_sym.append(send_sym)
    prev_sym = send_sym

ampa_0 = LUT1[sent_sym] # map the bits to {+1,-1} values
ampa_1 = LUT2[sent_sym]
# for i in range(0,Nsym): upsampled[N*i] = ampa[i]
#breakpoint()
upsampled_0 = np.zeros((N*NsymPlus1,1))
upsampled_1 = np.zeros((N*NsymPlus1,1))
upsampled_0[range(0,N*NsymPlus1,N)] = ampa_0.reshape(NsymPlus1,1)
upsampled_1[range(0,N*NsymPlus1,N)] = ampa_1.reshape(NsymPlus1,1)
s_i = np.convolve(upsampled_0.reshape((N*NsymPlus1,)),srrcout) # the transmitted signal
s_q = np.convolve(upsampled_1.reshape((N*NsymPlus1,)),srrcout) # the transmitted signal
# plt.figure(3); plt.clf()
# plt.plot(s_i[:int(len(s_i)/10)])
# plt.figure(4); plt.clf()
# plt.plot(s_q[:int(len(s_q)/10)])
# plt.show()

n = [x / cos_smple_rate for x in range(-1*(len(s_i)), len(s_i))]

cos_s = []
for i in range(0,len(s_i)):
    cos_s.append(np.sqrt(2)*s_i[i]*np.cos(TwoPiTs*n[i]+phase_offset_rad))

sin_s = []
for i in range(0,len(s_q)):
    sin_s.append(-1*np.sqrt(2)*s_q[i]*np.sin(TwoPiTs*n[i]+phase_offset_rad))

s_t = []
for i in range(0, len(cos_s)):
    s_t.append(sin_s[i]+cos_s[i])

r_t = s_t

n = [x / cos_smple_rate for x in range(-1*(len(r_t)), len(r_t))]
I_r = []
for i in range(0, len(r_t)):
    I_r.append(r_t[i]*np.sqrt(2)*np.cos(TwoPiTs*n[i]))

Q_r = []
for i in range(0, len(r_t)):
    Q_r.append(-1*np.sqrt(2)*r_t[i]*np.sin(TwoPiTs*n[i]))



x = np.convolve(I_r,srrcout) # the matched filter
plt.figure(9); plt.clf()
plt.plot(x[:int(len(x)/10)])

y = np.convolve(Q_r,srrcout) # the matched filter
plt.figure(10); plt.clf()
plt.plot(y[:int(len(y)/10)])

plt.show()

# plt.figure(11); plt.clf()
# plt.plot(x[:(2*Lp+1) + N*20])
# for i in range(20):
#     plt.plot(
#         np.array([2*Lp + i*N, 2*Lp + i*N]),
#         np.array([-2,2]),color='gray',linewidth=0.25)
# plt.show()

# plt.figure(12); plt.clf()
# plt.plot(y[:(2*Lp+1) + N*20])
# for i in range(20):
#     plt.plot(
#         np.array([2*Lp + i*N, 2*Lp + i*N]),
#         np.array([-2,2]),color='gray',linewidth=0.25)
# plt.show()
      
# first peak at 2*Lp, then every N samples after that
offset = (2*Lp - np.floor(N/2)).astype(int)
# ˆ ˆ
# 1st correlation |
# peak |
# move to center
Nsymtoss = 2*np.ceil(Lp/N) # throw away symbols at the end
nc = (np.floor((len(x) - offset - Nsymtoss*N)/N)).astype(int) # number of points of signal to plot
xreshape = x[offset:offset + nc*N].reshape(nc,N)
# plt.figure(13); plt.clf()

# plt.plot(np.arange(-np.floor(N/2), np.floor(N/2)+1), xreshape.T,color='b', linewidth=0.25)

Nsymtoss = 2*np.ceil(Lp/N) # throw away symbols at the end
nc = (np.floor((len(y) - offset - Nsymtoss*N)/N)).astype(int) # number of points of signal to plot
yreshape = y[offset:offset + nc*N].reshape(nc,N)
# plt.figure(14); plt.clf()

# plt.plot(np.arange(-np.floor(N/2), np.floor(N/2)+1), yreshape.T,color='b', linewidth=0.25)
# plt.show()

#slice
slice_out = []
first_sample = 2*Lp
nxt_smpl = first_sample
slice_out = []
theta = 0
z_past = 0.0
prev_loop_fil_o = 0.0
cnt = 1
storred_x = []
storred_y = []
theta_e_list = []
theta_list = []
for n in range(first_sample, NsymPlus1*N+first_sample):
    cnt -= 1
    if (cnt == 0):
        cnt = N
        #compute rotation parameters
        C = np.cos(-1*theta)
        S = np.sin(-1*theta)
        #perform rotation
        xr = C*x[n] - S*y[n]
        yr = S*x[n] + C*y[n]
        #slice
        for k in range(0, M):
            if k == 0:
                shortest_dist = np.sqrt((LUT1[k]-xr)**2 + (LUT2[k]-yr)**2)
                shat = k
            elif (shortest_dist > np.sqrt((LUT1[k]-xr)**2 + (LUT2[k]-yr)**2)):
                shortest_dist = np.sqrt((LUT1[k]-xr)**2 + (LUT2[k]-yr)**2)
                shat = k
        slice_out.append(shat)
        a0hat = LUT1[shat]
        a1hat = LUT2[shat]
        theta1 = yr*a0hat
        theta2 = xr*a1hat
        # theta1 = yr*a0hat
        # theta2 = xr*a1hat
        theta_e = theta1-theta2
        # print("\n\ntheta_e is ", theta_e, " for iteration ", n)
        detector_O = kp*theta_e
        #detector_O is out of args block
        #z_pres is input for z^-1 block
        #z_past is output of z^-1 block
        z_pres = detector_O*k2+z_past
        loop_fil_o = detector_O*k1+z_pres
        #update z_past
        z_past = z_pres
        # print("z_pres is ", z_pres)
        # print("theortical theta e is ", k0*prev_loop_fil_o+TwoPiTs)
        theta += k0*prev_loop_fil_o
        prev_loop_fil_o = loop_fil_o
        storred_x.append(xr)
        storred_y.append(yr)
        theta_list.append(theta)
        theta_e_list.append(theta_e)
        #DDS
alpha = np.arange(0.2, 1, 0.0008)
plt.figure(13)
for i in range(len(storred_x)-1):
    plt.scatter(storred_x[i], storred_y[i],c = 'blue', alpha = alpha[i])
plt.grid()
plt.figure(14)
plt.scatter(storred_x[65:], storred_y[65:],c = 'blue')
plt.grid()
plt.figure(15)
plt.plot(theta_e_list)
plt.grid()
plt.figure(16)
plt.plot(theta_list)
plt.grid()
plt.show()

# print("Length of sent sym is ", len(sent_sym), "\n\nLength of received symbols are ",len(slice_out))

data_out = []
prev_rec = slice_out[0]
for i in range(1, len(slice_out)):
    rec_sym = slice_out[i]
    if rec_sym-prev_rec < 0:
        data_out.append(rec_sym-prev_rec+4)
    else:
        data_out.append(rec_sym-prev_rec)
    prev_rec = rec_sym

sym_err = 0
for i in range(len(data_out)):
    if data_out[i] != data_bits[i]:
        sym_err += 1
print("\n\nThe probability of error in the system is ", sym_err/len(data_bits))


if data_out == data_bits:
    print("\nSent bits are the same as the received bits. Good work!")
else:
    print("\nBits received were not the bits sent...")