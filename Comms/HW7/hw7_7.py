import math
import random
import numpy as np
from numpy.random import rand
from srrc1 import srrc1
import matplotlib.pyplot as plt
from deriv import deriv

alpha = 1.0 # excess bandwidth
Nsym = 1000 # number of symbols
phase_offset_deg = 0
phase_offset_rad = phase_offset_deg*np.pi/180
NsymPlus1 = Nsym + 1
N = 9
Lp = 30 # SRRC truncation length
Ts = 1 # symbol time
LOG2_M = 2
M = 4
T = Ts/N # sample time
carrier_freq = 2.2
cos_smple_rate = N/carrier_freq
TwoPiTs = np.pi*2*Ts
L = 6
a = 1 # PAM amplitude
LUT1 = np.array([-1,-1,1,1])*a # LUT for x-array
LUT2 = np.array([-1,1,1,-1])*a
srrcout = srrc1(alpha,N,Lp,Ts); # get the square-root raised cosine pulse

plt.figure(0); plt.clf()
plt.plot(srrcout)
plt.show()
# breakpoint()

k0 = 2
kp = 0.8
zeta = 1/np.sqrt(2)
BnT = 0.01

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

print(len(r_t))

n = [x / cos_smple_rate for x in range(-1*(len(r_t)), len(r_t))]
I_r = []
for i in range(0, len(r_t)):
    I_r.append(r_t[i]*np.sqrt(2)*np.cos(TwoPiTs*n[i]))

Q_r = []
for i in range(0, len(r_t)):
    Q_r.append(-1*np.sqrt(2)*r_t[i]*np.sin(TwoPiTs*n[i]))



x = np.convolve(I_r,srrcout) # the matched filter

y = np.convolve(Q_r,srrcout) # the matched filter

x_deriv, x_delay = deriv(L, x, T, TwoPiTs)
y_deriv, y_delay = deriv(L, y, T, TwoPiTs)

x_deriv = x_deriv[L:]
x_delay = x_delay[L:]
y_deriv = y_deriv[L:]
y_delay = y_delay[L:]

D = 1
x_deriv = x_deriv[1::D]
x_delay = x_delay[1::D]
y_deriv = y_deriv[1::D]
y_delay = y_delay[1::D]

print((len(x_deriv)))
print((len(x_delay)))
print((len(y_deriv)))
print((len(y_delay)))
print((len(y)))
print((len(x)))


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
first_sample = int(2*Lp/D)
nxt_smpl = first_sample
slice_out = []
theta = 0
z_past = 0.0
prev_loop_fil_o = 0.0
cnt = 1
storred_x = []
storred_y = []
e_list = []
mu_list = []
W1 = 1
e = 0
NC01 = 1
MU1 = 0
cnt = 0
V21 = 0
FX1 = 0
FX2 = 0
FX3 = 0
FX4 = 0
FX5 = 0
FDX1 = 0
FDX2 = 0
FDX3 = 0
FDX4 = 0
FDX5 = 0

FY1 = 0
FY2 = 0
FY3 = 0
FY4 = 0
FY5 = 0
FDY1 = 0
FDY2 = 0
FDY3 = 0
FDY4 = 0
FDY5 = 0

for n in range(first_sample+3, int(NsymPlus1*N/D)+first_sample):
    temp = NC01 - W1
    if temp < 0:
        strobe = 1
        mu = NC01/W1
        nco = 1+temp
    else:
        strobe = 0
        mu = MU1
        nco = temp
    
    if strobe == 0:
        e = 0
    else:
        cnt += 1

        FX1 = x_delay[n]
        FDX1 = x_deriv[n]

        vF3 = FX4/6 - FX3/2 + FX2/2 - FX1/6
        vF2 = FX3/2 - FX2 + FX1/2
        vF1 = -FX4/6 + FX3 - FX2/2 -FX1/3
        vF0 = FX2
        xi = (vF3 * mu + (vF2 * mu + vF1)) * mu + vF0

        vF3 = FDX4/6 - FDX3/2 + FDX2/2 - FDX1/6
        vF2 = FDX3/2 - FDX2 + FDX1/2
        vF1 = -FDX4/6 + FDX3 - FDX2/2 -FDX1/3
        vF0 = FDX2
        xdi = (vF3 * mu + (vF2 * mu + vF1)) * mu + vF0

        FY1 = y_delay[n]
        FDYI = y_deriv[n]

        vF3 = FY4/6 - FY3/2 + FY2/2 - FY1/6
        vF2 = FY3/2 - FY2 + FY1/2
        vF1 = -FY4/6 + FY3 - FY2/2 -FY1/3
        vF0 = FY2
        yi = (vF3 * mu + (vF2 * mu + vF1)) * mu + vF0

        vF3 = FDY4/6 - FDY3/2 + FDY2/2 - FDY1/6
        vF2 = FDY3/2 - FDY2 + FDY1/2
        vF1 = -FDY4/6 + FDY3 - FDY2/2 -FDY1/3
        vF0 = FDY2
        ydi = (vF3 * mu + (vF2 * mu + vF1)) * mu + vF0

        storred_x.append(xi)
        storred_y.append(yi)
        #slice and store value
        for k in range(0, M):
            if k == 0:
                shortest_dist = np.sqrt((LUT1[k]-xi)**2 + (LUT2[k]-yi)**2)
                shat = k
            elif (shortest_dist > np.sqrt((LUT1[k]-xi)**2 + (LUT2[k]-yi)**2)):
                shortest_dist = np.sqrt((LUT1[k]-xi)**2 + (LUT2[k]-yi)**2)
                shat = k
        slice_out.append(shat)

        #compute e
        e = ((np.sign(xi) * xdi)) + ((np.sign(yi) * ydi))
        # e_cubic = ((np.sign(xi_cubic) * xdi_cubic)) + ((np.sign(yi_cubic) * ydi_cubic))

        e_list.append(e)

        # if (cnt < 50):
        #     print("strobe is high for sample ", n , " xi is ", xi, " xdi is ",xdi, " yi is ", yi, " ydi is ", ydi, " mu is ", mu, " error is ", e)
    
    v1 = k1*e*kp
    v2 = V21 + k2*e*kp
    v = v1+v2
    mu_list.append(mu)

    w = -v + 1/N

    FX4 = FX3
    FX3 = FX2
    FX2 = FX1
    FX1 = x_delay[n]


    FDX4 = FDX3
    FDX3 = FDX2
    FDX2 = FDX1
    FDX1 = x_deriv[n]

    FY4 = FY3
    FY3 = FY2
    FY2 = FY1
    FY1 = y_delay[n]

    FDY4 = FDY3
    FDY3 = FDY2
    FDY2 = FDY1
    FDY1 = y_deriv[n]

    # if (n < 300):
        # print("W1 is ", W1, " NC01 is ", NC01, " loop_fil_o is ", v)

    NC01 = nco
    MU1 = mu
    W1 = w


plt.figure(4); plt.clf()
plt.plot(x_deriv[2*Lp:int(len(x_deriv)/10)])
plt.plot(x_delay[2*Lp:int(len(x_deriv)/10)])
plt.title("Real Differentated Signal vs Delayed Signal")
plt.legend(["Differentated Signal", "Delayed Signal"], loc="lower right")
plt.grid()

plt.figure(5); plt.clf()
plt.plot(y_deriv[2*Lp:int(len(y_deriv)/10)])
plt.plot(y_delay[2*Lp:int(len(y_deriv)/10)])
plt.title("Imag Differentated Signal vs Delayed Signal")
plt.legend(["Differentated Signal", "Delayed Signal"], loc="lower right")
plt.grid()

plt.figure(7)
plt.scatter(storred_x[65:], storred_y[65:],c = 'blue')
plt.grid()
plt.figure(8)
plt.plot(e_list)
plt.title("error")
plt.grid()
plt.figure(9)
plt.title("mu")
plt.plot(mu_list)
plt.grid()
plt.show()


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