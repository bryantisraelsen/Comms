import numpy as np
import numpy.matlib
import math
import matplotlib.pyplot as plt
import struct   # i
from deriv import deriv
packints = 'ii'

# Set parameters
f_eps        = 0.0 # Carrier frequency offset percentage (0 = no offset)
Phase_Offset = 0.0 # Carrier phase offset (0 = no offset)
t_eps        = 0.0 # Clock freqency offset percentage (0 = no offset)
T_offset     = 0.0 # Clock phase offset (0 = no offset)
Ts = 1             # Symbol period
N = 4              # Samples per symbol period
L = 6
T = Ts/N
TwoPiTs = np.pi*2*Ts
Omega0 = math.pi/2*(1+f_eps)
# Select modulation type
# Use 8 bits per symbol and 256 square QAM
B = 8;            # Bits per symbol (B should be even: 8, 6, 4, 2)
# B = 4;
bits2index = 2**np.arange(B-1,-1,-1)
M = 2 ** B       # Number of symbols in the constellation
Mroot = math.floor(2**(B/2))
a = np.reshape(np.arange(-Mroot+1,Mroot,2),(2*B,1))
b = np.ones((Mroot,1))
LUT = np.hstack((np.kron(a,b), np.kron(b,a)))
# will be of the form (for example)
#  -3, -3
#  -3, -1
#  -3, 1
#  ...
#   3, 3
# of shape (B^2, 2)


k0 = 0.8
kp = 1.0
zeta = 1/np.sqrt(2)
# BnT = 0.065
BnT = 0.105

den = zeta+(1/(4*zeta))
bigterm = (BnT/den)

k1top = 4*zeta*bigterm
k1bottom = 1+2*zeta*bigterm+bigterm**2
k1 = k1top/(k1bottom*kp*k0)

k2top = 4*((BnT/den)**2)
k2bottom = 1+2*zeta*bigterm+bigterm**2
k2 = k2top/(k2bottom*k0*kp)

print("k1 is", k1, "\nk2 is", k2)
# Scale the constellation to have unit energy
Enorm = np.sum(LUT ** 2) / M;
LUT = LUT/math.sqrt(Enorm);

if 0:
    plt.figure(1)
    # Plot the constellation
    plt.plot(LUT[:,0],LUT[:,1],'o');
    for i in range(0,M):
        plt.text(LUT[i,0]+0.02,LUT[i,1]+.02,i)

    # grid on; axis((max(axis)+0.1/B)*[-1 1 -1 1]); axis square;
    plt.xlabel('In-Phase')
    plt.ylabel('Quadrature')
    plt.title('Constellation Diagram');


# Unique word (randomly generated)
uw = np.array([162,29,92,47,16,112,63,234,50,7,15,211,109,124,
                   239,255,243,134,119,40,134,158,182,0,101,
                   62,176,152,228,36])
uw_len = uw.size
uw = uw.reshape(uw_len,)

uwsym = LUT[uw,:]

fname = 'kitten1_8bit.dat'

fid= open(fname,'rb')
twoints1 = fid.read(struct.calcsize(packints))
twoints = struct.unpack(packints,twoints1)
rows = twoints[0];
cols = twoints[1];
image = fid.read(rows*cols)
fid.close()
image = np.frombuffer(image,dtype='uint8')
x = image.reshape(cols,rows).T

# Build the list of four possible UW rotations
angles = 2*math.pi*np.arange(0,4)/4
uwrotsyms = np.zeros((uw_len,2,4));

x = np.vstack((x,np.matlib.repmat(uw.reshape(uw_len,1),1,cols)))
rows = rows + uw_len

x = x.flatten('F')   # column scan
sym_stream = LUT[x,:]
sym_keep = sym_stream;
num_syms = sym_stream.shape[0]

# Generate received signal with a clock frequency offset
EB = 0.7;  # Excess bandwidth
To = (1+t_eps)
if(t_eps == 0):  #  No clock skew
  Lp = 12;
  t = np.arange(-Lp*N,Lp*N+1) /N + 1e-8;  # +1e-8 to avoid divide by zero
  tt = t + T_offset;
  srrc = ((np.sin(math.pi*(1-EB)*tt)+ 4*EB*tt * np.cos(math.pi*(1+EB)*tt))
     /((math.pi*tt)*(1-(4*EB*tt)**2)))
  srrc = srrc/math.sqrt(N);
  Isu = np.zeros((num_syms*N,1))
  Isu[range(0,num_syms*N,N)] = sym_stream[:,0].reshape(num_syms,1)
  Qsu = np.zeros((num_syms*N,1))
  Qsu[range(0,num_syms*N,N)] = sym_stream[:,1].reshape(num_syms,1)
  I = np.convolve(srrc,Isu.reshape((N*num_syms,)));
  Q = np.convolve(srrc,Qsu.reshape((N*num_syms,)));

f = open("sim4_2024", "r")
read_data = np.fromstring(f.read(), dtype=float, sep='\n')
r = read_data

print((len(r)))
n = np.arange(I.size)
cos_smpl_rate = N/Omega0

I_r = []
for i in range(0, len(r)):
    I_r.append(r[i]*np.sqrt(2)*np.cos(Omega0*n[i]))

Q_r = []
for i in range(0, len(r)):
    Q_r.append(-1*np.sqrt(2)*r[i]*np.sin(Omega0*n[i]))

# plt.figure(13); plt.clf()
# plt.plot(r_t)

plt.figure(7); plt.clf()
plt.plot(I[0:100])

plt.figure(8); plt.clf()
plt.plot(Q[0:100])
plt.show()

x = np.convolve(I_r,srrc) # the matched filter

y = np.convolve(Q_r,srrc) # the matched filter

x_deriv, x_delay = deriv(L, x, T, TwoPiTs)
y_deriv, y_delay = deriv(L, y, T, TwoPiTs)

plt.figure(44); plt.clf()
plt.plot(x[:int(len(x)/100)])
plt.plot(x_delay[:int(len(x_delay)/100)])

plt.figure(55); plt.clf()
plt.plot(y[:int(len(y)/100)])
plt.plot(y_delay[:int(len(y_delay)/100)])

x_deriv = x_deriv[L:]
x_delay = x_delay[L:]
y_deriv = y_deriv[L:]
y_delay = y_delay[L:]

D = 1
x_deriv = x_deriv[1::D]
x_delay = x_delay[1::D]
y_deriv = y_deriv[1::D]
y_delay = y_delay[1::D]

print("y is ", y)
print("y_delay is ", y_delay)
print("y_deriv is ", y_deriv)

print("x is ", x)
print("x_delay is ", x_delay)
print("x_deriv is ", x_deriv)

plt.figure(4); plt.clf()
plt.plot(x_deriv[:(2*4*Lp+1+5) + N*20])
plt.plot(x_delay[:(2*4*Lp+1+5) + N*20])
plt.title("Real Differentated Signal vs Delayed Signal")
plt.legend(["Differentated Signal", "Delayed Signal"], loc="lower right")
plt.grid()

plt.figure(5); plt.clf()
plt.plot(y_deriv[:(2*4*Lp+1+5) + N*20])
plt.plot(y_delay[:(2*4*Lp+1+5) + N*20])
plt.title("Imag Differentated Signal vs Delayed Signal")
plt.legend(["Differentated Signal", "Delayed Signal"], loc="lower right")
plt.grid()

plt.figure(6); plt.clf()
plt.plot(x[:(2*4*Lp+1+5) + N*20])

plt.figure(7); plt.clf()
plt.plot(y[:(2*4*Lp+1+5) + N*20])


plt.figure(12); plt.clf()
plt.plot(x_delay[:(2*4*Lp+1) + N*20])
for i in range(20):
    plt.plot(
        np.array([2*4*Lp + i*N, 2*4*Lp + i*N]),
        np.array([-2,2]),color='gray',linewidth=0.25)

plt.figure(19); plt.clf()
plt.plot(y_delay[:(4*2*Lp+1) + N*20])
for i in range(20):
    plt.plot(
        np.array([2*4*Lp + i*N, 2*4*Lp + i*N]),
        np.array([-2,2]),color='gray',linewidth=0.25)
plt.show()



      
# first peak at 2*Lp, then every N samples after that
offset = (2*4*Lp - np.floor(N/2)).astype(int)
# ˆ ˆ
# 1st correlation |
# peak |
# move to center
Nsymtoss = 2*np.ceil(Lp/N) # throw away symbols at the end
nc = (np.floor((len(x) - offset - Nsymtoss*N)/N)).astype(int) # number of points of signal to plot
xreshape = x[offset:offset + nc*N].reshape(nc,N)

Nsymtoss = 2*np.ceil(Lp/N) # throw away symbols at the end
nc = (np.floor((len(y) - offset - Nsymtoss*N)/N)).astype(int) # number of points of signal to plot
yreshape = y[offset:offset + nc*N].reshape(nc,N)

first_sample = 2*4*Lp
num_syms = int((len(x)-4*4*Lp)/(N))

#slice
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

print(first_sample)

for n in range(first_sample-1, int(num_syms*N/D)+first_sample):
    temp = NC01 - W1
    if temp <= 0:
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
                shortest_dist = np.sqrt((LUT[k][0]-xi)**2 + (LUT[k][1]-yi)**2)
                shat = k
            elif (shortest_dist > np.sqrt((LUT[k][0]-xi)**2 + (LUT[k][1]-yi)**2)):
                shortest_dist = np.sqrt((LUT[k][0]-xi)**2 + (LUT[k][1]-yi)**2)
                shat = k
        slice_out.append(shat)

        #compute e
        e = ((np.sign(xi) * xdi)) + ((np.sign(yi) * ydi))
        # e_cubic = ((np.sign(xi_cubic) * xdi_cubic)) + ((np.sign(yi_cubic) * ydi_cubic))

        e_list.append(e)

        if (cnt < 50):
            print("strobe is high for sample ", n , " xi is ", xi, " xdi is ",xdi, " yi is ", yi, " ydi is ", ydi, " mu is ", mu, " error is ", e)
    
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

    if (n < 300):
        print("on sample ", n, " W1 is ", W1, " NC01 is ", NC01, " loop_fil_o is ", v)

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

print(len(slice_out))
plt.figure(14)
plt.scatter(storred_x, storred_y,c = 'blue')
plt.grid()
plt.show()

data_out = np.array(slice_out)

uw_test0 = np.array([162, 29, 92, 47, 16, 112, 63, 234, 50, 7, 15, 211, 109, 124, 239, 255, 243, 134, 119, 40, 134, 158, 182, 0, 101, 62, 176, 152, 228, 36])
uw_test1 = np.array([162, 29, 92, 47, 16, 112, 63, 234, 50, 7, 15, 211, 109, 124, 239, 255, 243, 134, 119, 40, 134, 158, 182, 0, 101, 62, 176, 152, 228, 36])
uw_test2 = np.array([162, 29, 92, 47, 16, 112, 63, 234, 50, 7, 15, 211, 109, 124, 239, 255, 243, 134, 119, 40, 134, 158, 182, 0, 101, 62, 176, 152, 228, 36])
uw_test3 = np.array([162, 29, 92, 47, 16, 112, 63, 234, 50, 7, 15, 211, 109, 124, 239, 255, 243, 134, 119, 40, 134, 158, 182, 0, 101, 62, 176, 152, 228, 36])

uw_a0_list = []
uw_a1_list = []

for i in range(0,len(uw_test0)):
    uw_a0_list.append(LUT[uw_test0[i]][0])
    uw_a1_list.append(LUT[uw_test0[i]][1])

uw_a0 = np.array(uw_a0_list)
uw_a1 = np.array(uw_a1_list)

#90 degree offset
for i in range(0,len(uw_a0)):
    tempx = uw_a0[i]
    tempy = uw_a1[i]
    uw_a0[i] = tempy
    uw_a1[i] = -tempx

uw_list = []
for i in range(0, len(uw_a0)):
    for k in range(0, M):
        if k == 0:
            shortest_dist = np.sqrt((LUT[k][0]-uw_a0[i])**2 + (LUT[k][1]-uw_a1[i])**2)
            shat = k
        elif (shortest_dist > np.sqrt((LUT[k][0]-uw_a0[i])**2 + (LUT[k][1]-uw_a1[i])**2)):
            shortest_dist = np.sqrt((LUT[k][0]-uw_a0[i])**2 + (LUT[k][1]-uw_a1[i])**2)
            shat = k
    uw_list.append(shat)

uw_test1 = np.array(uw_list)

#180 degree offset
for i in range(0,len(uw_a0)):
    tempx = uw_a0[i]
    tempy = uw_a1[i]
    uw_a0[i] = -tempx
    uw_a1[i] = -tempy

uw_list = []
for i in range(0, len(uw_a0)):
    for k in range(0, M):
        if k == 0:
            shortest_dist = np.sqrt((LUT[k][0]-uw_a0[i])**2 + (LUT[k][1]-uw_a1[i])**2)
            shat = k
        elif (shortest_dist > np.sqrt((LUT[k][0]-uw_a0[i])**2 + (LUT[k][1]-uw_a1[i])**2)):
            shortest_dist = np.sqrt((LUT[k][0]-uw_a0[i])**2 + (LUT[k][1]-uw_a1[i])**2)
            shat = k
    uw_list.append(shat)

uw_test2 = np.array(uw_list)

#270 degree offset
for i in range(0,len(uw_a0)):
    tempx = uw_a0[i]
    tempy = uw_a1[i]
    uw_a0[i] = -tempy
    uw_a1[i] = tempx

uw_list = []
for i in range(0, len(uw_a0)):
    for k in range(0, M):
        if k == 0:
            shortest_dist = np.sqrt((LUT[k][0]-uw_a0[i])**2 + (LUT[k][1]-uw_a1[i])**2)
            shat = k
        elif (shortest_dist > np.sqrt((LUT[k][0]-uw_a0[i])**2 + (LUT[k][1]-uw_a1[i])**2)):
            shortest_dist = np.sqrt((LUT[k][0]-uw_a0[i])**2 + (LUT[k][1]-uw_a1[i])**2)
            shat = k
    uw_list.append(shat)

uw_test3 = np.array(uw_list)
# print(uw_test0)
# print(uw_test1)
print(uw_test2)
# print(uw_test3)
#find the number of columns/rows (have the number of pixels, # of UW is # of col)
#rows = pixels/cols
lock_offset = 0
cols = 0
shift = np.zeros(30)
for i in range(len(data_out)):
    shift[:29] = shift[1:]
    shift[29] = data_out[i]
    if np.all(shift == uw_test1):
        cols += 1
        lock_offset = 90
    elif np.all(shift == uw_test2):
        cols += 1
        lock_offset = 180
    elif np.all(shift == uw_test3):
        cols += 1
        lock_offset = 270
    elif np.all(shift == uw_test0):
        cols += 1
        lock_offset = 0

if(cols == 0):
   exit()

print(lock_offset)
print(data_out)
rows = 0
while rows == 0:
    if (len(data_out)/cols)%1 == 0:
        rows = int(len(data_out)/cols)
    else:
        cols += 1

a0hat = []
a1hat = []
#TODO: rotate the received values by the offset
if lock_offset != 0:
    for i in range(0, len(data_out)):
        a0hat.append(LUT[data_out[i]][0])
        a1hat.append(LUT[data_out[i]][1])
    for i in range(0, len(a0hat)):
        if lock_offset == 90:
            tempx = a0hat[i]
            tempy = a1hat[i]
            a0hat[i] = tempy
            a1hat[i] = -tempx
        #add in cases for other offsets

    slice_out = []
    for j in range(0, len(a0hat)):
        for k in range(0, M):
            if k == 0:
                shortest_dist = np.sqrt((LUT[k][0]-a0hat[j])**2 + (LUT[k][1]-a1hat[j])**2)
                shat = k
            elif (shortest_dist > np.sqrt((LUT[k][0]-a0hat[j])**2 + (LUT[k][1]-a1hat[j])**2)):
                shortest_dist = np.sqrt((LUT[k][0]-a0hat[j])**2 + (LUT[k][1]-a1hat[j])**2)
                shat = k
        slice_out.append(shat)
    data_out = np.array(slice_out)

print(cols)
data_out = data_out.reshape(cols,rows).T
plt.figure(3)
plt.imshow(255-data_out,cmap=plt.get_cmap('Greys'))
plt.title('Recovered image with UW')

for i in range(30):
    data_out = np.delete(data_out, rows-1, 0)
    rows -= 1


plt.figure(4)
plt.imshow(255-data_out,cmap=plt.get_cmap('Greys'))
plt.title('Recovered image without UW')
plt.show()