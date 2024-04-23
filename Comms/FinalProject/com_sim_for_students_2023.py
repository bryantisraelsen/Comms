# Digital communications simulation
#  Derived from Matlab code by Jake Gunther
# Date  :  April 1, 2024
# Class :  ECE 5660 (Utah State University)

import numpy as np
import numpy.matlib
import math
import matplotlib.pyplot as plt
import struct   # i
packints = 'ii'

# Set parameters
f_eps        = 0.0 # Carrier frequency offset percentage (0 = no offset)
Phase_Offset = 0.0 # Carrier phase offset (0 = no offset)
t_eps        = 0.0 # Clock freqency offset percentage (0 = no offset)
T_offset     = 0.0 # Clock phase offset (0 = no offset)
Ts = 1             # Symbol period
N = 4              # Samples per symbol period
TwoPiTs = np.pi*2*Ts

fname = 'kitten1_8bit.dat'
file_path = "kitten_pic.dat"

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

# Scale the constellation to have unit energy
Enorm = np.sum(LUT ** 2) / M;
LUT = LUT/math.sqrt(Enorm);

Eave = 1;
Eb = Eave/B;
EbN0dB = 30; # SNR in dB
N0 = Eb*10**(-EbN0dB/10);
nstd = math.sqrt(N0/2); # Noise standard deviation
# Note nstd is set to 0 below so there is no noise

if 1:
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
# print(uwsym)


# Build the list of four possible UW rotations
angles = 2*math.pi*np.arange(0,4)/4
uwrotsyms = np.zeros((uw_len,2,4));

for i in range(angles.size):
  C = math.cos(angles[i])
  S = -math.sin(angles[i])
  G = np.array([[C, -S],[S, C]])
  uwrot = uwsym @ G;  # Rotate the UW symbols
  uwrotsyms[:,:,i] = uwrot; # Save the rotated version

# Load and display the image
print('fname=',fname)

fid= open(fname,'rb')
twoints1 = fid.read(struct.calcsize(packints))
twoints = struct.unpack(packints,twoints1)
rows = twoints[0];
cols = twoints[1];
image = fid.read(rows*cols)
fid.close()
image = np.frombuffer(image,dtype='uint8')
x = image.reshape(cols,rows).T

plt.figure(2)
plt.imshow(255-x,cmap=plt.get_cmap('Greys'))
plt.title('Original image')

print('rows=',rows,' cols= ',cols,' pixels=',rows*cols)



# Insert the unique word at the end of each column
x = np.vstack((x,np.matlib.repmat(uw.reshape(uw_len,1),1,cols)))
rows = rows + uw_len
plt.figure(3)
plt.imshow(255-x,cmap=plt.get_cmap('Greys'))
plt.title('With UW')
plt.show()

print('rows=',rows,' cols= ',cols,' pixels=',rows*cols)

x = x.flatten('F')   # column scan
sym_stream = LUT[x,:]
sym_keep = sym_stream;
num_syms = sym_stream.shape[0]

# Generate received signal with a clock frequency offset
print('Generating transmitted I/Q waveforms ... ');
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
else: # Implement clock skew
  t = np.arange(0,num_syms*N)/N+1e-8;
  I = np.zeros((1,t.length))  # In-phase pulse train
  Q = np.zeros((1,t.length))  # Quadrature pulse train
  for i in range(num_syms):
    tt = t-i*To + T_offset;
    srrc = ((np.sin(math.pi*(1-EB)*tt)+4*EB*tt*np.cos(math.pi*(1+EB)*tt))
              /((math.pi*tt)*(1-(4*EB*tt)**2)))
    srrc = srrc/math.sqrt(N);
    I = I + srrc*sym_stream[i,0];
    Q = Q + srrc*sym_stream[i,1];
print('done.\n')

# Modulate the pulse trains
print('Modulating I/Q waveforms ... ')
Omega0 = math.pi/2*(1+f_eps)
n = np.arange(I.size)
C =  math.sqrt(2)*np.cos(Omega0*n + Phase_Offset)
S = -math.sqrt(2)*np.sin(Omega0*n + Phase_Offset)
nstd = 0   # for this test, there is no noise
r = I * C + Q * S + nstd*np.random.normal(0,1,I.shape); # Noisy received signal
print(r)

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

# plt.figure(7); plt.clf()
# plt.plot(I_r[0:100])

# plt.figure(8); plt.clf()
# plt.plot(Q_r[0:100])
# plt.show()

x = np.convolve(I_r,srrc) # the matched filter
plt.figure(9); plt.clf()
plt.plot(x)

y = np.convolve(Q_r,srrc) # the matched filter
plt.figure(10); plt.clf()
plt.plot(y)

plt.show()
breakpoint()

plt.figure(12); plt.clf()
plt.plot(x[:(2*4*Lp+1) + N*20])
for i in range(20):
    plt.plot(
        np.array([2*4*Lp + i*N, 2*4*Lp + i*N]),
        np.array([-2,2]),color='gray',linewidth=0.25)
plt.show()

plt.figure(12); plt.clf()
plt.plot(y[:(4*2*Lp+1) + N*20])
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
# plt.figure(13); plt.clf()

# plt.plot(np.arange(-np.floor(N/2), np.floor(N/2)+1), xreshape.T,color='b', linewidth=0.25)

Nsymtoss = 2*np.ceil(Lp/N) # throw away symbols at the end
nc = (np.floor((len(y) - offset - Nsymtoss*N)/N)).astype(int) # number of points of signal to plot
xreshape = y[offset:offset + nc*N].reshape(nc,N)
# plt.figure(14); plt.clf()

# plt.plot(np.arange(-np.floor(N/2), np.floor(N/2)+1), xreshape.T,color='b', linewidth=0.25)
# plt.show()

first_sample = 2*4*Lp
nxt_smpl = first_sample
x_received = []
y_received = []
received_bits = []
#slice (this will change depending on the constellation)
for i in range(num_syms):
    x_received.append(x[nxt_smpl])
    y_received.append(y[nxt_smpl])
    nxt_smpl = nxt_smpl+N

plt.figure(15); plt.clf()
plt.scatter(x_received,y_received)
plt.show()

# print(LUT)

slice_out = []

#slice
for i in range(0, len(x_received)):
    for k in range(0, M):
        # print(LUT[k][0])
        if k == 0:
            shortest_dist = np.sqrt((LUT[k][0]-x_received[i])**2 + (LUT[k][1]-y_received[i])**2)
            shat = k
        elif (shortest_dist > np.sqrt((LUT[k][0]-x_received[i])**2 + (LUT[k][1]-y_received[i])**2)):
            shortest_dist = np.sqrt((LUT[k][0]-x_received[i])**2 + (LUT[k][1]-y_received[i])**2)
            shat = k
    slice_out.append(shat)

print(slice_out)

data_out = np.array(slice_out)


data_out = data_out.reshape(cols,rows).T
# print(data_out)
plt.figure(3)
plt.imshow(255-data_out,cmap=plt.get_cmap('Greys'))
plt.title('Recovered image')
plt.show()