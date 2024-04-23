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

# Scale the constellation to have unit energy
Enorm = np.sum(LUT ** 2) / M;
LUT = LUT/math.sqrt(Enorm);

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

f = open("sim1_2024", "r")
read_data = np.fromstring(f.read(), dtype=float, sep='\n')
np.fromstring('1 2', dtype=int, sep=' ')
r = read_data

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

Nsymtoss = 2*np.ceil(Lp/N) # throw away symbols at the end
nc = (np.floor((len(y) - offset - Nsymtoss*N)/N)).astype(int) # number of points of signal to plot
xreshape = y[offset:offset + nc*N].reshape(nc,N)

first_sample = 2*4*Lp
print(first_sample)

plt.figure(18); plt.clf()
plt.plot(x[0:200])
plt.show()

nxt_smpl = first_sample
x_received = []
y_received = []
received_bits = []
num_syms = int((len(x)-4*4*Lp)/(N))
#slice (this will change depending on the constellation)
for i in range(num_syms):
    x_received.append(x[nxt_smpl])
    y_received.append(y[nxt_smpl])
    nxt_smpl = nxt_smpl+N

plt.figure(15); plt.clf()
plt.scatter(x_received,y_received)
plt.show()

slice_out = []

#slice
for i in range(0, len(x_received)):
    for k in range(0, M):
        if k == 0:
            shortest_dist = np.sqrt((LUT[k][0]-x_received[i])**2 + (LUT[k][1]-y_received[i])**2)
            shat = k
        elif (shortest_dist > np.sqrt((LUT[k][0]-x_received[i])**2 + (LUT[k][1]-y_received[i])**2)):
            shortest_dist = np.sqrt((LUT[k][0]-x_received[i])**2 + (LUT[k][1]-y_received[i])**2)
            shat = k
    slice_out.append(shat)

data_out = np.array(slice_out)
uw_test = np.array([162, 29, 92, 47, 16, 112, 63, 234, 50, 7, 15, 211, 109, 124, 239, 255, 243, 134, 119, 40, 134, 158, 182, 0, 101, 62, 176, 152, 228, 36])

#find the number of columns/rows (have the number of pixels, # of UW is # of col)
#rows = pixels/cols
cols = 0
shift = np.zeros(30)
for i in range(len(data_out)):
   shift[:29] = shift[1:]
   shift[29] = data_out[i]
   if np.all(shift == uw_test):
      cols += 1

if(cols == 0):
   exit()
rows = int(len(data_out)/cols)

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