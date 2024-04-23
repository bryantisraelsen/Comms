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

fname = 'kitten1_8bit.dat'

# Select modulation type
# Use 8 bits per symbol and 256 square QAM
B = 8;            # Bits per symbol (B should be even: 8, 6, 4, 2)
# B = 4;
bits2index = 2**np.arange(B-1,-1,-1)
M = 2 ** B       # Number of symbols in the constellation
Mroot = math.floor(2**(B/2))
a = np.reshape(np.arange(-Mroot+1,Mroot,2),(2*B,1))
b = np.ones((Mroot,1))

f = open("sim1_2024", "r")
read_data = f.read()

# Load and display the image
print('fname=',fname)

fid= open(fname,'rb')
twoints1 = fid.read(struct.calcsize(packints))
twoints = struct.unpack(packints,twoints1)
rows = twoints[0];
cols = twoints[1];
image = fid.read(rows*cols)
fid.close()
breakpoint()
image = np.frombuffer(image,dtype='uint8')
x = image.reshape(cols,rows).T

plt.figure(2)
plt.imshow(255-x,cmap=plt.get_cmap('Greys'))
plt.title('Original image')
plt.show()

print('rows=',rows,' cols= ',cols,' pixels=',rows*cols)

uw = np.array([162,29,92,47,16,112,63,234,50,7,15,211,109,124,
                   239,255,243,134,119,40,134,158,182,0,101,
                   62,176,152,228,36])
uw_len = uw.size
uw = uw.reshape(uw_len,)

LUT = np.hstack((np.kron(a,b), np.kron(b,a)))

uwsym = LUT[uw,:]

x = np.vstack((x,np.matlib.repmat(uw.reshape(uw_len,1),1,cols)))
rows = rows + uw_len
plt.figure(3)
plt.imshow(255-x,cmap=plt.get_cmap('Greys'))
plt.title('With UW')
plt.show()