import numpy as np
import numpy.matlib
import math
import matplotlib.pyplot as plt
import struct   # i
packints = 'ii'

fname = 'kitten1_8bit.dat'

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

x = np.vstack((x,np.matlib.repmat(uw.reshape(uw_len,1),1,cols)))
rows = rows + uw_len
plt.figure(3)
plt.imshow(255-x,cmap=plt.get_cmap('Greys'))
plt.title('With UW')
plt.show()