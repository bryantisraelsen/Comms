import math
import random
import numpy as np
from numpy.random import rand
import matplotlib.pyplot as plt
from scipy import special as sp

def qfunc(x):
    return 0.5-0.5*sp.erf(x/np.sqrt(2))

a = 1
sqrt2div2 = np.sqrt(2)/2
LUT1 = np.array([1, 1, -1, -1, 3, 0, -3, 0])*a # LUT for x axis
LUT2 = np.array([-1, 1, -1, 1, 0, -3, 0, 3])*a # LUT for y axis
#                 0  1   2  3  4   5  6  7
distance0 = 2*a
distance1 = np.sqrt((2)**2+(1)**2)
print("distance0 is ", distance0)
print("distance1 is ", distance1)
M = len(LUT1)
LOG2M = 3
Es = 5.5 #average symbol energy
Eb = Es/LOG2M #average bit enery
startSNRdB = 0
endSNRdB = 13
SNRdBstep = 1
Psymerrorlist = []
Pbiterrorlist = []
theoretical_sym = []
maxsymbolerrorcount = 20

for SNRdb in range(startSNRdB,endSNRdB,SNRdBstep):
    print("SNRdb is ", SNRdb)
    SNR = 10**(SNRdb/10) #SNRdb = 10Log10(SNR)
    N0 = Eb/SNR
    sigmaSquare = N0/2
    sigma = np.sqrt(sigmaSquare)
    symbolerrorcount = 0
    numsymbols = 0
    numbits = 0
    biterrorcount = 0
    while(symbolerrorcount < maxsymbolerrorcount):
        numsymbols += 1
        numbits += LOG2M

        # transmitter
        bits = (rand(LOG2M)> 0.5).astype(int) # generate random bits {0,1}
        for i in range(0, int(len(bits)/LOG2M)):   # S/P converter (makes it log_2_M size)
            log2_bit_list = []
            res = 0
            for j in range(0, LOG2M):
                log2_bit_list.append(bits[LOG2M*i+j])
            for ele in log2_bit_list:
                res = (res << 1) | ele
            sent_symb = res
        s_x = LUT1[sent_symb]
        s_y = LUT2[sent_symb]

        # channel
        r_x = s_x + np.random.normal(0, sigma) #generate noise n
        r_y = s_y + np.random.normal(0, sigma)

        shortest_dist = 500
        # receiver
        for i in range(0, M):
            if i == 0:
                shortest_dist = np.sqrt((LUT1[i]-r_x)**2 + (LUT2[i]-r_y)**2)
                shat = i
            elif (shortest_dist > np.sqrt((LUT1[i]-r_x)**2 + (LUT2[i]-r_y)**2)):
                shortest_dist = np.sqrt((LUT1[i]-r_x)**2 + (LUT2[i]-r_y)**2)
                shat = i
        # print("shat is", shat, "r_x is ", r_x, "r_y is ", r_y)
        
        if(shat != sent_symb):
            symbolerrorcount += 1
            xored = shat ^ sent_symb
            bitsoff = 0
            for i in range(0, LOG2M):
                if (((xored >> i) % 2) == 1):
                    bitsoff += 1
            biterrorcount += bitsoff
            print("***symbolerrorcnt is ", symbolerrorcount)

    Psymerror = symbolerrorcount/numsymbols
    Pbiterror = biterrorcount/numbits
    Psymerrorlist.append(Psymerror) #Save Psymerror in an array for plotting
    Pbiterrorlist.append(Pbiterror) #Save Pbiterror in an array for plotting
    theoretical_sym.append(2*qfunc(distance1/(2*sigma))+qfunc(a/sigma))

plt.figure(1); plt.clf()
plt.semilogy(range(startSNRdB,endSNRdB,SNRdBstep), Pbiterrorlist)
plt.semilogy(range(startSNRdB,endSNRdB,SNRdBstep), Psymerrorlist)
plt.semilogy(range(startSNRdB,endSNRdB,SNRdBstep), theoretical_sym)
plt.legend(["Bit Error", "Symbol Error", "Theoretical Symbol"], loc="upper right")
plt.xlabel("Eb/N0  (dB)")
plt.ylabel("P error")
plt.show()
