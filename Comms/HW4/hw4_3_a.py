import math
import random
import numpy as np
from numpy.random import rand
import matplotlib.pyplot as plt
from scipy import special as sp

def qfunc(x):
    return 0.5-0.5*sp.erf(x/np.sqrt(2))

a = 1 # PAM amplitude
# LUT1 = np.array([-1,-1,1,1])*a # LUT
# LUT2 = np.array([-1,1,-1,1])*a
LUT1 = np.array([-1,1])*a 
LOG2M = 1
Es = 1 #average symbol energy
Eb = Es/LOG2M #average bit enery
startSNRdB = 0
endSNRdB = 11
SNRdBstep = 1
Psymerrorlist = []
Pbiterrorlist = []
theoretical = []
maxsymbolerrorcount = 20

for SNRdb in range(startSNRdB,endSNRdB,SNRdBstep):
    print("SNRdb is ", SNRdb)
    SNR = 10**(SNRdb/10) #SNRdb = 10Log10(SNR)
    N0 = Eb/SNR
    sigmaSquare = N0/2
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
            curr_sym = res
        s_x = LUT1[curr_sym]
        # s_y = LUT2[curr_sym]

        # channel
        r_x = s_x + np.random.normal(0, np.sqrt(sigmaSquare)) #generate noise n with variance sigmaˆ2 in each coordinate direction
        # r_y = s_y + np.random.normal(0, np.sqrt(sigmaSquare))

        # receiver
        # if (r_x > 0 and r_y > 0):
        #     shat = 3
        # elif (r_x > 0 and r_y < 0):
        #     shat = 2
        # elif (r_x < 0 and r_y > 0):
        #     shat = 1
        # elif (r_x < 0 and r_y < 0):
        #     shat = 0
        if r_x > 0:
            shat = 1
        else:
            shat = 0
        
        if(shat != curr_sym):
            symbolerrorcount += 1
            biterrorcount += LOG2M
            print("***symbolerrorcnt is ", symbolerrorcount)

    Psymerror = symbolerrorcount/numsymbols
    Pbiterror = biterrorcount/numbits
    Psymerrorlist.append(Psymerror) #Save Psymerror in an array for plotting
    Pbiterrorlist.append(Pbiterror) #Save Pbiterror in an array for plotting
    theoretical.append(qfunc(np.sqrt(6*(LOG2M*Eb)/(3*N0))))
print("Exited loop")
plt.figure(1); plt.clf()
plt.semilogy(range(startSNRdB,endSNRdB,SNRdBstep), Pbiterrorlist)
plt.axvline(x=9.6, color = 'r')
plt.semilogy(range(startSNRdB,endSNRdB,SNRdBstep), theoretical)
plt.show()
#semilogy(range(startSNRdB,endSNRdB,SNRdBstep), Pbiterrorlist)
# Plots should be done on a logarithmic y axis
