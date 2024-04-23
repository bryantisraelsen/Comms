import math
import random
import numpy as np
from numpy.random import rand
import matplotlib.pyplot as plt
from scipy import special as sp

def qfunc(x):
    return 0.5-0.5*sp.erf(x/np.sqrt(2))

a = 1 # PAM amplitude
LUT1 = np.array([1,-1,1,-1])*a # LUT
LUT2 = np.array([1,1,-1,-1])*a
M = len(LUT1)
LOG2M = 2
Es = 2*a #average symbol energy
Eb = Es/LOG2M #average bit enery
startSNRdB = 0
endSNRdB = 12
SNRdBstep = 1
Psymerrorlist = []
Pbiterrorlist = []
theoretical_bit = []
theoretical_sym = []
maxsymbolerrorcount = 40

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

        # receiver
        if (r_x > 0 and r_y > 0):
            shat = 0
        elif (r_x < 0 and r_y > 0):
            shat = 1
        elif (r_x > 0 and r_y < 0):
            shat = 2
        elif (r_x < 0 and r_y < 0):
            shat = 3
        
        if(shat != sent_symb):
            # print("r_x is", r_x, "r_y is", r_y, "s_x is", s_x, "s_y is", s_y, "shat is", shat, "sent_symb is", sent_symb)
            symbolerrorcount += 1
            if ((shat == 3 and sent_symb == 0) or (shat == 0 and sent_symb == 3) or (shat == 2 and sent_symb == 1) or (shat == 1 and sent_symb == 2)):
                biterrorcount += LOG2M
            else:
                biterrorcount += 1
            print("***symbolerrorcnt is ", symbolerrorcount)

    Psymerror = symbolerrorcount/numsymbols
    Pbiterror = biterrorcount/numbits
    Psymerrorlist.append(Psymerror) #Save Psymerror in an array for plotting
    Pbiterrorlist.append(Pbiterror) #Save Pbiterror in an array for plotting
    theoretical_bit.append((2/LOG2M)*qfunc(np.sqrt((Eb/N0)*2*(LOG2M)*(np.sin(np.pi/M))**2)))
    theoretical_sym.append(2*qfunc(np.sqrt((Es/N0)*2*np.sin(np.pi/M)**2)))
print("Exited loop")
plt.figure(1); plt.clf()
plt.semilogy(range(startSNRdB,endSNRdB,SNRdBstep), Pbiterrorlist)
plt.semilogy(range(startSNRdB,endSNRdB,SNRdBstep), Psymerrorlist)
plt.semilogy(range(startSNRdB,endSNRdB,SNRdBstep), theoretical_bit)
plt.semilogy(range(startSNRdB,endSNRdB,SNRdBstep), theoretical_sym)
plt.legend(["Bit Error", "Symbol Error", "Theoretical Bit", "Theoretical Symbol"], loc="upper right")
plt.xlabel("Eb/N0  (dB)")
plt.ylabel("P error")
plt.show()
