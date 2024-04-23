import math
import random
import numpy as np
from numpy.random import rand
import matplotlib.pyplot as plt
from scipy import special as sp

def qfunc(x):
    return 0.5-0.5*sp.erf(x/np.sqrt(2))

a = 1 # PAM amplitude
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
maxsymbolerrorcount = 10

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

        # channel
        r_x = s_x + np.random.normal(0, sigma) #generate noise n with variance sigmaˆ2 in each coordinate direction

        # receiver
        if r_x > 0:
            shat = 1
        else:
            shat = 0
        
        if(shat != sent_symb):
            symbolerrorcount += 1
            biterrorcount += LOG2M
            print("***symbolerrorcnt is ", symbolerrorcount)

    Psymerror = symbolerrorcount/numsymbols
    Pbiterror = biterrorcount/numbits
    Psymerrorlist.append(Psymerror) #Save Psymerror in an array for plotting
    Pbiterrorlist.append(Pbiterror) #Save Pbiterror in an array for plotting
    theoretical.append(qfunc(np.sqrt(2*(LOG2M*Eb)/(N0))))
    # theoretical.append(2*qfunc(a/sigma))
print("Exited loop")
plt.figure(1); plt.clf()
plt.semilogy(range(startSNRdB,endSNRdB,SNRdBstep), Pbiterrorlist)
plt.semilogy(range(startSNRdB,endSNRdB,SNRdBstep), theoretical)
plt.axvline(x=9.6, color = 'r')
plt.legend(["Bit Error vs SNR", "Theoretical", "9.6 dB"], loc="upper right")
plt.xlabel("Eb/N0  (dB)")
plt.ylabel("Pb")
plt.show()
