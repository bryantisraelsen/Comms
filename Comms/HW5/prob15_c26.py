import numpy as np
import matplotlib.pyplot as plt
from scipy import special as sp

Len = 80
A = 1
k0 = 1
kp = A
zeta = 1/np.sqrt(2)
BnT = 0.05

den = zeta+(1/(4*zeta))
bigterm = (BnT/den)

k1top = 4*zeta*bigterm
k1bottom = 1+2*zeta*bigterm+bigterm**2
k1 = k1top/(k1bottom*kp*k0)

k2top = 4*((BnT/den)**2)
k2bottom = 1+2*zeta*bigterm+bigterm**2
k2 = k2top/(k2bottom*k0*kp)

print("k1 is", k1, "\nk2 is", k2)

Omega0 = np.pi/10
theta = np.pi/2
theta_hat = 0.0
theta_err = 0.0
theta_error = []
detector_O = 0.0
z_pres = 0.0
z_past = 0.0
loop_fil_o = 0.0
prev_loop_fil_o = 0.0
angle = 0.0

input_phase = []
error_phase = []
cos_in = []
cos_out = []

input = np.exp(theta)
out = np.exp(theta_hat)

for n in range(0, Len):
    
    #The phase detector
    theta = Omega0*n+39*np.pi/40
    input = np.exp(1j*theta)
    out = np.conjugate(np.exp(1j*theta_hat))
    error_sig = input*out
    theta_err = np.imag(error_sig)

    #for plot2 (The error plot)
    theta_error.append(theta_err)
    #for plot3 (input vs output exponential values)
    input_phase.append(theta)
    error_phase.append(theta_hat)
    #for plot1 (with the sinusoids)
    cos_in.append(np.cos(theta))
    cos_out.append(np.cos(theta_hat))

    detector_O = kp*theta_err
    #detector_O is out of args block

    #z_pres is input for z^-1 block
    #z_past is output of z^-1 block
    z_pres = detector_O*k2+z_past
    loop_fil_o = detector_O*k1+z_pres
    #update z_past
    z_past = z_pres
    #loop filter output is loop_fil_o

    #theta_hat is based off of previous loop_filter output
    theta_hat += k0*prev_loop_fil_o+Omega0
    prev_loop_fil_o = loop_fil_o

    


plt.figure(1); plt.clf()
plt.plot(input_phase, linestyle = 'dashed')
plt.plot(error_phase)
plt.xlim(0,Len)


plt.figure(2); plt.clf()
plt.plot(theta_error)
plt.xlim(0,Len)

plt.figure(3); plt.clf()
plt.plot(cos_in, linestyle = 'dashed')
plt.plot(cos_out)
plt.xlim(0,Len)
plt.show()
