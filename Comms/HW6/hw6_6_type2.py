import math
import random
import numpy as np
from numpy.random import rand
import matplotlib.pyplot as plt

sym_known = 1

#Constellation
M = 2
# M = 8
# M = 16
a = 1 # PAM amplitude
sqrt2div2 = np.sqrt(2)/2
LUT1 = np.array([-1,1])*a # LUT for x-array
LUT2 = np.array([0,0])*a
# LUT1 = np.array([0, sqrt2div2, sqrt2div2, 1, -1*sqrt2div2, -1, 0, -1*sqrt2div2])*a # LUT for x axis
# LUT2 = np.array([1, sqrt2div2, -1*sqrt2div2, 0, sqrt2div2, 0, -1, -1*sqrt2div2])*a # LUT for y axis
#                0      1           2     3         4      5   6         7
# LUT1 = np.array([-3, -3, -3, -3, -1, -1, -1, -1, 1, 1, 1, 1, 3, 3, 3, 3])*a # LUT for x axis
# LUT2 = np.array([-3, -1, 1, 3, -3, -1, 1, 3, -3, -1, 1, 3, -3, -1, 1, 3])*a # LUT for y axis

xaxis = np.arange(-1*np.pi, np.pi, 0.01)

# print(xaxis)

avg_theta_e = []
sym_ang = []

for i in range(0, M):
    sym_ang.append(np.arctan2(LUT2[i], LUT1[i]))

for i in range(0,len(xaxis)):
    avg_thetae = 0
    for j in range(0,M):
        a0 = LUT1[j]
        a1 = LUT2[j]
        yp = (a0*np.sin(xaxis[i]) + a1*np.cos(xaxis[i]))
        xp = (a0*np.cos(xaxis[i]) - a1*np.sin(xaxis[i]))
        if (sym_known == 1):
            a0hat = a0
            a1hat = a1
        else:
            shortest_dist = 500
            # receiver
            if (xp < 0):
                a0hat = -1
            else:
                a0hat = 1
            a1hat = 0
            if (a0 != a0hat and a1 != a1hat):
                print("different a0hat and/or a1hat then what was sent")
        theta1 = np.arctan2(yp, xp)
        theta2 = np.arctan2(0, a0hat)
        # if i == 315:
        #     print("yp is ", yp, " xp is ", xp)
        #     print("a1hat is ", a1hat, " a0hat is ", a0hat)
        #     print("theta1 is ", theta1, " theta2 is ", theta2)
        # theta1 = yp*a0hat
        # theta2 = xp*a1hat
        thetadiff = theta1-theta2
        avg_thetae += thetadiff
    if i >= 315:
        avg_theta_e.append(avg_thetae/M+np.pi)
    else:
        avg_theta_e.append(avg_thetae/M)

plt.figure(1); plt.clf()
plt.plot(xaxis, avg_theta_e)
plt.show()