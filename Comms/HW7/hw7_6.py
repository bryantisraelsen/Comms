import numpy as np
import matplotlib.pyplot as plt

true_function_y = []
true_function_x = []
sampled_points_y = []
sampled_points_x = []
T = 1
F0T = 0.15
F0 = F0T/T
cnt = 0
for i in np.arange(T, 2*T+0.001, 0.001):
    if (cnt % 100 == 0) :
        sampled_points_y.append(np.sin(np.pi*2*F0*i))
        sampled_points_x.append(i)
    true_function_y.append(np.sin(np.pi*2*F0*i))
    true_function_x.append(i)
    cnt += 1

sampled_points1_y = []
sampled_points1_x = []
true_function_x1 = []
true_function_y1 = []
for i in np.arange(0, 4*T, T):
    sampled_points1_x.append(i)
    sampled_points1_y.append(np.sin(np.pi*2*F0*i))

for i in np.arange(0, 3*T+0.001, 0.001):
    true_function_x1.append(i)
    true_function_y1.append(np.sin(np.pi*2*F0*i))

plt.figure(0)
plt.scatter(sampled_points1_x,sampled_points1_y, c = 'orange')
plt.plot(true_function_x1, true_function_y1)
plt.title("Linear Interpolator")
plt.grid()

x_m_plus_u = []
x_m_plus_u_xaxis = []

for myu in np.arange(0,1.1,0.1):
    x_m_plus_u.append((sampled_points1_y[2] - sampled_points1_y[1])*myu + sampled_points1_y[1])
    x_m_plus_u_xaxis.append(myu+T)

plt.scatter(x_m_plus_u_xaxis,x_m_plus_u, c = 'green')

x_m_plus_u = []
x_m_plus_u_xaxis = []
for myu in np.arange(0,1.1,0.1):
    term1 = sampled_points1_y[3]*(myu**3/6-myu/6)
    term2 = sampled_points1_y[2]*(myu**3/2-myu**2/2-myu)
    term3 = sampled_points1_y[1]*(myu**3/2-myu**2-myu/2+1)
    term4 = sampled_points1_y[0]*(myu**3/6-myu**2/2-myu/3)
    last_term = term1-term2+term3-term4

    x_m_plus_u.append(last_term)
    x_m_plus_u_xaxis.append(myu+T)

plt.figure(1)
plt.scatter(sampled_points1_x,sampled_points1_y, c = 'orange')
plt.plot(true_function_x1, true_function_y1)
plt.title("Cubic interpolator")
plt.scatter(x_m_plus_u_xaxis,x_m_plus_u, c = 'black')
plt.grid()
plt.show()
