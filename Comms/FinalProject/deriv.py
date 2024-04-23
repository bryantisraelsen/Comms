import numpy as np
import matplotlib.pyplot as plt

def deriv(L,r,T,w0) -> list:
    n = range(-L, L+1)
    x = [] #returned signal
    h_d = [] #deriv transfer function
    delay = [] # delay transfer function
    blackman = np.blackman(2*L+1)
    delay_r = [] # delayed received signal

    for i in range(0,len(n)):
        if n[i] == 0:
            h_d.append(0)
            delay.append(1)
        else:
            h_d.append(1/(T)*((-1)**n[i])/n[i])
            delay.append(0)

    # plt.figure(0); plt.clf()
    # plt.stem(n,h_d)
    # plt.title("impulse response of differentiator")

    for x in range(0,len(h_d)):
        h_d[x] = h_d[x]/w0

    # plt.figure(1); plt.clf()
    # plt.stem(n,delay)
    # plt.title("impulse response of delay")


    h_d_fir = h_d*blackman

    A = np.fft.fft(blackman, 2048) / ((len(blackman))/2)
    mag = np.abs(np.fft.fftshift(A))
    freq = np.linspace(-0.5, 0.5, len(A))
    with np.errstate(divide='ignore', invalid='ignore'):
        response = 20 * np.log10(mag)
    response = np.clip(response, -100, 100)
    # plt.figure(2); plt.clf()
    # plt.plot(freq, response)
    # plt.xlabel("Normalized Frequency (cycles per sample)")
    # plt.ylabel("Magnitude (dB)")

    A = np.fft.fft(h_d_fir, 2049) / (len(h_d_fir)/2)
    mag = np.abs(np.fft.fftshift(A))
    freq = np.linspace(-0.5, 0.5, len(A))
    response = (mag)
    response = np.clip(response, -100, 100)
    # plt.figure(3); plt.clf()
    # plt.plot(freq, response)
    # plt.axvline(-0.3, color = 'red')
    # plt.xlabel("Normalized Frequency (cycles per sample)")
    # plt.ylabel("Magnitude")

    # plt.figure(16); plt.clf()
    angle = np.angle(np.fft.fftshift(A))
    response = (angle)
    response = np.clip(response, -100, 100)
    freq = np.linspace(-0.5, 0.5, len(A))
    # plt.plot(freq[int(len(freq)/2):], response[int(len(response)/2):])
    # plt.xlabel("Normalized Frequency (cycles per sample)")
    # plt.ylabel("Phase")

    A = np.fft.fft(delay, 2049) / (len(delay)/2)
    mag = np.abs(np.fft.fftshift(A))
    freq = np.linspace(-0.5, 0.5, len(A))
    response = (mag)
    response = np.clip(response, -100, 100)
    # plt.figure(4); plt.clf()
    # plt.plot(freq, response)
    # plt.xlabel("Normalized Frequency (cycles per sample)")
    # plt.ylabel("Magnitude")
    # plt.show()

    delay_r = np.convolve(r,delay)
    x = np.convolve(r,h_d_fir)
    return x, delay_r