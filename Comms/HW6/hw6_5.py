import math
import random
import numpy as np
from numpy.random import rand
import matplotlib.pyplot as plt

#Constellation
LOG2_M = 2
a = 1 # PAM amplitude
LUT1 = np.array([-1,-1,1,1])*a # LUT for x-array
LUT2 = np.array([-1,1,1,-1])*a

Nsym = 300 # number of symbols
bits = []
index_bits = []
bits = (rand(Nsym*LOG2_M)> 0.5).astype(int) # generate random bits {0,1}
for i in range(0, int(len(bits)/LOG2_M)):   # S/P converter (makes it log_2_M size)
    log2_bit_list = []
    res = 0
    for j in range(0, LOG2_M):
        log2_bit_list.append(bits[LOG2_M*i+j])
    for ele in log2_bit_list:
        res = (res << 1) | ele
    index_bits.append(res)

# print(index_bits)

#encoder
initsym = 0
sent_sym = []
sent_sym.append(initsym)
prev_sym = initsym
for i in range(0, len(index_bits)):
    curr_sym = index_bits[i]
    send_sym = prev_sym + curr_sym
    if send_sym > 3:
        send_sym -= 4
    # print("send_sym for iteration ", i, " is ", send_sym, " because prev sym was ", prev_sym, " and curr sym is ", curr_sym)
    sent_sym.append(send_sym)
    prev_sym = send_sym

sent_real = LUT1[sent_sym]
sent_img = LUT2[sent_sym]

#phase offset from transmission
for error in range(0,4):
    received_real = []
    received_img = []
    received_sym = []
    received = 0
    for i in range(0, len(sent_sym)):
        if (sent_sym[i] + error > 3):
            received = sent_sym[i] + error - 4
            received_real.append(LUT1[received])
            received_img.append(LUT2[received])
        else:
            received = sent_sym[i] + error
            received_real.append(LUT1[received])
            received_img.append(LUT2[received])
        received_sym.append(received)

            
    #decoder

    #slice
    slice_out = []
    for i in range(0, len(received_sym)):
        if received_real[i] == -1 and received_img[i] == -1:
            slice_out.append(0)
        elif received_real[i] == -1 and received_img[i] == 1:
            slice_out.append(1)
        elif received_real[i] == 1 and received_img[i] == 1:
            slice_out.append(2)
        elif received_real[i] == 1 and received_img[i] == -1:
            slice_out.append(3)

    data_out = []
    prev_rec = slice_out[0]
    for i in range(1, len(slice_out)):
        rec_sym = slice_out[i]
        if rec_sym-prev_rec < 0:
            data_out.append(rec_sym-prev_rec+4)
        else:
            data_out.append(rec_sym-prev_rec)
        prev_rec = rec_sym

    angle_err = error*90
    if index_bits != data_out:
        print("Error!! When error was ", error, " the data received was not what was sent!")
        print(index_bits)
        print(data_out)
    else:
        print("The data sent is the same as the data sent where the phase offset of ",  angle_err, " degrees. Good work!")
        print("If you have trust issues you can verify that they are the same below")
        print("Here are the sent sybmols: ")
        print(index_bits)
        print("Here are the received sybmols: ")
        print(data_out)
        print("\n\n\n")