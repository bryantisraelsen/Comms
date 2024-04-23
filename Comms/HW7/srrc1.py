from math import sin, cos, sqrt, pi

def srrc1(alpha,N,Lp,Ts) -> list:
    x = []
    n = list(range((-Lp),Lp-1))
    if N %2 == 1:
        for i in range(len(n)):
            #if the denominator doesn't go to zero calculate this way
            if ((pi*n[i]/N)*(1-(4*alpha*n[i]/N)**2)) != 0:
                num = (sin(pi*(1-alpha)*n[i]/N)+(4*alpha*n[i]/N)*(cos(pi*(1+alpha)*n[i]/N)))
                den = ((pi*n[i]/N)*(1-(4*alpha*n[i]/N)**2))
                x.append((1/sqrt(N))*num/den)
            #if the denomiator goes to zero use L'Hopitals rule to find the value
            else:
                num = (-4*pi*alpha*(alpha+1)*n[i]*sin((pi*(alpha+1)*n[i]/N))/(N**2) + (4*alpha*cos(pi*(alpha+1)*n[i]/N))/N + (pi*(1-alpha)*cos(pi*(1-alpha)*n[i]/N))/N)
                den = -1*pi*(12*n[i]**2-N**2)/N**3
                # print((1/sqrt(N))*num/den)
                x.append((1/sqrt(N))*num/den)
    else:
        n = list(range((-Lp),Lp+2))
        N += 1
        for i in range(len(n)):
            #if the denominator doesn't go to zero calculate this way
            if ((pi*n[i]/N)*(1-(4*alpha*n[i]/N)**2)) != 0:
                num = (sin(pi*(1-alpha)*n[i]/N)+(4*alpha*n[i]/N)*(cos(pi*(1+alpha)*n[i]/N)))
                den = ((pi*n[i]/N)*(1-(4*alpha*n[i]/N)**2))
                x.append((1/sqrt(N))*num/den)
            #if the denomiator goes to zero use L'Hopitals rule to find the value
            # else:
            #     num = (-4*pi*alpha*(alpha+1)*n[i]*sin((pi*(alpha+1)*n[i]/N))/(N**2) + (4*alpha*cos(pi*(alpha+1)*n[i]/N))/N + (pi*(1-alpha)*cos(pi*(1-alpha)*n[i]/N))/N)
            #     den = -1*pi*(12*n[i]**2-N**2)/N**3
            #     # print((1/sqrt(N))*num/den)
            #     x.append((1/sqrt(N))*num/den)

    sum = 0

    for i in range(len(x)):
        sum = sum + x[i]**2

    print("Sum is", sum)
    return x