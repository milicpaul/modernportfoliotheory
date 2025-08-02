import numpy as np
m1 = 526.8
m2 = 978.25
s = 0
delta = 0
a = 0
b = 0
for i in range(58, 86):
    a = np.round((i - 58 + 1) * m1 * 12, 2)
    b = np.round((i - 65 + 1) * m2 * 12, 2)
    if i < 65:
        print(f"{i} {a} {0:<5}")
    else:
        print(f"{i} {a} {b:<5}")
    s = m1 + s

lpp1 = 555452.2
lpp2 = 138523

lpp =  round(lpp1 + lpp2, 2)

r1 = 4020

r =  (4020*(lpp1+lpp2))/lpp1
print(r + (r*0.025))

