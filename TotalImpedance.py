import math
R = 250  
L1 = 1*10**-3
L2 =  1*10**-3
C1 = 55*10**-6 
C2 = 555*10**-6 
f = 50 
ZL1 = 1 * 2 * math.pi * f * L1
ZC1 = 1 / (1 * 2 * math.pi * f * C1)
ZL2 = 1 * 2 * math.pi * f * L2
ZC2 = 1 / (1 * 2 * math.pi * f * C2)
Zbranch1 = 1 / (1 / ZL1 + 1 / ZC1)
Zbranch2 = 1 / (1 / ZL2 + 1 / ZC2)
Ztotal = R + 1 / (1 / Zbranch1 + 1 / Zbranch2)
print(f"Total impedance of the circuit: {abs(Ztotal):.2f} ohms")