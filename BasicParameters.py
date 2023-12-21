import math
Vpeak = 230 * math.sqrt(2)  
R = 50  
L = 33*10**-3  
f = 50  
w = 2 * math.pi * f
Z = math.sqrt(R**2 + (1 *w* L)**2)
I = Vpeak / Z
P = abs(I)**2 * R
S = I * Vpeak.conjugate()
cos_phi = P / abs(S)
print(f"Current (I): {abs(I):.2f} A")
print(f"Average Power (P): {P:.2f} W")
print(f"Apparent Power (S): {abs(S):.2f} VA")
print(f"Power Factor (PF): {cos_phi:.2f}")
