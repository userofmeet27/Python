import numpy as np
a = 10
count = 0
rn = np.random.randint(1, 3, size=a)
print("The outcomes of the coin are:", rn)
for outcome in rn:
    if outcome == 1:
        count += 1
print("Probability of head :", count/10)
