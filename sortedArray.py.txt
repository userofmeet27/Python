import numpy as np
m= np.array([10, 7, 14, 3, 8, 5, 12, 9])
mean = np.sum(m) / len(m)
sorted_array = np.sort(m)
print(sorted_array)
if len(sorted_array) % 2 == 0:
    middle1 = sorted_array[len(sorted_array) // 2 - 1]
    middle2 = sorted_array[len(sorted_array) // 2]
    median = (middle1 + middle2) / 2
else:
    median = sorted_array[len(sorted_array) // 2]
min_value = sorted_array[0]
print("Array:",m)
print("Mean:",mean)
print("Median:",median)
print("Minimum:",min_value)
