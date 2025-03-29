import numpy as np

# make a  16x16 blank array
arr = np.zeros((16, 16))
for i in range(16):
    for j in range(16):
        print(4*i + 8, 4*j + 8)