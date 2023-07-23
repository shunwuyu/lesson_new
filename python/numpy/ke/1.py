# fruitPrice = [5, 4, 6, 2]
# for i in range(4):
#     fruitPrice[i] = fruitPrice[i] + 1
# print(fruitPrice)
import numpy as np
fruitPrice = np.array([5, 4, 6, 2])
# // numpy 广播机制  避免了复杂的循环操作
fruitPrice = fruitPrice + 1
print(fruitPrice)