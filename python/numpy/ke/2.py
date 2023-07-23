import numpy as np
import time
arr = np.random.rand(100000)
start_time = time.time()
arr_squared = arr ** 2
print("Numpy 平方运算时间：", (time.time() - start_time), "秒")
start_time = time.time()
arr_squared = []
for i in range(len(arr)):
    arr_squared.append(arr[i] ** 2 )
print("循环平方运算时间：", time.time() - start_time, "秒")