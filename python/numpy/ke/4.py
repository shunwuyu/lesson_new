import numpy as np
arr = np.array([[1,2,3], [4, 5, 6]])
print('原始数据类型', arr.dtype)
arr = arr.astype('float')
print('强制转化后数据类型', arr.dtype)
# print(arr.shape, arr.ndim, arr.size, arr.dtype)