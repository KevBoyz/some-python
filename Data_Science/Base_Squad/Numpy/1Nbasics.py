import numpy as np

arr = np.arange(0, 100, 10).reshape(2, 5)
# print(arr, '\n')

rg = np.random.default_rng()
# print(rg.random((4, 3)))

lin = np.linspace(-10, 10, 6)
# print(lin)

zeros = np.zeros(shape=(3, 3, 3, 3))  # 4 dims  -- ones 0 -> 1
# print(zeros)

empty = np.empty((2, 5))
# print(empty)  # None empty values

print(arr[0,:])
print(arr * np.array(2))