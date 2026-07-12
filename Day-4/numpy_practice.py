import numpy as np

# 1. Create a 1D array
arr_1d = np.array([10, 20, 30, 40, 50])
print("1D array:", arr_1d)

# 2. Create a 2D array (like a small grid of numbers)
arr_2d = np.array([[1, 2, 3], [4, 5, 6]])
print("\n2D array:")
print(arr_2d)

# 3. Arithmetic operations on arrays (applies to every element at once)
print("\nArray + 5:", arr_1d + 5)
print("Array * 2:", arr_1d * 2)
print("Array1D + Array1D:", arr_1d + arr_1d)

# 4. Max, min, mean, sum
print("\nMax:", arr_1d.max())
print("Min:", arr_1d.min())
print("Mean:", arr_1d.mean())
print("Sum:", arr_1d.sum())

# 5. Reshape an array into different dimensions
arr_range = np.arange(1, 13)  # numbers 1 to 12
print("\nOriginal array:", arr_range)

reshaped = arr_range.reshape(3, 4)  # 3 rows, 4 columns
print("Reshaped into 3x4:")
print(reshaped)

reshaped2 = arr_range.reshape(4, 3)  # 4 rows, 3 columns
print("Reshaped into 4x3:")
print(reshaped2)

# 6. Slicing and indexing
print("\nFirst element of 1D array:", arr_1d[0])
print("Last element of 1D array:", arr_1d[-1])
print("Slice (index 1 to 3):", arr_1d[1:4])

print("\nElement at row 0, col 2 of 2D array:", arr_2d[0, 2])
print("First row of 2D array:", arr_2d[0])
print("First column of 2D array:", arr_2d[:, 0])

# 7. Mathematical functions
print("\nSquare root of each element:", np.sqrt(arr_1d))
print("Square of each element:", np.square(arr_1d))
print("Absolute values:", np.abs(np.array([-10, 20, -30])))
print("Log of each element:", np.log(arr_1d))