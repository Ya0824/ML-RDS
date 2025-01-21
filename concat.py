import sys
import numpy as np
	
data = []
	
idx = 0
for fname in sys.argv[1:]:
	arr = np.load(fname)
	arr = arr.reshape((arr.shape[0] * arr.shape[1], arr.shape[2]))
	arr = (arr - np.mean(arr, axis=0, keepdims=True)) / np.std(arr, axis=0, keepdims=True)
	arr = np.concatenate([arr, np.array([np.full(arr.shape[0], int(fname.split("_")[1].split(".")[0]))]).T], axis=1)
	data.append(arr)
	print(idx, arr.shape)
	idx += 1
	
data = np.concatenate(data, axis=0)
print(data.shape)
np.save("concat", data)
