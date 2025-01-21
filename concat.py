import os
import numpy as np

# 指定存放 .npy 文件的目录
directory = "."  # 当前目录
# directory = "Weight_files"  # 如果在 Weight_files 目录下，取消注释这行

# 获取所有文件名
all_files = os.listdir(directory)

# 筛选符合 Weights_*.npy 格式的文件
file_list = [f for f in all_files if f.startswith("weights_") and f.endswith(".npy")]

# 按文件名中的数字排序
file_list = sorted(file_list, key=lambda x: int(x.split("_")[1].split(".")[0]))

data = []
idx = 0

for fname in file_list:
    full_path = os.path.join(directory, fname)  # 获取完整路径
    arr = np.load(full_path)
    
    # 处理数据
    arr = arr.reshape((arr.shape[0] * arr.shape[1], arr.shape[2]))
    arr = (arr - np.mean(arr, axis=0, keepdims=True)) / np.std(arr, axis=0, keepdims=True)

    # 提取文件名中的数字作为标签
    label = int(fname.split("_")[1].split(".")[0])
    arr = np.concatenate([arr, np.array([np.full(arr.shape[0], label)]).T], axis=1)

    data.append(arr)
    print(f"Processing {idx}: {fname} -> shape {arr.shape}")
    idx += 1

# 合并所有数据
data = np.concatenate(data, axis=0)
print("Final concatenated shape:", data.shape)

# 保存结果
np.save("concat.npy", data)
print("Saved to concat.npy")

