# 计算测试集上ML2RE和MAE
import numpy as np

# true
data_set = np.loadtxt('/data2/project_share/yangbo/NS_pipe_Dirichlet/data/data_Pipe_filtered.txt')
v_ref = 0.1
v = data_set[:, 4]
indices = []
for i in range(data_set.shape[0]):
    if np.abs(v[i] - v_ref) < 1e-8:
        indices.append(i)  
    else:
        continue

sample_num = len(indices)


# true
label_part1 = np.loadtxt("/data2/project_share/yangbo/NS_pipe_Dirichlet/data/part1/label_part1.txt")
true_part1 = label_part1[indices, :: ]
print(true_part1.shape)
label_part2 = np.loadtxt("/data2/project_share/yangbo/NS_pipe_Dirichlet/data/part2/label_part2.txt")
true_part2 = label_part2[indices, :: ]
print(true_part2.shape)

# pred
pred_part1 = np.loadtxt("/data2/yangbo/NS_pipe_DD_reorder/train_UZ_40400/DD_L/batch/part1.txt")
pred_part2 = np.loadtxt("/data2/yangbo/NS_pipe_DD_reorder/train_UZ_40400/DD_L/batch/part2.txt")
print(pred_part1.shape)
print(pred_part2.shape)

epsilon = 1e-10

# --------------------- 计算平均L2相对误差 ---------------------
def compute_avg_l2_relative_error(true, pred, epsilon=epsilon):
    l2_relative_errors = []
    for i in range(true.shape[0]):
        true_row = true[i, :]
        pred_row = pred[i, :]
        # 计算分子：预测与真实值的L2范数差
        numerator = np.linalg.norm(pred_row - true_row, ord=2)
        # 计算分母：真实值的L2范数（避免为零）
        denominator = np.linalg.norm(true_row, ord=2) + epsilon
        # 单样本的相对误差
        l2_relative_error = numerator / denominator
        l2_relative_errors.append(l2_relative_error)
    # 返回全局平均
    return np.mean(l2_relative_errors)

# --------------------- 计算平均相对误差 ---------------------
def compute_avg_absolute_error(true, pred):
    total_mae = 0.0
    for i in range(true.shape[0]):
        true_row = true[i, :]
        pred_row = pred[i, :]
        total_mae += np.mean(np.abs(pred_row - true_row))  # 逐行计算MAE
    return total_mae / true.shape[0]

# --------------------- 调用函数 ---------------------
# true = true_part1
# pred = pred_part1
true = np.concatenate([true_part1, true_part2], axis = 1)
pred = np.concatenate([pred_part1, pred_part2], axis = 1)

avg_l2_re = compute_avg_l2_relative_error(true, pred)
avg_re = compute_avg_absolute_error(true, pred)

print(f"平均L2相对误差: {avg_l2_re:.6f}")
print(f"平均绝对误差: {avg_re:.6f}")

