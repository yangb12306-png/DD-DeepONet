###############################################################################################################
# 计算测试集上ML2RE和MAE
import numpy as np

data_set = np.loadtxt('/data2/project_share/yangbo/NS_pipe_DD_reorder/data_Gen/ZUZ/data_ZUZ_100.txt')
dx1 = data_set[:, 0]
dy1 = data_set[:, 1]
dx2 = data_set[:, 2]
dy2 = data_set[:, 3]
dx3 = data_set[:, 4]
dy3 = data_set[:, 5]
dx4 = data_set[:, 6]
dy4 = data_set[:, 7]
dx5 = data_set[:, 8]
dy5 = data_set[:, 9]
inlet_vel = data_set[:, 10]
test_num = len(inlet_vel)


# true
true_part1 = np.loadtxt("/data2/project_share/yangbo/NS_pipe_DD_reorder/data_Gen/ZUZ/label_part1.txt")
true_part2 = np.loadtxt("/data2/project_share/yangbo/NS_pipe_DD_reorder/data_Gen/ZUZ/label_part2.txt")
true_part3 = np.loadtxt("/data2/project_share/yangbo/NS_pipe_DD_reorder/data_Gen/ZUZ/label_part3.txt")
true_part4 = np.loadtxt("/data2/project_share/yangbo/NS_pipe_DD_reorder/data_Gen/ZUZ/label_part4.txt")
true_part5 = np.loadtxt("/data2/project_share/yangbo/NS_pipe_DD_reorder/data_Gen/ZUZ/label_part5.txt")
print(true_part1.shape)
print(true_part2.shape)
print(true_part3.shape)
print(true_part4.shape)
print(true_part5.shape)

# pred
pred_part1 = np.loadtxt("/data2/yangbo/NS_pipe_DD_reorder/train_UZ_40400/DD_23_ZUZ/batch_2000/part1.txt")
pred_part2 = np.loadtxt("/data2/yangbo/NS_pipe_DD_reorder/train_UZ_40400/DD_23_ZUZ/batch_2000/part2.txt")
pred_part3 = np.loadtxt("/data2/yangbo/NS_pipe_DD_reorder/train_UZ_40400/DD_23_ZUZ/batch_2000/part3.txt")
pred_part4 = np.loadtxt("/data2/yangbo/NS_pipe_DD_reorder/train_UZ_40400/DD_23_ZUZ/batch_2000/part4.txt")
pred_part5 = np.loadtxt("/data2/yangbo/NS_pipe_DD_reorder/train_UZ_40400/DD_23_ZUZ/batch_2000/part5.txt")
print(pred_part1.shape)
pred_part1 = pred_part1[0:100, :]
pred_part2 = pred_part2[0:100, :]
pred_part3 = pred_part3[0:100, :]
pred_part4 = pred_part4[0:100, :]
pred_part5 = pred_part5[0:100, :]
print(pred_part1.shape)
print(pred_part2.shape)
print(pred_part3.shape)
print(pred_part4.shape)
print(pred_part5.shape)

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
    np.savetxt("L2RE.txt", l2_relative_errors, fmt="%.8e")
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
true = np.concatenate([true_part1, true_part2, true_part3, true_part4, true_part5], axis = 1)
pred = np.concatenate([pred_part1, pred_part2, pred_part3, pred_part4, pred_part5], axis = 1)

avg_l2_re = compute_avg_l2_relative_error(true, pred)
avg_re = compute_avg_absolute_error(true, pred)

print(f"平均L2相对误差: {avg_l2_re:.6f}")
print(f"平均绝对误差: {avg_re:.6f}")

