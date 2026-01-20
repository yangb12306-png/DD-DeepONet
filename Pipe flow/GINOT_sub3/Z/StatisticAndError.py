###############################################################################################################
# 计算测试集上ML2RE和MAE
import numpy as np

data_set = np.loadtxt('/data2/project_share/yangbo/NS_pipe_DD_reorder/data_Z_skewing/data_skewing_10073.txt')
dx1 = data_set[:, 0]
dy1 = data_set[:, 1]
dx2 = data_set[:, 2]
dy2 = data_set[:, 3]
dx3 = data_set[:, 4]
dy3 = data_set[:, 5]
inlet_vel = data_set[:, 6]

fraction_train = 0.8
sample_num = len(inlet_vel)
N_train = int( sample_num * fraction_train )
train_case = np.array(list(range(N_train))) 
test_case =  np.array(list(range(N_train, sample_num))) 
test_num = len(test_case)

# true
true_part1 = np.loadtxt("/data2/project_share/yangbo/NS_pipe_DD_reorder/data_Z_skewing/label_part1.txt")
true_part2 = np.loadtxt("/data2/project_share/yangbo/NS_pipe_DD_reorder/data_Z_skewing/label_part23.txt")
# true_part3 = np.loadtxt("/data2/project_share/yangbo/NS_pipe_DD_reorder/data_U_skewing/label_part4.txt")
true_part1_skew = true_part1[test_case, :]
true_part2_skew = true_part2[test_case, :]
# true_part3_skew = true_part3[test_case, :]
print(true_part1_skew.shape)
print(true_part2_skew.shape)
# print(true_part3_skew.shape)








data_set = np.loadtxt('/data2/project_share/yangbo/NS_pipe_DD_reorder/data_Z_train_10100/data_10100_pipe.txt')
dx1 = data_set[:, 0]
dy1 = data_set[:, 1]
dx2 = data_set[:, 2]
dy2 = data_set[:, 3]
dx3 = data_set[:, 4]
dy3 = data_set[:, 5]
inlet_vel = data_set[:, 6]

fraction_train = 0.8
sample_num = len(inlet_vel)
N_train = int( sample_num * fraction_train )
train_case = np.array(list(range(N_train))) 
test_case =  np.array(list(range(N_train, sample_num))) 
test_num = len(test_case)

# true
true_part1 = np.loadtxt("/data2/project_share/yangbo/NS_pipe_DD_reorder/data_Z_train_10100/label_part1.txt")
true_part2 = np.loadtxt("/data2/project_share/yangbo/NS_pipe_DD_reorder/data_Z_train_10100/label_part23.txt")
# true_part3 = np.loadtxt("/data2/project_share/yangbo/NS_pipe_DD_reorder/data_U_train_10100/label_part4.txt")
true_part1_01 = true_part1[test_case, :]
true_part2_01 = true_part2[test_case, :]
# true_part3_01 = true_part3[test_case, :]
print(true_part1_01.shape)
print(true_part2_01.shape)
# print(true_part3_01.shape)







true_part1 = np.vstack([true_part1_01, true_part1_skew])
true_part2 = np.vstack([true_part2_01, true_part2_skew])
# true_part3 = np.vstack([true_part3_01, true_part3_skew])
print(true_part1.shape)
print(true_part2.shape)
# print(true_part3.shape)




# pred
pred_part1 = np.loadtxt("/data2/yangbo/GINOT/GINOT-main/Z/part1.txt")
pred_part2 = np.loadtxt("/data2/yangbo/GINOT/GINOT-main/Z/part2.txt")
# pred_part3 = np.loadtxt("/data2/yangbo/NS_pipe_DD_reorder/train_UZ_40400/DD_U/batch/both/part3.txt")
print(pred_part1.shape)
print(pred_part2.shape)
# print(pred_part3.shape)

def load_ragged_1d_txt(path, dtype=float, sep=None):
    """
    sep=None 表示按任意空白分隔（空格/Tab 都行）
    返回：list[np.ndarray]，每行一个 1D 向量，长度可不同
    """
    out = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line == "":
                out.append(np.array([], dtype=dtype))
            else:
                out.append(np.array(line.split(sep), dtype=dtype))
    return out

true_part3 = load_ragged_1d_txt("/data2/yangbo/GINOT/GINOT-main/Z/part3_true.txt")
pred_part3 = load_ragged_1d_txt("/data2/yangbo/GINOT/GINOT-main/Z/part3_pred.txt")

def concat_parts_ragged(part1_2d, part2_2d, part3_ragged):
    """
    part1_2d: np.ndarray, shape (N, D1)
    part2_2d: np.ndarray, shape (N, D2)
    part3_ragged: list[np.ndarray], len=N, each is shape (Li,)
    return: list[np.ndarray], len=N, each is shape (D1+D2+Li,)
    """
    N = part1_2d.shape[0]
    assert part2_2d.shape[0] == N
    assert len(part3_ragged) == N

    out = []
    for i in range(N):
        v1 = np.asarray(part1_2d[i]).reshape(-1)
        v2 = np.asarray(part2_2d[i]).reshape(-1)
        v3 = np.asarray(part3_ragged[i]).reshape(-1)
        out.append(np.concatenate([v1, v2, v3], axis=0))
    return out

true = concat_parts_ragged(true_part1, true_part2, true_part3)
pred = concat_parts_ragged(pred_part1, pred_part2, pred_part3)

epsilon = 1e-10
# --------------------- 计算平均L2相对误差 ---------------------
def compute_avg_l2_relative_error(true, pred, epsilon=epsilon):
    l2_relative_errors = []
    for i in range(len(true)):
        true_row = true[i]
        pred_row = pred[i]
        # 计算分子：预测与真实值的L2范数差
        numerator = np.linalg.norm(pred_row - true_row, ord=2)
        # 计算分母：真实值的L2范数（避免为零）
        denominator = np.linalg.norm(true_row, ord=2) + epsilon
        # 单样本的相对误差
        l2_relative_error = numerator / denominator
        l2_relative_errors.append(l2_relative_error)
    # np.savetxt("L2RE.txt", l2_relative_errors, fmt="%.8e")
    # 返回全局平均
    return np.mean(l2_relative_errors)

# --------------------- 计算平均相对误差 ---------------------
def compute_avg_absolute_error(true, pred):
    total_mae = 0.0
    for i in range(len(true)):
        true_row = true[i]
        pred_row = pred[i]
        total_mae += np.mean(np.abs(pred_row - true_row))  # 逐行计算MAE
    return total_mae / len(true)

# --------------------- 调用函数 ---------------------


avg_l2_re = compute_avg_l2_relative_error(true, pred)
avg_re = compute_avg_absolute_error(true, pred)

print(f"平均L2相对误差: {avg_l2_re:.6f}")
print(f"平均绝对误差: {avg_re:.6f}")

