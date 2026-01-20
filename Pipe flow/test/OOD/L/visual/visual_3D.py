###############################################################################
import numpy as np
from vtkwrite import write_vtk
data_id = -1

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

# fraction_train = 0.8
# N_train = int( sample_num * fraction_train )
# train_case = np.array(list(range(N_train))) 
# test_case =  np.array(list(range(N_train, sample_num))) 
# print(sample_num)
# print(len(train_case))
print(f'indices_{data_id} =', indices[sample_num + data_id] + 1)

# true
label_part1 = np.loadtxt("/data2/project_share/yangbo/NS_pipe_Dirichlet/data/part1/label_part1.txt")
label_part2 = np.loadtxt("/data2/project_share/yangbo/NS_pipe_Dirichlet/data/part2/label_part2.txt")
# print(label_part3.shape)
true_part1 = label_part1[indices, :: ]
true_part2 = label_part2[indices, :: ]
true_part1 = true_part1[data_id, :: ]
true_part2 = true_part2[data_id, :: ]

dx1 = data_set[indices, 0]
dy1 = data_set[indices, 1]
dx2 = data_set[indices, 2]
dy2 = data_set[indices, 3]
inlet_vel = data_set[indices, 4]

dx1 = dx1[-1]
dy1 = dy1[-1]
dx2 = dx2[-1]
dy2 = dy2[-1]
inlet_vel = inlet_vel[-1]

print("dx1 =", dx1)
print("dy1 =", dy1)
print("dx2 =", dx2)
print("dy2 =", dy2)

# pred
pred_part1 = np.loadtxt("/data2/yangbo/NS_pipe_DD_reorder/train_UZ_40400/DD_L/batch/part1.txt")
pred_part2 = np.loadtxt("/data2/yangbo/NS_pipe_DD_reorder/train_UZ_40400/DD_L/batch/part2.txt")
pred_part1 = pred_part1[data_id, :: ]
pred_part2 = pred_part2[data_id, :: ]

trunk_part1 = np.loadtxt('/data2/project_share/yangbo/NS_pipe_Dirichlet/data/part1/trunk_part1.txt')
trunk_part1_x = trunk_part1[:, 0] * (dx1 + dx2)
trunk_part1_y = trunk_part1[:, 1] * dy1
trunk_part1[:,0] = trunk_part1_x
trunk_part1[:,1] = trunk_part1_y
trunk_part1 = np.concatenate([trunk_part1, np.zeros([trunk_part1.shape[0], 1])], axis = 1)

trunk_part2 = np.loadtxt('/data2/project_share/yangbo/NS_pipe_Dirichlet/data/part2/trunk_part2.txt')
trunk_part2_x = trunk_part2[:, 0] * dx2 + dx1
trunk_part2_y = trunk_part2[:, 1] * dy2 + (dy1 - dy2)
trunk_part2[:,0] = trunk_part2_x
trunk_part2[:,1] = trunk_part2_y
trunk_part2 = np.concatenate([trunk_part2, np.zeros([trunk_part2.shape[0], 1])], axis = 1)

error_part1 = np.abs(true_part1 - pred_part1)
error_part2 = np.abs(true_part2 - pred_part2)

write_vtk(f'/data2/yangbo/NS_pipe_DD_reorder/train_UZ_40400/DD_L/visual/vtk/true_part1_{data_id}.vtk', trunk_part1, true_part1, precision=8)
write_vtk(f'/data2/yangbo/NS_pipe_DD_reorder/train_UZ_40400/DD_L/visual/vtk/pred_part1_{data_id}.vtk', trunk_part1, pred_part1, precision=8)
write_vtk(f'/data2/yangbo/NS_pipe_DD_reorder/train_UZ_40400/DD_L/visual/vtk/error_part1_{data_id}.vtk', trunk_part1, error_part1, precision=8)

write_vtk(f'/data2/yangbo/NS_pipe_DD_reorder/train_UZ_40400/DD_L/visual/vtk/true_part2_{data_id}.vtk', trunk_part2, true_part2, precision=8)
write_vtk(f'/data2/yangbo/NS_pipe_DD_reorder/train_UZ_40400/DD_L/visual/vtk/pred_part2_{data_id}.vtk', trunk_part2, pred_part2, precision=8)
write_vtk(f'/data2/yangbo/NS_pipe_DD_reorder/train_UZ_40400/DD_L/visual/vtk/error_part2_{data_id}.vtk', trunk_part2, error_part2, precision=8)



###############################################################################


# # 直接做预测的结果，不做DDM的可视化
# import numpy as np
# from vtkwrite import write_vtk
# data_id = 0

# # true
# data_set = np.loadtxt('/data2/project_share/yangbo/NS_pipe/data/data_Pipe_filtered.txt')

# v_ref = 0.1
# v = data_set[:, 4]
# indices = []
# for i in range(data_set.shape[0]):
#     if np.abs(v[i] - v_ref) < 1e-8:
#         indices.append(i)  
#     else:
#         continue

# sample_num = len(indices)

# fraction_train = 0.8
# N_train = int( sample_num * fraction_train )
# train_case = np.array(list(range(N_train))) 
# test_case =  np.array(list(range(N_train, sample_num))) 
# # print(sample_num)
# # print(len(train_case))
# print(f'indices_{data_id} =', indices[len(train_case) + data_id] + 1)

# # true
# label_part1 = np.loadtxt("/data2/project_share/yangbo/NS_pipe/data/part1/label_part1.txt")
# label_part2 = np.loadtxt("/data2/project_share/yangbo/NS_pipe/data/part2/label_part2.txt")
# label_part3 = np.loadtxt("/data2/project_share/yangbo/NS_pipe/data/part3/label_part3.txt")
# # print(label_part3.shape)
# true_part1 = label_part1[indices, :: ]
# true_part2 = label_part2[indices, :: ]
# true_part3 = label_part3[indices, :: ]
# true_part1 = true_part1[len(train_case) + data_id, :: ]
# true_part2 = true_part2[len(train_case) + data_id, :: ]
# true_part3 = true_part3[len(train_case) + data_id, :: ]

# dx1 = data_set[indices, 0]
# dy1 = data_set[indices, 1]
# dx2 = data_set[indices, 2]
# dy2 = data_set[indices, 3]
# inlet_vel = data_set[indices, 4]
# dx1 = dx1[len(train_case) + data_id]
# dy1 = dy1[len(train_case) + data_id]
# dx2 = dx2[len(train_case) + data_id]
# dy2 = dy2[len(train_case) + data_id]
# inlet_vel = inlet_vel[len(train_case) + data_id]
# print("dx1 =", dx1)
# print("dy1 =", dy1)
# print("dx2 =", dx2)
# print("dy2 =", dy2)

# # pred
# pred_part1 = np.loadtxt("/data2/project_share/yangbo/NS_pipe/train_v_0.1/test/part1/pred_part1.txt")
# pred_part2 = np.loadtxt("/data2/project_share/yangbo/NS_pipe/train_v_0.1/test/part2/pred_part2.txt")
# pred_part3 = np.loadtxt("/data2/project_share/yangbo/NS_pipe/train_v_0.1/test/part3/pred_part3.txt")
# pred_part1 = pred_part1[data_id, :: ]
# pred_part2 = pred_part2[data_id, :: ]
# pred_part3 = pred_part3[data_id, :: ]

# trunk_part1 = np.loadtxt('/data2/project_share/yangbo/NS_pipe/data/part1/trunk_part1.txt')
# trunk_part1_x = trunk_part1[:, 0] * dx1
# trunk_part1_y = trunk_part1[:, 1] * dy1
# trunk_part1[:,0] = trunk_part1_x
# trunk_part1[:,1] = trunk_part1_y
# trunk_part1 = np.concatenate([trunk_part1, np.zeros([trunk_part1.shape[0], 1])], axis = 1)

# trunk_part2 = np.loadtxt('/data2/project_share/yangbo/NS_pipe/data/part2/trunk_part2.txt')
# trunk_part2_x = trunk_part2[:, 0] * dx2 + dx1
# trunk_part2_y = trunk_part2[:, 1] * dy1
# trunk_part2[:,0] = trunk_part2_x
# trunk_part2[:,1] = trunk_part2_y
# trunk_part2 = np.concatenate([trunk_part2, np.zeros([trunk_part2.shape[0], 1])], axis = 1)

# trunk_part3 = np.loadtxt('/data2/project_share/yangbo/NS_pipe/data/part3/trunk_part3.txt')
# trunk_part3_x = trunk_part3[:, 0] * dx2 + dx1
# trunk_part3_y = trunk_part3[:, 1] * (dy2 - dy1) - (dy2 - dy1)
# trunk_part3[:,0] = trunk_part3_x
# trunk_part3[:,1] = trunk_part3_y
# trunk_part3 = np.concatenate([trunk_part3, np.zeros([trunk_part3.shape[0], 1])], axis = 1)

# error_part1 = np.abs(true_part1 - pred_part1)
# error_part2 = np.abs(true_part2 - pred_part2)
# error_part3 = np.abs(true_part3 - pred_part3)

# write_vtk(f'/data2/project_share/yangbo/NS_pipe/train_v_0.1/visual/vtk_result/Direct/true_part1_{data_id}.vtk', trunk_part1, true_part1, precision=8)
# write_vtk(f'/data2/project_share/yangbo/NS_pipe/train_v_0.1/visual/vtk_result/Direct/pred_part1_{data_id}.vtk', trunk_part1, pred_part1, precision=8)
# write_vtk(f'/data2/project_share/yangbo/NS_pipe/train_v_0.1/visual/vtk_result/Direct/error_part1_{data_id}.vtk', trunk_part1, error_part1, precision=8)

# write_vtk(f'/data2/project_share/yangbo/NS_pipe/train_v_0.1/visual/vtk_result/Direct/true_part2_{data_id}.vtk', trunk_part2, true_part2, precision=8)
# write_vtk(f'/data2/project_share/yangbo/NS_pipe/train_v_0.1/visual/vtk_result/Direct/pred_part2_{data_id}.vtk', trunk_part2, pred_part2, precision=8)
# write_vtk(f'/data2/project_share/yangbo/NS_pipe/train_v_0.1/visual/vtk_result/Direct/error_part2_{data_id}.vtk', trunk_part2, error_part2, precision=8)

# write_vtk(f'/data2/project_share/yangbo/NS_pipe/train_v_0.1/visual/vtk_result/Direct/true_part3_{data_id}.vtk', trunk_part3, true_part3, precision=8)
# write_vtk(f'/data2/project_share/yangbo/NS_pipe/train_v_0.1/visual/vtk_result/Direct/pred_part3_{data_id}.vtk', trunk_part3, pred_part3, precision=8)
# write_vtk(f'/data2/project_share/yangbo/NS_pipe/train_v_0.1/visual/vtk_result/Direct/error_part3_{data_id}.vtk', trunk_part3, error_part3, precision=8)










