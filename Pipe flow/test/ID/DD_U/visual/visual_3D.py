###############################################################################
import numpy as np
from vtkwrite import write_vtk
data_id = -1

# true
data_set = np.loadtxt('/data2/project_share/yangbo/NS_pipe_DD_reorder/data_U_train_10100/data_10100_pipe.txt')
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
# print(test_case)
test_num = len(test_case)

dx1 = dx1[ test_case ]
dy1 = dy1[ test_case ]
dx2 = dx2[ test_case ]
dy2 = dy2[ test_case ]
dx3 = dx3[ test_case ]
dy3 = dy3[ test_case ]
inlet_vel = inlet_vel[test_case ]
dx1 = dx1[data_id]
dy1 = dy1[data_id]
dx2 = dx2[data_id]
dy2 = dy2[data_id]
dx3 = dx3[data_id]
dy3 = dy3[data_id]

print(dx1, dy1, dx2, dy2, dx3, dy3)

# true
true_part1 = np.loadtxt("/data2/project_share/yangbo/NS_pipe_DD_reorder/data_U_train_10100/label_part1.txt")
true_part2 = np.loadtxt("/data2/project_share/yangbo/NS_pipe_DD_reorder/data_U_train_10100/label_part23.txt")
true_part3 = np.loadtxt("/data2/project_share/yangbo/NS_pipe_DD_reorder/data_U_train_10100/label_part4.txt")
true_part1 = true_part1[test_case, :: ]
true_part2 = true_part2[test_case, :: ]
true_part3 = true_part3[test_case, :: ]
true_part1 = true_part1[data_id, :: ]
true_part2 = true_part2[data_id, :: ]
true_part3 = true_part3[data_id, :: ]



# pred
pred_part1 = np.loadtxt("/data2/yangbo/NS_pipe_DD_reorder/train_UZ_40400/DD_U/batch/both/part1.txt")
pred_part2 = np.loadtxt("/data2/yangbo/NS_pipe_DD_reorder/train_UZ_40400/DD_U/batch/both/part2.txt")
pred_part3 = np.loadtxt("/data2/yangbo/NS_pipe_DD_reorder/train_UZ_40400/DD_U/batch/both/part3.txt")
# pred_part1 = np.loadtxt("/data2/yangbo/NS_pipe_DD_reorder/L4U_test_100/train_L/pred_directly/part1_directly.txt")
# pred_part2 = np.loadtxt("/data2/yangbo/NS_pipe_DD_reorder/L4U_test_100/train_L/pred_directly/part2_directly.txt")
# pred_part3 = np.loadtxt("/data2/yangbo/NS_pipe_DD_reorder/L4U_test_100/train_L/pred_directly/part3_directly.txt")
# pred_part4 = np.loadtxt("/data2/yangbo/NS_pipe_DD_reorder/L4U_test_100/train_L/pred_directly/part4_directly.txt")
pred_part1 = pred_part1[data_id, :: ]
pred_part2 = pred_part2[data_id, :: ]
pred_part3 = pred_part3[data_id, :: ]

# part1 coordinate points
x_points_num = 160
y_points_num = 40
i, j = np.mgrid[0:x_points_num, 0:y_points_num]
i, j = i.ravel(), j.ravel()  # 矢量化索引
x_coords = i * ((dx1 + dx2) / (x_points_num - 1))
y_coords = j * (dy1 / (y_points_num - 1))
trunk_part1 = np.column_stack((x_coords, y_coords, np.zeros_like(x_coords)))

# part2 coordinate points
x_points_num = 40
y_points_num = 200
i, j = np.mgrid[0:x_points_num, 0:y_points_num]
i, j = i.ravel(), j.ravel()  # 矢量化索引
x_coords = dx1 + i * (dx2 / (x_points_num - 1))
y_coords = dy1 - dy2 - dy3 + j * ((dy2 + dy3) / (y_points_num - 1))
trunk_part2 = np.column_stack((x_coords, y_coords, np.zeros_like(x_coords)))

# part3 coordinate points
x_points_num = 160
y_points_num = 40
i, j = np.mgrid[0:x_points_num, 0:y_points_num]
i, j = i.ravel(order="F"), j.ravel(order="F")  # 矢量化索引
x_coords = dx1 + dx2 - dx3 + i * (dx3 / (x_points_num - 1))
y_coords = dy1 - dy2 - j * (dy3 / (y_points_num - 1))
trunk_part3 = np.column_stack((x_coords, y_coords, np.zeros_like(x_coords)))

error_part1 = np.abs(true_part1 - pred_part1)
error_part2 = np.abs(true_part2 - pred_part2)
error_part3 = np.abs(true_part3 - pred_part3)

write_vtk(f'/data2/yangbo/NS_pipe_DD_reorder/train_UZ_40400/DD_U/visual/vtk/true_part1_{data_id}.vtk', trunk_part1, true_part1, precision=8)
write_vtk(f'/data2/yangbo/NS_pipe_DD_reorder/train_UZ_40400/DD_U/visual/vtk/pred_part1_{data_id}.vtk', trunk_part1, pred_part1, precision=8)
write_vtk(f'/data2/yangbo/NS_pipe_DD_reorder/train_UZ_40400/DD_U/visual/vtk/error_part1_{data_id}.vtk', trunk_part1, error_part1, precision=8)

write_vtk(f'/data2/yangbo/NS_pipe_DD_reorder/train_UZ_40400/DD_U/visual/vtk/true_part2_{data_id}.vtk', trunk_part2, true_part2, precision=8)
write_vtk(f'/data2/yangbo/NS_pipe_DD_reorder/train_UZ_40400/DD_U/visual/vtk/pred_part2_{data_id}.vtk', trunk_part2, pred_part2, precision=8)
write_vtk(f'/data2/yangbo/NS_pipe_DD_reorder/train_UZ_40400/DD_U/visual/vtk/error_part2_{data_id}.vtk', trunk_part2, error_part2, precision=8)

write_vtk(f'/data2/yangbo/NS_pipe_DD_reorder/train_UZ_40400/DD_U/visual/vtk/true_part3_{data_id}.vtk', trunk_part3, true_part3, precision=8)
write_vtk(f'/data2/yangbo/NS_pipe_DD_reorder/train_UZ_40400/DD_U/visual/vtk/pred_part3_{data_id}.vtk', trunk_part3, pred_part3, precision=8)
write_vtk(f'/data2/yangbo/NS_pipe_DD_reorder/train_UZ_40400/DD_U/visual/vtk/error_part3_{data_id}.vtk', trunk_part3, error_part3, precision=8)

# write_vtk(f'/data2/yangbo/NS_pipe_DD_reorder/L4U_test_100/test_U/visual/vtk_directly/true_part1_{data_id}.vtk', trunk_part1, true_part1, precision=8)
# write_vtk(f'/data2/yangbo/NS_pipe_DD_reorder/L4U_test_100/test_U/visual/vtk_directly/pred_part1_{data_id}.vtk', trunk_part1, pred_part1, precision=8)
# write_vtk(f'/data2/yangbo/NS_pipe_DD_reorder/L4U_test_100/test_U/visual/vtk_directly/error_part1_{data_id}.vtk', trunk_part1, error_part1, precision=8)

# write_vtk(f'/data2/yangbo/NS_pipe_DD_reorder/L4U_test_100/test_U/visual/vtk_directly/true_part2_{data_id}.vtk', trunk_part2, true_part2, precision=8)
# write_vtk(f'/data2/yangbo/NS_pipe_DD_reorder/L4U_test_100/test_U/visual/vtk_directly/pred_part2_{data_id}.vtk', trunk_part2, pred_part2, precision=8)
# write_vtk(f'/data2/yangbo/NS_pipe_DD_reorder/L4U_test_100/test_U/visual/vtk_directly/error_part2_{data_id}.vtk', trunk_part2, error_part2, precision=8)

# write_vtk(f'/data2/yangbo/NS_pipe_DD_reorder/L4U_test_100/test_U/visual/vtk_directly/true_part3_{data_id}.vtk', trunk_part3, true_part3, precision=8)
# write_vtk(f'/data2/yangbo/NS_pipe_DD_reorder/L4U_test_100/test_U/visual/vtk_directly/pred_part3_{data_id}.vtk', trunk_part3, pred_part3, precision=8)
# write_vtk(f'/data2/yangbo/NS_pipe_DD_reorder/L4U_test_100/test_U/visual/vtk_directly/error_part3_{data_id}.vtk', trunk_part3, error_part3, precision=8)

# write_vtk(f'/data2/yangbo/NS_pipe_DD_reorder/L4U_test_100/test_U/visual/vtk_directly/true_part4_{data_id}.vtk', trunk_part4, true_part4, precision=8)
# write_vtk(f'/data2/yangbo/NS_pipe_DD_reorder/L4U_test_100/test_U/visual/vtk_directly/pred_part4_{data_id}.vtk', trunk_part4, pred_part4, precision=8)
# write_vtk(f'/data2/yangbo/NS_pipe_DD_reorder/L4U_test_100/test_U/visual/vtk_directly/error_part4_{data_id}.vtk', trunk_part4, error_part4, precision=8)

###############################################################################











