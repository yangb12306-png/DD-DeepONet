###############################################################################
import numpy as np
from vtkwrite import write_vtk
data_id = -4

# true
data_set = np.loadtxt('/data2/project_share/yangbo/NS_pipe_DD_reorder/data_Gen/UZU/data_UZU_100.txt')
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

dx1 = dx1[data_id]
dy1 = dy1[data_id]
dx2 = dx2[data_id]
dy2 = dy2[data_id]
dx3 = dx3[data_id]
dy3 = dy3[data_id]
dx4 = dx4[data_id]
dy4 = dy4[data_id]
dx5 = dx5[data_id]
dy5 = dy5[data_id]

# true
true_part1 = np.loadtxt("/data2/project_share/yangbo/NS_pipe_DD_reorder/data_Gen/UZU/label_part1.txt")
true_part2 = np.loadtxt("/data2/project_share/yangbo/NS_pipe_DD_reorder/data_Gen/UZU/label_part2.txt")
true_part3 = np.loadtxt("/data2/project_share/yangbo/NS_pipe_DD_reorder/data_Gen/UZU/label_part3.txt")
true_part4 = np.loadtxt("/data2/project_share/yangbo/NS_pipe_DD_reorder/data_Gen/UZU/label_part4.txt")
true_part5 = np.loadtxt("/data2/project_share/yangbo/NS_pipe_DD_reorder/data_Gen/UZU/label_part5.txt")
true_part1 = true_part1[data_id, :: ]
true_part2 = true_part2[data_id, :: ]
true_part3 = true_part3[data_id, :: ]
true_part4 = true_part4[data_id, :: ]
true_part5 = true_part5[data_id, :: ]



# pred
pred_part1 = np.loadtxt("/data2/yangbo/NS_pipe_DD_reorder/train_UZ_40400/DD_23_UZU/batch/part1.txt")
pred_part2 = np.loadtxt("/data2/yangbo/NS_pipe_DD_reorder/train_UZ_40400/DD_23_UZU/batch/part2.txt")
pred_part3 = np.loadtxt("/data2/yangbo/NS_pipe_DD_reorder/train_UZ_40400/DD_23_UZU/batch/part3.txt")
pred_part4 = np.loadtxt("/data2/yangbo/NS_pipe_DD_reorder/train_UZ_40400/DD_23_UZU/batch/part4.txt")
pred_part5 = np.loadtxt("/data2/yangbo/NS_pipe_DD_reorder/train_UZ_40400/DD_23_UZU/batch/part5.txt")
# pred_part1 = np.loadtxt("/data2/yangbo/NS_pipe_DD_reorder/L4U_test_100/train_L/pred_directly/part1_directly.txt")
# pred_part2 = np.loadtxt("/data2/yangbo/NS_pipe_DD_reorder/L4U_test_100/train_L/pred_directly/part2_directly.txt")
# pred_part3 = np.loadtxt("/data2/yangbo/NS_pipe_DD_reorder/L4U_test_100/train_L/pred_directly/part3_directly.txt")
# pred_part4 = np.loadtxt("/data2/yangbo/NS_pipe_DD_reorder/L4U_test_100/train_L/pred_directly/part4_directly.txt")
pred_part1 = pred_part1[data_id, :: ]
pred_part2 = pred_part2[data_id, :: ]
pred_part3 = pred_part3[data_id, :: ]
pred_part4 = pred_part4[data_id, :: ]
pred_part5 = pred_part5[data_id, :: ]

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
x_points_num = 200
y_points_num = 40
i, j = np.mgrid[0:x_points_num, 0:y_points_num]
i, j = i.ravel(order="F"), j.ravel(order="F")  # 矢量化索引
x_coords = dx1 + dx2 - dx3 - dx4 + i * ((dx3 + dx4) / (x_points_num - 1))
y_coords = dy1 - dy2 -j * (dy3 / (y_points_num - 1))
trunk_part3 = np.column_stack((x_coords, y_coords, np.zeros_like(x_coords)))

# part4 coordinate points
x_points_num = 40
y_points_num = 200
i, j = np.mgrid[0:x_points_num, 0:y_points_num]
i, j = i.ravel(), j.ravel()  # 矢量化索引
x_coords = (dx1 + dx2 - dx3) - i * (dx4 / (x_points_num - 1))
y_coords = dy1 - dy2 -dy4 - dy5 + j * ((dy4 + dy5) / (y_points_num - 1))
trunk_part4 = np.column_stack((x_coords, y_coords, np.zeros_like(x_coords)))

# part5 coordinate points
x_points_num = 160
y_points_num = 40
i, j = np.mgrid[0:x_points_num, 0:y_points_num]
i, j = i.ravel(order="F"), j.ravel(order="F")  # 矢量化索引
x_coords = (dx1 + dx2 - dx3 - dx4 + dx5) - i * (dx5 / (x_points_num - 1))
y_coords = dy1 - dy2 - dy4 - j * (dy5 / (y_points_num - 1))
trunk_part5 = np.column_stack((x_coords, y_coords, np.zeros_like(x_coords)))

error_part1 = np.abs(true_part1 - pred_part1)
error_part2 = np.abs(true_part2 - pred_part2)
error_part3 = np.abs(true_part3 - pred_part3)
error_part4 = np.abs(true_part4 - pred_part4)
error_part5 = np.abs(true_part5 - pred_part5)

write_vtk(f'/data2/yangbo/NS_pipe_DD_reorder/train_UZ_40400/DD_23_UZU/visual/vtk/true_part1_{data_id}.vtk', trunk_part1, true_part1, precision=8)
write_vtk(f'/data2/yangbo/NS_pipe_DD_reorder/train_UZ_40400/DD_23_UZU/visual/vtk/pred_part1_{data_id}.vtk', trunk_part1, pred_part1, precision=8)
write_vtk(f'/data2/yangbo/NS_pipe_DD_reorder/train_UZ_40400/DD_23_UZU/visual/vtk/error_part1_{data_id}.vtk', trunk_part1, error_part1, precision=8)

write_vtk(f'/data2/yangbo/NS_pipe_DD_reorder/train_UZ_40400/DD_23_UZU/visual/vtk/true_part2_{data_id}.vtk', trunk_part2, true_part2, precision=8)
write_vtk(f'/data2/yangbo/NS_pipe_DD_reorder/train_UZ_40400/DD_23_UZU/visual/vtk/pred_part2_{data_id}.vtk', trunk_part2, pred_part2, precision=8)
write_vtk(f'/data2/yangbo/NS_pipe_DD_reorder/train_UZ_40400/DD_23_UZU/visual/vtk/error_part2_{data_id}.vtk', trunk_part2, error_part2, precision=8)

write_vtk(f'/data2/yangbo/NS_pipe_DD_reorder/train_UZ_40400/DD_23_UZU/visual/vtk/true_part3_{data_id}.vtk', trunk_part3, true_part3, precision=8)
write_vtk(f'/data2/yangbo/NS_pipe_DD_reorder/train_UZ_40400/DD_23_UZU/visual/vtk/pred_part3_{data_id}.vtk', trunk_part3, pred_part3, precision=8)
write_vtk(f'/data2/yangbo/NS_pipe_DD_reorder/train_UZ_40400/DD_23_UZU/visual/vtk/error_part3_{data_id}.vtk', trunk_part3, error_part3, precision=8)

write_vtk(f'/data2/yangbo/NS_pipe_DD_reorder/train_UZ_40400/DD_23_UZU/visual/vtk/true_part4_{data_id}.vtk', trunk_part4, true_part4, precision=8)
write_vtk(f'/data2/yangbo/NS_pipe_DD_reorder/train_UZ_40400/DD_23_UZU/visual/vtk/pred_part4_{data_id}.vtk', trunk_part4, pred_part4, precision=8)
write_vtk(f'/data2/yangbo/NS_pipe_DD_reorder/train_UZ_40400/DD_23_UZU/visual/vtk/error_part4_{data_id}.vtk', trunk_part4, error_part4, precision=8)

write_vtk(f'/data2/yangbo/NS_pipe_DD_reorder/train_UZ_40400/DD_23_UZU/visual/vtk/true_part5_{data_id}.vtk', trunk_part5, true_part5, precision=8)
write_vtk(f'/data2/yangbo/NS_pipe_DD_reorder/train_UZ_40400/DD_23_UZU/visual/vtk/pred_part5_{data_id}.vtk', trunk_part5, pred_part5, precision=8)
write_vtk(f'/data2/yangbo/NS_pipe_DD_reorder/train_UZ_40400/DD_23_UZU/visual/vtk/error_part5_{data_id}.vtk', trunk_part5, error_part5, precision=8)

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











