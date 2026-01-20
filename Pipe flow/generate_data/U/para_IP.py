##############################################
# label
##############################################

########################################################################################
#  part 1

# from paraview.simple import *
# import numpy as np
# import vtk
# import os
# from vtk.util import numpy_support as vtk_np

# paraview.simple._DisableFirstRenderCameraReset()

# data_Pipe = np.loadtxt('/data2/project_share/yangbo/NS_pipe_DD_reorder/data_U_skewing/data_skewing_10100.txt')

# x_points_num = 160
# y_points_num = 40
# i, j = np.mgrid[0:x_points_num, 0:y_points_num]
# i, j = i.ravel(), j.ravel()  # 矢量化索引

# for data_id in range(1, 10101):
#     print(f'Processing data_id = {data_id}')

#     vtu_path = f'/data2/project_share/yangbo/NS_pipe_DD_reorder/data_U_skewing/U_dataset/Pipe_ns_{data_id}.vtu'
#     if not os.path.exists(vtu_path):
#         print(f"Skipping non-existent file: {vtu_path}")
#         continue

#     # 读取VTU文件
#     reader = XMLUnstructuredGridReader(FileName=[vtu_path])
#     reader.TimeArray = 'None'

#     # 生成插值点坐标
#     dx1, dy1, dx2 = data_Pipe[data_id-1, :3]
#     x_coords = i * ((dx1 + dx2) / (x_points_num - 1))
#     y_coords = j * (dy1 / (y_points_num - 1))
#     coords = np.column_stack((x_coords, y_coords, np.zeros_like(x_coords)))

#     # 创建VTK表格
#     vtk_table = vtk.vtkTable()
#     for name, data in zip(['X', 'Y', 'Z'], coords.T):
#         arr = vtk_np.numpy_to_vtk(data.copy(), deep=True)
#         arr.SetName(name)
#         vtk_table.AddColumn(arr)
    
#     # 转换为点集
#     source = TrivialProducer()
#     source.GetClientSideObject().SetOutput(vtk_table)
#     points = TableToPoints(Input=source, XColumn='X', YColumn='Y', ZColumn='Z')

#     # 执行数据重采样
#     resampled = ResampleWithDataset(SourceDataArrays=reader, DestinationMesh=points)
#     resampled.UpdatePipeline()  # 手动更新

#     # 保存结果
#     SaveData(f'/data2/project_share/yangbo/NS_pipe_DD_reorder/data_U_skewing/temp/part1_label_{data_id}.csv', 
#              proxy=resampled,
#              PointDataArrays=['U', 'vtkValidPointMask', '压力', '速度场，x_分量', '速度场，y_分量'])

#     # 释放资源
#     Delete(reader)
#     Delete(resampled)
#     Delete(points)
#     Delete(source)
#     del reader, resampled, points, source
########################################################################################






########################################################################################
#  part 2

# from paraview.simple import *
# import numpy as np
# import vtk
# import os
# from vtk.util import numpy_support as vtk_np

# paraview.simple._DisableFirstRenderCameraReset()

# data_Pipe = np.loadtxt('/data2/project_share/yangbo/NS_pipe_DD_reorder/data_U_train_50000/data_50000.txt')

# x_points_num = 40
# y_points_num = 160
# i, j = np.mgrid[0:x_points_num, 0:y_points_num]
# i, j = i.ravel(order="F"), j.ravel(order="F")  # 矢量化索引

# for data_id in range(1, 50001): 
#     print(f'Processing data_id = {data_id}')

#     vtu_path = f'/data2/project_share/yangbo/NS_pipe_DD_reorder/data_U_train_50000/U_dataset_50000/Pipe_ns_{data_id}.vtu'
#     if not os.path.exists(vtu_path):
#         print(f"Skipping non-existent file: {vtu_path}")
#         continue

#     # 读取VTU文件
#     reader = XMLUnstructuredGridReader(FileName=[vtu_path])
#     reader.TimeArray = 'None'

#     # 生成插值点坐标
#     dx1, dy1, dx2, dy2, dx3, dy3, U_in = data_Pipe[data_id-1, :]
#     x_coords = dx1 + i * (dx2 / (x_points_num - 1))
#     y_coords = dy1 - dy2 + j * (dy2 / (y_points_num - 1))
#     coords = np.column_stack((x_coords, y_coords, np.zeros_like(x_coords)))

#     # 创建VTK表格
#     vtk_table = vtk.vtkTable()
#     for name, data in zip(['X', 'Y', 'Z'], coords.T):
#         arr = vtk_np.numpy_to_vtk(data.copy(), deep=True)
#         arr.SetName(name)
#         vtk_table.AddColumn(arr)
    
#     # 转换为点集
#     source = TrivialProducer()
#     source.GetClientSideObject().SetOutput(vtk_table)
#     points = TableToPoints(Input=source, XColumn='X', YColumn='Y', ZColumn='Z')

#     # 执行数据重采样
#     resampled = ResampleWithDataset(SourceDataArrays=reader, DestinationMesh=points)
#     resampled.UpdatePipeline()  # 手动更新

#     # 保存结果
#     SaveData(f'/data2/project_share/yangbo/NS_pipe_DD_reorder/data_U_train_50000/temp/part2_label_{data_id}.csv', 
#              proxy=resampled,
#              PointDataArrays=['U', 'vtkValidPointMask', '压力', '速度场，x_分量', '速度场，y_分量'])

#     # 释放资源
#     Delete(reader)
#     Delete(resampled)
#     Delete(points)
#     Delete(source)
#     del reader, resampled, points, source

########################################################################################

# #  part 3

# from paraview.simple import *
# import numpy as np
# import vtk
# import os
# from vtk.util import numpy_support as vtk_np

# paraview.simple._DisableFirstRenderCameraReset()

# data_Pipe = np.loadtxt('/data2/project_share/yangbo/NS_pipe_DD_reorder/data_U_train_50000/data_50000.txt')

# x_points_num = 40
# y_points_num = 160
# i, j = np.mgrid[0:x_points_num, 0:y_points_num]
# i, j = i.ravel(order="F"), j.ravel(order="F")  # 矢量化索引

# for data_id in range(1, 50001): # 101
#     print(f'Processing data_id = {data_id}')

#     vtu_path = f'/data2/project_share/yangbo/NS_pipe_DD_reorder/data_U_train_50000/U_dataset_50000/Pipe_ns_{data_id}.vtu'
#     if not os.path.exists(vtu_path):
#         print(f"Skipping non-existent file: {vtu_path}")
#         continue

#     # 读取VTU文件
#     reader = XMLUnstructuredGridReader(FileName=[vtu_path])
#     reader.TimeArray = 'None'

#     # 生成插值点坐标
#     dx1, dy1, dx2, dy2, dx3, dy3, U_in = data_Pipe[data_id-1, :]
#     x_coords = dx1 + i * (dx2 / (x_points_num - 1))
#     y_coords = -j * ((dy2 - dy1 + dy3) / (y_points_num - 1))
#     coords = np.column_stack((x_coords, y_coords, np.zeros_like(x_coords)))

#     # 创建VTK表格
#     vtk_table = vtk.vtkTable()
#     for name, data in zip(['X', 'Y', 'Z'], coords.T):
#         arr = vtk_np.numpy_to_vtk(data.copy(), deep=True)
#         arr.SetName(name)
#         vtk_table.AddColumn(arr)
    
#     # 转换为点集
#     source = TrivialProducer()
#     source.GetClientSideObject().SetOutput(vtk_table)
#     points = TableToPoints(Input=source, XColumn='X', YColumn='Y', ZColumn='Z')

#     # 执行数据重采样
#     resampled = ResampleWithDataset(SourceDataArrays=reader, DestinationMesh=points)
#     resampled.UpdatePipeline()  # 手动更新

#     # 保存结果
#     SaveData(f'/data2/project_share/yangbo/NS_pipe_DD_reorder/data_U_train_50000/temp/part3_label_{data_id}.csv', 
#              proxy=resampled,
#              PointDataArrays=['U', 'vtkValidPointMask', '压力', '速度场，x_分量', '速度场，y_分量'])

#     # 释放资源
#     Delete(reader)
#     Delete(resampled)
#     Delete(points)
#     Delete(source)
#     del reader, resampled, points, source

########################################################################################

# #  part 4

# from paraview.simple import *
# import numpy as np
# import vtk
# import os
# from vtk.util import numpy_support as vtk_np

# paraview.simple._DisableFirstRenderCameraReset()

# data_Pipe = np.loadtxt('/data2/project_share/yangbo/NS_pipe_DD_reorder/data_U_skewing/data_skewing_10100.txt')

# x_points_num = 160
# y_points_num = 40
# i, j = np.mgrid[0:x_points_num, 0:y_points_num]
# i, j = i.ravel(order="F"), j.ravel(order="F")  # 矢量化索引

# for data_id in range(1, 10101): # 101
#     print(f'Processing data_id = {data_id}')

#     vtu_path = f'/data2/project_share/yangbo/NS_pipe_DD_reorder/data_U_skewing/U_dataset/Pipe_ns_{data_id}.vtu'
#     if not os.path.exists(vtu_path):
#         print(f"Skipping non-existent file: {vtu_path}")
#         continue

#     # 读取VTU文件
#     reader = XMLUnstructuredGridReader(FileName=[vtu_path])
#     reader.TimeArray = 'None'

#     # 生成插值点坐标
#     dx1, dy1, dx2, dy2, dx3, dy3, U0, U1 = data_Pipe[data_id-1, :]
#     x_coords = (dx1 + dx2 - dx3) + i * (dx3 / (x_points_num - 1))
#     y_coords = dy1 - dy2 - j * (dy3 / (y_points_num - 1))
#     coords = np.column_stack((x_coords, y_coords, np.zeros_like(x_coords)))

#     # 创建VTK表格
#     vtk_table = vtk.vtkTable()
#     for name, data in zip(['X', 'Y', 'Z'], coords.T):
#         arr = vtk_np.numpy_to_vtk(data.copy(), deep=True)
#         arr.SetName(name)
#         vtk_table.AddColumn(arr)
    
#     # 转换为点集
#     source = TrivialProducer()
#     source.GetClientSideObject().SetOutput(vtk_table)
#     points = TableToPoints(Input=source, XColumn='X', YColumn='Y', ZColumn='Z')

#     # 执行数据重采样
#     resampled = ResampleWithDataset(SourceDataArrays=reader, DestinationMesh=points)
#     resampled.UpdatePipeline()  # 手动更新

#     # 保存结果
#     SaveData(f'/data2/project_share/yangbo/NS_pipe_DD_reorder/data_U_skewing/temp/part4_label_{data_id}.csv', 
#              proxy=resampled,
#              PointDataArrays=['U', 'vtkValidPointMask', '压力', '速度场，x_分量', '速度场，y_分量'])

#     # 释放资源
#     Delete(reader)
#     Delete(resampled)
#     Delete(points)
#     Delete(source)
#     del reader, resampled, points, source

########################################################################################

# #  part 4_reorder

# from paraview.simple import *
# import numpy as np
# import vtk
# import os
# from vtk.util import numpy_support as vtk_np

# paraview.simple._DisableFirstRenderCameraReset()

# data_Pipe = np.loadtxt('/data2/project_share/yangbo/NS_pipe_DD_reorder/data_U_train_50000/data_50000.txt')

# x_points_num = 160
# y_points_num = 40
# i, j = np.mgrid[0:x_points_num, 0:y_points_num]
# i, j = i.ravel(), j.ravel()  # 矢量化索引

# for data_id in range(10101, 50001): # 101
#     print(f'Processing data_id = {data_id}')

#     vtu_path = f'/data2/project_share/yangbo/NS_pipe_DD_reorder/data_U_train_50000/U_dataset_50000/Pipe_ns_{data_id}.vtu'
#     if not os.path.exists(vtu_path):
#         print(f"Skipping non-existent file: {vtu_path}")
#         continue

#     # 读取VTU文件
#     reader = XMLUnstructuredGridReader(FileName=[vtu_path])
#     reader.TimeArray = 'None'

#     # 生成插值点坐标
#     dx1, dy1, dx2, dy2, dx3, dy3, U_in = data_Pipe[data_id-1, :]
#     x_coords = (dx1 + dx2 - dx3) + i * (dx3 / (x_points_num - 1))
#     y_coords = dy1 - dy2 - j * (dy3 / (y_points_num - 1))
#     coords = np.column_stack((x_coords, y_coords, np.zeros_like(x_coords)))

#     # 创建VTK表格
#     vtk_table = vtk.vtkTable()
#     for name, data in zip(['X', 'Y', 'Z'], coords.T):
#         arr = vtk_np.numpy_to_vtk(data.copy(), deep=True)
#         arr.SetName(name)
#         vtk_table.AddColumn(arr)
    
#     # 转换为点集
#     source = TrivialProducer()
#     source.GetClientSideObject().SetOutput(vtk_table)
#     points = TableToPoints(Input=source, XColumn='X', YColumn='Y', ZColumn='Z')

#     # 执行数据重采样
#     resampled = ResampleWithDataset(SourceDataArrays=reader, DestinationMesh=points)
#     resampled.UpdatePipeline()  # 手动更新

#     # 保存结果
#     SaveData(f'/data2/project_share/yangbo/NS_pipe_DD_reorder/data_U_train_50000/temp/part4_label_reorder_{data_id}.csv', 
#              proxy=resampled,
#              PointDataArrays=['U', 'vtkValidPointMask', '压力', '速度场，x_分量', '速度场，y_分量'])

#     # 释放资源
#     Delete(reader)
#     Delete(resampled)
#     Delete(points)
#     Delete(source)
#     del reader, resampled, points, source

########################################################################################


########################################################################################
# #  part 2 and 3

# from paraview.simple import *
# import numpy as np
# import vtk
# import os
# from vtk.util import numpy_support as vtk_np

# paraview.simple._DisableFirstRenderCameraReset()

# data_Pipe = np.loadtxt('/data2/project_share/yangbo/NS_pipe_DD_reorder/data_U_skewing/data_skewing_10100.txt')

# x_points_num = 40
# y_points_num = 200
# i, j = np.mgrid[0:x_points_num, 0:y_points_num]
# i, j = i.ravel(), j.ravel()  # 矢量化索引

# for data_id in range(1, 10101): # 101
#     print(f'Processing data_id = {data_id}')

#     vtu_path = f'/data2/project_share/yangbo/NS_pipe_DD_reorder/data_U_skewing/U_dataset/Pipe_ns_{data_id}.vtu'
#     if not os.path.exists(vtu_path):
#         print(f"Skipping non-existent file: {vtu_path}")
#         continue

#     # 读取VTU文件
#     reader = XMLUnstructuredGridReader(FileName=[vtu_path])
#     reader.TimeArray = 'None'

#     # 生成插值点坐标
#     dx1, dy1, dx2, dy2, dx3, dy3, U0, U1 = data_Pipe[data_id-1, :]
#     x_coords = dx1 + i * (dx2 / (x_points_num - 1))
#     y_coords = dy1 - dy2 - dy3 + j * ((dy2 + dy3) / (y_points_num - 1))
#     coords = np.column_stack((x_coords, y_coords, np.zeros_like(x_coords)))

#     # 创建VTK表格
#     vtk_table = vtk.vtkTable()
#     for name, data in zip(['X', 'Y', 'Z'], coords.T):
#         arr = vtk_np.numpy_to_vtk(data.copy(), deep=True)
#         arr.SetName(name)
#         vtk_table.AddColumn(arr)
    
#     # 转换为点集
#     source = TrivialProducer()
#     source.GetClientSideObject().SetOutput(vtk_table)
#     points = TableToPoints(Input=source, XColumn='X', YColumn='Y', ZColumn='Z')

#     # 执行数据重采样
#     resampled = ResampleWithDataset(SourceDataArrays=reader, DestinationMesh=points)
#     resampled.UpdatePipeline()  # 手动更新

#     # 保存结果
#     SaveData(f'/data2/project_share/yangbo/NS_pipe_DD_reorder/data_U_skewing/temp/part23_label_{data_id}.csv', 
#              proxy=resampled,
#              PointDataArrays=['U', 'vtkValidPointMask', '压力', '速度场，x_分量', '速度场，y_分量'])

#     # 释放资源
#     Delete(reader)
#     Delete(resampled)
#     Delete(points)
#     Delete(source)
#     del reader, resampled, points, source
########################################################################################




##############################################
# BC, boundary condition
##############################################

########################################################################################
# #  bc_inlet

# from paraview.simple import *
# import numpy as np
# import vtk
# import os
# from vtk.util import numpy_support as vtk_np

# paraview.simple._DisableFirstRenderCameraReset()

# data_Pipe = np.loadtxt('/data2/project_share/yangbo/NS_pipe_DD_reorder/data_U_skewing/data_skewing_10100.txt')

# y_points_num = 40
# j = np.linspace(0, 39, 40)

# for data_id in range(1, 10101):  
#     print(f'Processing data_id = {data_id}')

#     vtu_path = f'/data2/project_share/yangbo/NS_pipe_DD_reorder/data_U_skewing/U_dataset/Pipe_ns_{data_id}.vtu'
#     if not os.path.exists(vtu_path):
#         print(f"Skipping non-existent file: {vtu_path}")
#         continue

#     # 读取VTU文件
#     reader = XMLUnstructuredGridReader(FileName=[vtu_path])
#     reader.TimeArray = 'None'

#     # 生成插值点坐标
#     dx1, dy1, dx2, dy2, dx3, dy3 = data_Pipe[data_id-1, :6]
#     x_coords = 0.0001
#     x_coords = np.tile(x_coords, 40)
#     y_coords = j * (dy1 / (y_points_num - 1))
#     coords = np.column_stack((x_coords, y_coords, np.zeros_like(x_coords)))

#     # 创建VTK表格
#     vtk_table = vtk.vtkTable()
#     for name, data in zip(['X', 'Y', 'Z'], coords.T):
#         arr = vtk_np.numpy_to_vtk(data.copy(), deep=True)
#         arr.SetName(name)
#         vtk_table.AddColumn(arr)
    
#     # 转换为点集
#     source = TrivialProducer()
#     source.GetClientSideObject().SetOutput(vtk_table)
#     points = TableToPoints(Input=source, XColumn='X', YColumn='Y', ZColumn='Z')

#     # 执行数据重采样
#     resampled = ResampleWithDataset(SourceDataArrays=reader, DestinationMesh=points)
#     resampled.UpdatePipeline()  # 手动更新

#     # 保存结果
#     SaveData(f'/data2/project_share/yangbo/NS_pipe_DD_reorder/data_U_skewing/temp/inlet_bc_IP_{data_id}.csv', 
#              proxy=resampled,
#              PointDataArrays=['U', 'vtkValidPointMask', '压力', '速度场，x_分量', '速度场，y_分量'])

#     # 释放资源
#     Delete(reader)
#     Delete(resampled)
#     Delete(points)
#     Delete(source)
#     del reader, resampled, points, source


########################################################################################
# #  bc1, part1 and part2

# from paraview.simple import *
# import numpy as np
# import vtk
# import os
# from vtk.util import numpy_support as vtk_np

# paraview.simple._DisableFirstRenderCameraReset()

# data_Pipe = np.loadtxt('/data2/project_share/yangbo/NS_pipe_DD_reorder/data_U_skewing/data_skewing_10100.txt')

# y_points_num = 40
# j = np.linspace(0, 39, 40)

# for data_id in range(1, 10101):  
#     print(f'Processing data_id = {data_id}')

#     vtu_path = f'/data2/project_share/yangbo/NS_pipe_DD_reorder/data_U_skewing/U_dataset/Pipe_ns_{data_id}.vtu'
#     if not os.path.exists(vtu_path):
#         print(f"Skipping non-existent file: {vtu_path}")
#         continue

#     # 读取VTU文件
#     reader = XMLUnstructuredGridReader(FileName=[vtu_path])
#     reader.TimeArray = 'None'

#     # 生成插值点坐标
#     dx1, dy1 = data_Pipe[data_id-1, :2]
#     x_coords = dx1
#     x_coords = np.tile(x_coords, 40)
#     y_coords = j * (dy1 / (y_points_num - 1))
#     coords = np.column_stack((x_coords, y_coords, np.zeros_like(x_coords)))

#     # 创建VTK表格
#     vtk_table = vtk.vtkTable()
#     for name, data in zip(['X', 'Y', 'Z'], coords.T):
#         arr = vtk_np.numpy_to_vtk(data.copy(), deep=True)
#         arr.SetName(name)
#         vtk_table.AddColumn(arr)
    
#     # 转换为点集
#     source = TrivialProducer()
#     source.GetClientSideObject().SetOutput(vtk_table)
#     points = TableToPoints(Input=source, XColumn='X', YColumn='Y', ZColumn='Z')

#     # 执行数据重采样
#     resampled = ResampleWithDataset(SourceDataArrays=reader, DestinationMesh=points)
#     resampled.UpdatePipeline()  # 手动更新

#     # 保存结果
#     SaveData(f'/data2/project_share/yangbo/NS_pipe_DD_reorder/data_U_skewing/temp/part1And2_bc_IP_{data_id}.csv', 
#              proxy=resampled,
#              PointDataArrays=['U', 'vtkValidPointMask', '压力', '速度场，x_分量', '速度场，y_分量'])

#     # 释放资源
#     Delete(reader)
#     Delete(resampled)
#     Delete(points)
#     Delete(source)
#     del reader, resampled, points, source


########################################################################################
# bc2, part1 and part2 and part3

# from paraview.simple import *
# import numpy as np
# import vtk
# import os
# from vtk.util import numpy_support as vtk_np

# paraview.simple._DisableFirstRenderCameraReset()

# data_Pipe = np.loadtxt('/data2/project_share/yangbo/NS_pipe_DD_reorder/data_U_skewing/data_skewing_10100.txt')

# x_points_num = 40
# j = np.linspace(0, 39, 40)

# for data_id in range(1, 10101):  
#     print(f'Processing data_id = {data_id}')

#     vtu_path = f'/data2/project_share/yangbo/NS_pipe_DD_reorder/data_U_skewing/U_dataset/Pipe_ns_{data_id}.vtu'
#     if not os.path.exists(vtu_path):
#         print(f"Skipping non-existent file: {vtu_path}")
#         continue

#     # 读取VTU文件
#     reader = XMLUnstructuredGridReader(FileName=[vtu_path])
#     reader.TimeArray = 'None'

#     # 生成插值点坐标
#     dx1, dy1, dx2, dy2 = data_Pipe[data_id-1, :4]
#     y_coords = 0
#     y_coords = np.tile(y_coords, 40)
#     x_coords = dx1 + j * (dx2 / (x_points_num - 1))
#     coords = np.column_stack((x_coords, y_coords, np.zeros_like(x_coords)))

#     # 创建VTK表格
#     vtk_table = vtk.vtkTable()
#     for name, data in zip(['X', 'Y', 'Z'], coords.T):
#         arr = vtk_np.numpy_to_vtk(data.copy(), deep=True)
#         arr.SetName(name)
#         vtk_table.AddColumn(arr)
    
#     # 转换为点集
#     source = TrivialProducer()
#     source.GetClientSideObject().SetOutput(vtk_table)
#     points = TableToPoints(Input=source, XColumn='X', YColumn='Y', ZColumn='Z')

#     # 执行数据重采样
#     resampled = ResampleWithDataset(SourceDataArrays=reader, DestinationMesh=points)
#     resampled.UpdatePipeline()  # 手动更新

#     # 保存结果
#     SaveData(f'/data2/project_share/yangbo/NS_pipe_DD_reorder/data_U_skewing/temp/part1And2And3_bc_IP_{data_id}.csv', 
#              proxy=resampled,
#              PointDataArrays=['U', 'vtkValidPointMask', '压力', '速度场，x_分量', '速度场，y_分量'])

#     # 释放资源
#     Delete(reader)
#     Delete(resampled)
#     Delete(points)
#     Delete(source)
#     del reader, resampled, points, source


########################################################################################
# #  bc3, part2 and part3 and part4

# from paraview.simple import *
# import numpy as np
# import vtk
# import os
# from vtk.util import numpy_support as vtk_np

# paraview.simple._DisableFirstRenderCameraReset()

# data_Pipe = np.loadtxt('/data2/project_share/yangbo/NS_pipe_DD_reorder/data_U_skewing/data_skewing_10100.txt')

# x_points_num = 40
# j = np.linspace(0, 39, 40)

# for data_id in range(1, 10101):  
#     print(f'Processing data_id = {data_id}')

#     vtu_path = f'/data2/project_share/yangbo/NS_pipe_DD_reorder/data_U_skewing/U_dataset/Pipe_ns_{data_id}.vtu'
#     if not os.path.exists(vtu_path):
#         print(f"Skipping non-existent file: {vtu_path}")
#         continue

#     # 读取VTU文件
#     reader = XMLUnstructuredGridReader(FileName=[vtu_path])
#     reader.TimeArray = 'None'

#     # 生成插值点坐标
#     dx1, dy1, dx2, dy2 = data_Pipe[data_id-1, :4]
#     y_coords = dy1 - dy2
#     y_coords = np.tile(y_coords, 40)
#     x_coords = dx1 + j * (dx2 / (x_points_num - 1))
#     coords = np.column_stack((x_coords, y_coords, np.zeros_like(x_coords)))

#     # 创建VTK表格
#     vtk_table = vtk.vtkTable()
#     for name, data in zip(['X', 'Y', 'Z'], coords.T):
#         arr = vtk_np.numpy_to_vtk(data.copy(), deep=True)
#         arr.SetName(name)
#         vtk_table.AddColumn(arr)
    
#     # 转换为点集
#     source = TrivialProducer()
#     source.GetClientSideObject().SetOutput(vtk_table)
#     points = TableToPoints(Input=source, XColumn='X', YColumn='Y', ZColumn='Z')

#     # 执行数据重采样
#     resampled = ResampleWithDataset(SourceDataArrays=reader, DestinationMesh=points)
#     resampled.UpdatePipeline()  # 手动更新

#     # 保存结果
#     SaveData(f'/data2/project_share/yangbo/NS_pipe_DD_reorder/data_U_skewing/temp/part2And3And4_bc_IP_{data_id}.csv', 
#              proxy=resampled,
#              PointDataArrays=['U', 'vtkValidPointMask', '压力', '速度场，x_分量', '速度场，y_分量'])

#     # 释放资源
#     Delete(reader)
#     Delete(resampled)
#     Delete(points)
#     Delete(source)
#     del reader, resampled, points, source




########################################################################################
# #  bc4, part3 and part4

# from paraview.simple import *
# import numpy as np
# import vtk
# import os
# from vtk.util import numpy_support as vtk_np

# paraview.simple._DisableFirstRenderCameraReset()

# data_Pipe = np.loadtxt('/data2/project_share/yangbo/NS_pipe_DD_reorder/data_U_skewing/data_skewing_10100.txt')

# y_points_num = 40
# j = np.linspace(0, 39, 40)

# for data_id in range(1, 10101):  
#     print(f'Processing data_id = {data_id}')

#     vtu_path = f'/data2/project_share/yangbo/NS_pipe_DD_reorder/data_U_skewing/U_dataset/Pipe_ns_{data_id}.vtu'
#     if not os.path.exists(vtu_path):
#         print(f"Skipping non-existent file: {vtu_path}")
#         continue

#     # 读取VTU文件
#     reader = XMLUnstructuredGridReader(FileName=[vtu_path])
#     reader.TimeArray = 'None'

#     # 生成插值点坐标
#     dx1, dy1, dx2, dy2, dx3, dy3 = data_Pipe[data_id-1, :6]
#     x_coords = dx1
#     x_coords = np.tile(x_coords, 40)
#     y_coords = dy1 - dy2 - j * (dy3 / (y_points_num - 1))
#     coords = np.column_stack((x_coords, y_coords, np.zeros_like(x_coords)))

#     # 创建VTK表格
#     vtk_table = vtk.vtkTable()
#     for name, data in zip(['X', 'Y', 'Z'], coords.T):
#         arr = vtk_np.numpy_to_vtk(data.copy(), deep=True)
#         arr.SetName(name)
#         vtk_table.AddColumn(arr)
    
#     # 转换为点集
#     source = TrivialProducer()
#     source.GetClientSideObject().SetOutput(vtk_table)
#     points = TableToPoints(Input=source, XColumn='X', YColumn='Y', ZColumn='Z')

#     # 执行数据重采样
#     resampled = ResampleWithDataset(SourceDataArrays=reader, DestinationMesh=points)
#     resampled.UpdatePipeline()  # 手动更新

#     # 保存结果
#     SaveData(f'/data2/project_share/yangbo/NS_pipe_DD_reorder/data_U_skewing/temp/part3And4_bc_IP_{data_id}.csv', 
#              proxy=resampled,
#              PointDataArrays=['U', 'vtkValidPointMask', '压力', '速度场，x_分量', '速度场，y_分量'])

#     # 释放资源
#     Delete(reader)
#     Delete(resampled)
#     Delete(points)
#     Delete(source)
#     del reader, resampled, points, source




########################################################################################
# #  bc_outlet

# from paraview.simple import *
# import numpy as np
# import vtk
# import os
# from vtk.util import numpy_support as vtk_np

# paraview.simple._DisableFirstRenderCameraReset()

# data_Pipe = np.loadtxt('/data2/project_share/yangbo/NS_pipe_DD_reorder/data_U_train_50000/data_50000.txt')

# y_points_num = 40
# j = np.linspace(0, 39, 40)

# for data_id in range(1, 50001):  
#     print(f'Processing data_id = {data_id}')

#     vtu_path = f'/data2/project_share/yangbo/NS_pipe_DD_reorder/data_U_train_50000/U_dataset_50000/Pipe_ns_{data_id}.vtu'
#     if not os.path.exists(vtu_path):
#         print(f"Skipping non-existent file: {vtu_path}")
#         continue

#     # 读取VTU文件
#     reader = XMLUnstructuredGridReader(FileName=[vtu_path])
#     reader.TimeArray = 'None'

#     # 生成插值点坐标
#     dx1, dy1, dx2, dy2, dx3, dy3 = data_Pipe[data_id-1, :6]
#     x_coords = dx1 + dx2 - dx3 + 0.0001
#     x_coords = np.tile(x_coords, 40)
#     y_coords = dy1 - dy2 - j * (dy3 / (y_points_num - 1))
#     coords = np.column_stack((x_coords, y_coords, np.zeros_like(x_coords)))

#     # 创建VTK表格
#     vtk_table = vtk.vtkTable()
#     for name, data in zip(['X', 'Y', 'Z'], coords.T):
#         arr = vtk_np.numpy_to_vtk(data.copy(), deep=True)
#         arr.SetName(name)
#         vtk_table.AddColumn(arr)
    
#     # 转换为点集
#     source = TrivialProducer()
#     source.GetClientSideObject().SetOutput(vtk_table)
#     points = TableToPoints(Input=source, XColumn='X', YColumn='Y', ZColumn='Z')

#     # 执行数据重采样
#     resampled = ResampleWithDataset(SourceDataArrays=reader, DestinationMesh=points)
#     resampled.UpdatePipeline()  # 手动更新

#     # 保存结果
#     SaveData(f'/data2/project_share/yangbo/NS_pipe_DD_reorder/data_U_train_50000/temp/outlet_bc_IP_{data_id}.csv', 
#              proxy=resampled,
#              PointDataArrays=['U', 'vtkValidPointMask', '压力', '速度场，x_分量', '速度场，y_分量'])

#     # 释放资源
#     Delete(reader)
#     Delete(resampled)
#     Delete(points)
#     Delete(source)
#     del reader, resampled, points, source