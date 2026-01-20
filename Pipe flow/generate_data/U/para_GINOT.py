# trace generated using paraview version 5.12.0-RC3
#import paraview
#paraview.compatibility.major = 5
#paraview.compatibility.minor = 12

########################################################################################
# # 从vtk文件中提取子域边界和子域网格点

#### import the simple module from the paraview
from paraview.simple import *
import numpy as np
import os
#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

data_Pipe = np.loadtxt('/data2/project_share/yangbo/NS_pipe_DD_reorder/data_U_skewing/data_skewing_10100.txt')

for data_id in range(1, 10101):
    print(f'Processing data_id = {data_id}')

    vtu_path = f'/data2/project_share/yangbo/NS_pipe_DD_reorder/data_U_skewing/U_dataset/Pipe_ns_{data_id}.vtu'
    if not os.path.exists(vtu_path):
        print(f"Skipping non-existent file: {vtu_path}")
        continue

    reader = XMLUnstructuredGridReader(FileName=[vtu_path])
    reader.TimeArray = 'None'

    # 读取几何数据
    dx1, dy1, dx2, dy2, dx3, dy3, U0, U0 = data_Pipe[data_id-1, :]
    YleqC = dy1 - dy2

    # create a new 'Calculator'
    calculator1 = Calculator(registrationName='Calculator1', Input=reader)
    # Properties modified on calculator1
    calculator1.ResultArrayName = 'X'
    calculator1.Function = 'coordsX'

    # create a new 'Calculator'
    calculator2 = Calculator(registrationName='Calculator2', Input=calculator1)
    # Properties modified on calculator2
    calculator2.ResultArrayName = 'Y'
    calculator2.Function = 'coordsY'

    # create a new 'Threshold'
    threshold1 = Threshold(registrationName='Threshold1', Input=calculator2)
    # ✅ 指定用 POINTS 的 Y 数组做阈值
    threshold1.Scalars = ['POINTS', 'Y']
    threshold1.ThresholdMethod = 'Between'
    # Properties modified on threshold1
    threshold1.LowerThreshold = -1e+30
    threshold1.UpperThreshold = YleqC


    # save data
    SaveData(f'/data2/project_share/yangbo/NS_pipe_DD_reorder/data_U_skewing/Geo/Geo_{data_id}.csv', 
              proxy=threshold1, 
              PointDataArrays=['U', 'X', 'Y', '压力', '速度场，x_分量', '速度场，y_分量'],
              Precision=6)



    # create a new 'Delaunay 2D'
    delaunay2D1 = Delaunay2D(registrationName='Delaunay2D1', Input=threshold1)
    # create a new 'Feature Edges'
    featureEdges1 = FeatureEdges(registrationName='FeatureEdges1', Input=delaunay2D1)
    # Properties modified on featureEdges1
    featureEdges1.BoundaryEdges = 1
    featureEdges1.FeatureEdges = 0
    featureEdges1.ManifoldEdges = 0
    featureEdges1.NonManifoldEdges = 0


    # save data
    SaveData(f'/data2/project_share/yangbo/NS_pipe_DD_reorder/data_U_skewing/BC/_BC_{data_id}.csv', 
              proxy=featureEdges1, 
              PointDataArrays=['U', 'X', 'Y', '压力', '速度场，x_分量', '速度场，y_分量'],
              Precision=6)

    # 释放资源
    Delete(reader)
    Delete(calculator1)
    Delete(calculator2)
    Delete(threshold1)
    Delete(delaunay2D1)
    Delete(featureEdges1)

    del reader, calculator1, calculator2, threshold1, delaunay2D1, featureEdges1

########################################################################################