# # generate .pkl
import os
import pickle
import pandas as pd
import numpy as np

# 设置路径
data_Pipe = np.loadtxt('/data2/project_share/yangbo/NS_pipe_DD_reorder/data_Z_skewing/data_skewing_10100.txt')

Drichlet_value = np.loadtxt('/data2/project_share/yangbo/NS_pipe_DD_reorder/data_Z_skewing/Dirichlet_2and3and4_part4.txt')

input_folder_Geo = '/data2/project_share/yangbo/NS_pipe_DD_reorder/data_Z_skewing/Geo'
input_folder_BC = '/data2/project_share/yangbo/NS_pipe_DD_reorder/data_Z_skewing/BC'

# 列名（如果读取时指定）
column_names = ['u', 'v', 'U', 'p', 'X', 'Y', 'Points_0', 'Points_1', 'Points_2']

out_pkl_path = '/data2/project_share/yangbo/NS_pipe_DD_reorder/data_Z_skewing/NS_Z_skewing.pkl'
if os.path.exists(out_pkl_path):
    os.remove(out_pkl_path) 

samples = {
    "points_cloud": [],
    "BC_percentage": [],
    "Drichlet": [],
    "vertices": [],     
    "U": [],   
}

# 开始读取数据
_count = 0
for data_id in range(1, 10101):
    print("data_id = ", data_id)

    filename_BC = f'_BC_{data_id}.csv'
    filepath_BC = os.path.join(input_folder_BC, filename_BC)

    filename_Geo = f'Geo_{data_id}.csv'
    filepath_Geo = os.path.join(input_folder_Geo, filename_Geo)

    dx1, dy1, dx2, dy2, dx3, dy3, U0, U1 = data_Pipe[data_id-1, :]

    if not os.path.exists(filepath_BC):
        print(f"file_BC is not exist, skip this loop: data_id = {data_id}")
        continue
    if not os.path.exists(filepath_Geo):
        print(f"file_BC is not exist, skip this loop: data_id = {data_id}")
        continue

    try:

        # 读取 BC.CSV
        df_BC = pd.read_csv(filepath_BC, skiprows=1, names=column_names, delimiter=',')
        coord_BC_X = df_BC['X'].to_numpy().reshape(-1)
        coord_BC_Y = df_BC['Y'].to_numpy().reshape(-1)
        points_interface = np.column_stack([
            np.linspace(dx1 + dx2, dx1, 51),
            np.full(51, dy1 - dy2),        
        ])
        coord_BC_X = np.concatenate([coord_BC_X, points_interface[:, 0]])
        coord_BC_Y = np.concatenate([coord_BC_Y, points_interface[:, 1]])
        coord_BC = np.stack([coord_BC_X, coord_BC_Y], axis=0).T.astype(np.float32)


        # 读取 Geo.CSV
        df_Geo = pd.read_csv(filepath_Geo, skiprows=1, names=column_names, delimiter=',')
        U_Geo = df_Geo['U'].to_numpy().reshape(-1).astype(np.float32)
        coord_Geo_X = df_Geo['X'].to_numpy().reshape(-1)
        coord_Geo_Y = df_Geo['Y'].to_numpy().reshape(-1)
        coord_Geo = np.stack([coord_Geo_X, coord_Geo_Y], axis=0).T.astype(np.float32)

        BC_percentage = np.array([dx2 / dx3, dx2, dx3], dtype=np.float32)

        Drichlet_ = Drichlet_value[_count].astype(np.float32)

        samples["points_cloud"].append(coord_BC)
        samples["BC_percentage"].append(BC_percentage)
        samples["Drichlet"].append(Drichlet_)
        samples["vertices"].append(coord_Geo)
        samples["U"].append(U_Geo)

        if np.isnan(U_Geo).any():
            raise ValueError("Error: U_Geo contains NaN values")
        if np.isinf(U_Geo).any():
            raise ValueError("Error: U_Geo contains Inf values")
        if coord_BC_X.size == 0:
            raise ValueError("Error: coord_BC_X is empty")
        if coord_Geo_X.size == 0:
            raise ValueError("Error: coord_Geo_X is empty")
        if len(df_Geo['U']) == 0:
            raise ValueError("Error: df_Geo['U'] is empty")
        _count = _count + 1

    except Exception as e:
        raise ValueError(f"[错误] data_id = {data_id} 读取失败: {e}")
    
with open(out_pkl_path, "wb") as f:
    pickle.dump(samples, f, protocol=pickle.HIGHEST_PROTOCOL)
