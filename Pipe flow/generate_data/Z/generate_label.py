# #######################################################################################

# # generate lable.txt(part1)
# import os
# import pandas as pd
# import numpy as np

# # 设置路径
# output_path = '/data2/project_share/yangbo/NS_pipe_DD_reorder/data_Z_skewing/label_part23.txt'
# input_folder = '/data2/project_share/yangbo/NS_pipe_DD_reorder/data_Z_skewing/temp/'

# # 初始化输出文件（清空/创建）
# with open(output_path, 'w') as f:
#     pass  # 创建文件或清空内容

# # 列名（如果读取时指定）
# column_names = ['U', 'p', 'u', 'v', 'non', 'Points_0', 'Points_1', 'Points_2']

# # 开始读取数据
# for data_id in range(1, 10101):
#     print("data_id = ", data_id)
#     filename = f'part23_label_{data_id}.csv'
#     filepath = os.path.join(input_folder, filename)

#     if not os.path.exists(filepath):
#         print(f"file is not exist, skip this loop: data_id = {data_id}")
#         continue

#     try:
#         # 读取 CSV
#         df = pd.read_csv(filepath, skiprows=1, names=column_names, delimiter=',')

#         # 提取 'U' 列数据并转置为行向量
#         u_values = df['U'].to_numpy().reshape(1, -1)

#         if np.isnan(u_values).any():
#             raise ValueError("Error: u_values contains NaN values")
#         if np.isinf(u_values).any():
#             raise ValueError("Error: u_values contains Inf values")
#         if u_values.size == 0:
#             raise ValueError("Error: u_values is empty")
#         if len(df['U']) == 0:
#             raise ValueError("Error: df['U'] is empty")

#         # 写入输出文件（追加模式，制表符分隔）
#         with open(output_path, 'a') as f:
#             np.savetxt(f, u_values, delimiter='\t', fmt='%.6f')

#     except Exception as e:
#         raise ValueError(f"[错误] data_id = {data_id} 读取失败: {e}")


# #######################################################################################


# # # generate BC.txt

# import numpy as np
# import pandas as pd
# import os

# # 定义列名
# IP_column_names = [
#     'U', 'p', 'u', 'v', 'non', 
#     'Points_0', 'Points_1', 'Points_2'
# ]

# Dirichlet_path = '/data2/project_share/yangbo/NS_pipe_DD_reorder/data_Z_skewing/Dirichlet_3and4.txt'

# # 初始化输出文件（清空/创建）
# with open(Dirichlet_path, 'w') as f:
#     pass  # 创建文件或清空内容

# for data_id in range(1, 10101):  # data_id从1到30000
#     # 生成文件路径
#     IP_file = f'/data2/project_share/yangbo/NS_pipe_DD_reorder/data_Z_skewing/temp/part3And4_bc_IP_{data_id}.csv'

#     # 检查文件是否存在
#     if not (os.path.exists(IP_file)):
#         print(f'file is not exist, skip this loop: data_id = {data_id}')
#         continue

#     try:
#         # 读取插值数据
#         IP_data = pd.read_csv(
#             IP_file, 
#             skiprows=1, 
#             header=None, 
#             names=IP_column_names,
#             delimiter=','
#         )
#     except Exception as e:
#         print(f'read data wrong, data_id={data_id}: {e}')
#         raise ValueError(f'read data wrong, data_id={data_id}: {e}')
#         continue
    
#     # 提取边界值
#     boundary_value = IP_data['U'].to_numpy().reshape(1, -1)

#     # 写入输出文件（追加模式，制表符分隔）
#     with open(Dirichlet_path, 'a') as f:
#         np.savetxt(f, boundary_value, delimiter='\t', fmt='%.6f')
