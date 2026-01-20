# DeepONet
import sys 
import deepxde as dde
import numpy as np
import torch
import matplotlib.pyplot as plt
import time
import h5py
from deepxde.data.data import Data
from deepxde.data.sampler import BatchSampler
from deepxde.nn.activations import get
from deepxde.nn.pytorch.fnn import FNN

# GINOT package
import matplotlib.pyplot as plt
import numpy as np
import torch
from models.NS_GINOT import LoadModel
from models import configs
from tqdm import tqdm
import time
import pyvista as pv
import json

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import TensorDataset, DataLoader, Dataset
import os
import pickle
from sklearn.model_selection import train_test_split
from typing import Union
import time

# DeepONet structure
class MIONetCartesianProd_4(dde.maps.NN):
    """MIONet with three input functions for Cartesian product format."""

    def __init__(
        self,
        layer_sizes_branch1,
        layer_sizes_branch2,
        layer_sizes_branch3,####
        layer_sizes_branch4,####
        layer_sizes_trunk,
        activation,
        kernel_initializer,
        regularization=None,
        trunk_last_activation=False,
        merge_operation="mul",
        layer_sizes_merger=None,
        output_merge_operation="mul",
        layer_sizes_output_merger=None,
    ):
        super().__init__()

        if isinstance(activation, dict):
            self.activation_branch1 = activations.get(activation["branch1"])
            self.activation_branch2 = activations.get(activation["branch2"])
            self.activation_branch3 = activations.get(activation["branch3"])####
            self.activation_trunk = activations.get(activation["trunk"])
        else:
            self.activation_branch1 = (
                self.activation_branch2
            ) = self.activation_branch3 \
              = self.activation_branch4 = self.activation_trunk = get(activation)#activations.get(activation)####
        if callable(layer_sizes_branch1[1]):
            # User-defined network
            self.branch1 = layer_sizes_branch1[1]
        else:
            # Fully connected network
            self.branch1 = FNN(
                layer_sizes_branch1, self.activation_branch1, kernel_initializer
            )
        if callable(layer_sizes_branch2[1]):
            # User-defined network
            self.branch2 = layer_sizes_branch2[1]
        else:
            # Fully connected network
            self.branch2 = FNN(
                layer_sizes_branch2, self.activation_branch2, kernel_initializer
            )
        if callable(layer_sizes_branch3[1]):          ####
            # User-defined network                    ####
            self.branch3 = layer_sizes_branch3[1]     ####    
        else:                                         ####
            # Fully connected network                 ####
            self.branch3 = FNN(                       ####
                layer_sizes_branch3, self.activation_branch3, kernel_initializer
            )
        if callable(layer_sizes_branch4[1]):          ####
            # User-defined network                    ####
            self.branch4 = layer_sizes_branch4[1]     ####    
        else:                                         ####
            # Fully connected network                 ####
            self.branch4 = FNN(                       ####
                layer_sizes_branch4, self.activation_branch4, kernel_initializer
            )
        if layer_sizes_merger is not None:
            self.activation_merger = activations.get(activation["merger"])
            if callable(layer_sizes_merger[1]):
                # User-defined network
                self.merger = layer_sizes_merger[1]
            else:
                # Fully connected network
                self.merger = FNN(
                    layer_sizes_merger, self.activation_merger, kernel_initializer
                )
        else:
            self.merger = None
        if layer_sizes_output_merger is not None:
            self.activation_output_merger = activations.get(activation["output merger"])
            if callable(layer_sizes_output_merger[1]):
                # User-defined network
                self.output_merger = layer_sizes_output_merger[1]
            else:
                # Fully connected network
                self.output_merger = FNN(
                    layer_sizes_output_merger,
                    self.activation_output_merger,
                    kernel_initializer,
                )
        else:
            self.output_merger = None
        self.trunk = FNN(layer_sizes_trunk, self.activation_trunk, kernel_initializer)
        self.b = torch.nn.parameter.Parameter(torch.tensor(0.0))
        self.regularizer = regularization
        self.trunk_last_activation = trunk_last_activation
        self.merge_operation = merge_operation
        self.output_merge_operation = output_merge_operation

    def forward(self, inputs):
        x_func1 = inputs[0]
        x_func2 = inputs[1]
        x_func3 = inputs[2]####
        x_func4 = inputs[3]####
        x_loc = inputs[4]
        # Branch net to encode the input function
        y_func1 = self.branch1(x_func1)
        y_func2 = self.branch2(x_func2)
        y_func3 = self.branch3(x_func3)####
        y_func4 = self.branch4(x_func4)####
        if self.merge_operation == "cat":
            x_merger = torch.cat((y_func1, y_func2), 1)
        else:
            if y_func1.shape[-1] != y_func2.shape[-1] or y_func1.shape[-1] != y_func3.shape[-1]:
                raise AssertionError(
                    "Output sizes of branch1 net and branch2 net do not match."
                )
            if self.merge_operation == "add":
                x_merger = y_func1 + y_func2
            elif self.merge_operation == "mul":
                x_merger = torch.mul(torch.mul(torch.mul(y_func1, y_func2), y_func3), y_func4) ####
            else:
                raise NotImplementedError(
                    f"{self.merge_operation} operation to be implimented"
                )
        # Optional merger net
        if self.merger is not None:
            y_func = self.merger(x_merger)
        else:
            y_func = x_merger
        # Trunk net to encode the domain of the output function
        if self._input_transform is not None:
            x_loc = self._input_transform(x_loc)
        y_loc = self.trunk(x_loc)
        if self.trunk_last_activation:
            y_loc = self.activation_trunk(y_loc)
        # Dot product
        if y_func.shape[-1] != y_loc.shape[-1]:
            raise AssertionError(
                "Output sizes of merger net and trunk net do not match."
            )
        # output merger net
        if self.output_merger is None:
            y = torch.einsum("ip,jp->ij", y_func, y_loc)
        else:
            y_func = y_func[:, None, :]
            y_loc = y_loc[None, :]
            if self.output_merge_operation == "mul":
                y = torch.mul(y_func, y_loc)
            elif self.output_merge_operation == "add":
                y = y_func + y_loc
            elif self.output_merge_operation == "cat":
                y_func = y_func.repeat(1, y_loc.shape[1], 1)
                y_loc = y_loc.repeat(y_func.shape[0], 1, 1)
                y = torch.cat((y_func, y_loc), dim=2)
            shape0 = y.shape[0]
            shape1 = y.shape[1]
            y = y.reshape(shape0 * shape1, -1)
            y = self.output_merger(y)
            y = y.reshape(shape0, shape1)
        # Add bias
        y += self.b
        if self._output_transform is not None:
            y = self._output_transform(inputs, y)
        return y

###########################################################################################################################################
start_time1 = time.perf_counter()
# parameter setting

# device = torch.device('cpu')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.set_grad_enabled(False)
Iter = 31
theta = 0.2

domain_1_x = 160
domain_1_y = 40

domain_2_x = 40
domain_2_y = 200

domain_3_x = 40
domain_3_y = 160

###########################################################################################################################################
# load well-training model
domain_1_net = MIONetCartesianProd_4(
    [6 , 512, 512, 512],
    [40, 512, 512, 512],
    [40, 512, 512, 512],
    [1 , 512, 512, 512],
    [2 , 512, 512, 512],
    "relu",
    "Glorot normal",
).to(device)
domain_1_net.load_state_dict(torch.load('/data2/yangbo/NS_pipe_DD_reorder/train_UZ_40400/part1/part_model-500000.pt', 
                                         map_location=device, weights_only=False)['model_state_dict'])
domain_1_net.eval()

domain_2_net = MIONetCartesianProd_4(
    [6 , 512, 512, 512],
    [40, 512, 512, 512],
    [40, 512, 512, 512],
    [2 , 512, 512, 512],
    [2 , 512, 512, 512],
    "relu",
    "Glorot normal",
).to(device)
domain_2_net.load_state_dict(torch.load('/data2/yangbo/NS_pipe_DD_reorder/train_UZ_40400/part23_Z/part_model-500000.pt', 
                                         map_location=device, weights_only=False)['model_state_dict'])
domain_2_net.eval()

###########################################################################################################################################
# load trunk.txt

# domain_1 trunk
domain_1_trunk = np.loadtxt("/data2/project_share/yangbo/NS_pipe_DD_reorder/trunk_UZ/trunk_part1.txt", dtype=np.float32)
domain_1_trunk = torch.from_numpy(domain_1_trunk).to(device)

# domain_2 trunk
domain_2_trunk = np.loadtxt("/data2/project_share/yangbo/NS_pipe_DD_reorder/trunk_UZ/trunk_part23.txt", dtype=np.float32)
domain_2_trunk = torch.from_numpy(domain_2_trunk).to(device)

###########################################################################################################################################
data_set = np.loadtxt('/data2/project_share/yangbo/NS_pipe_DD_reorder/data_Z_skewing/data_skewing_10073.txt')
dx1 = data_set[:, 0]
dy1 = data_set[:, 1]
dx2 = data_set[:, 2]
dy2 = data_set[:, 3]
dx3 = data_set[:, 4]
dy3 = data_set[:, 5]

fraction_train = 0.8
sample_num = len(dx1)
N_train = int( sample_num * fraction_train )
train_case = np.array(list(range(N_train))) 
test_case =  np.array(list(range(N_train, sample_num))) 

dx1 = dx1[ test_case ]
dy1 = dy1[ test_case ]
dx2 = dx2[ test_case ]
dy2 = dy2[ test_case ]
dx3 = dx3[ test_case ]
dy3 = dy3[ test_case ]

domain_1_branch_2_skew = np.loadtxt("/data2/project_share/yangbo/NS_pipe_DD_reorder/data_Z_skewing/Dirichlet_inlet.txt")
domain_1_branch_2_skew = torch.from_numpy(domain_1_branch_2_skew.astype(np.float32)).to(device)
domain_1_branch_2_skew = domain_1_branch_2_skew[test_case, :]

data_set = np.loadtxt('/data2/project_share/yangbo/NS_pipe_DD_reorder/data_Z_train_10100/data_10100_pipe.txt')
dx1_01 = data_set[:, 0]
dy1_01 = data_set[:, 1]
dx2_01 = data_set[:, 2]
dy2_01 = data_set[:, 3]
dx3_01 = data_set[:, 4]
dy3_01 = data_set[:, 5]
inlet_vel_01 = data_set[:, 6]

fraction_train = 0.8
sample_num = len(dx1_01)
N_train = int( sample_num * fraction_train )
train_case = np.array(list(range(N_train))) 
test_case =  np.array(list(range(N_train, sample_num))) 


dx1_01 = dx1_01[ test_case ]
dy1_01 = dy1_01[ test_case ]
dx2_01 = dx2_01[ test_case ]
dy2_01 = dy2_01[ test_case ]
dx3_01 = dx3_01[ test_case ]
dy3_01 = dy3_01[ test_case ]
inlet_vel_01 = inlet_vel_01[test_case]

dx1 = np.concatenate([dx1_01, dx1])
dy1 = np.concatenate([dy1_01, dy1])
dx2 = np.concatenate([dx2_01, dx2])
dy2 = np.concatenate([dy2_01, dy2])
dx3 = np.concatenate([dx3_01, dx3])
dy3 = np.concatenate([dy3_01, dy3])

test_num = len(dx1)
BC_num = 40

# neural networks input
domain_1_branch_1 = np.column_stack([dx1 + dx2, 1 / (dx1 + dx2), 1 / (dx1 + dx2)**2, dy1, 1 / dy1, 1 / dy1**2])
domain_1_branch_1 = torch.from_numpy(domain_1_branch_1.astype(np.float32)).to(device)

domain_1_branch_2_01 = np.tile(inlet_vel_01.reshape(-1, 1), (1, 40))
domain_1_branch_2_01 = torch.from_numpy(domain_1_branch_2_01.astype(np.float32)).to(device)
domain_1_branch_2 = torch.cat([domain_1_branch_2_01, domain_1_branch_2_skew], dim=0)
domain_1_branch_4 = dx2 / (dx1 + dx2)
domain_1_branch_4 = domain_1_branch_4.reshape(-1, 1)
domain_1_branch_4 = torch.from_numpy(domain_1_branch_4.astype(np.float32)).to(device)

domain_2_branch_1 = np.column_stack([dx2, 1 / dx2, 1 / dx2**2, dy2 + dy3, 1 / (dy2 + dy3), 1 / (dy2 + dy3)**2])
domain_2_branch_1 = torch.from_numpy(domain_2_branch_1.astype(np.float32)).to(device)
domain_2_branch_4 = [dy1 / (dy2 + dy3), dy3 / (dy2 + dy3)]
domain_2_branch_4 = np.column_stack(domain_2_branch_4)
domain_2_branch_4 = torch.from_numpy(domain_2_branch_4.astype(np.float32)).to(device)

###########################################################################################################################################
# interface coordinate
coord_part1 = torch.from_numpy(np.ones([40 * test_num, 2]).astype(np.float32)).to(device)
coord_part2_in = torch.from_numpy(np.ones([40 * test_num, 2]).astype(np.float32)).to(device)
coord_part2_out = torch.from_numpy(np.ones([40 * test_num, 2]).astype(np.float32)).to(device)

y_interval = 1 / (40 - 1)
coord_part1_reshaped = coord_part1.view(-1, 40, 2)
coord_part1_reshaped[:, :, 1] = torch.arange(40, device=device) * y_interval
coord_part1 = coord_part1_reshaped.view(-1, 2)
part1_x = dx1 / (dx1 + dx2)
part1_x = torch.from_numpy(part1_x.astype(np.float32)).to(device)
part1_x_expanded = part1_x.unsqueeze(1).repeat(1, 40)
coord_part1[:, 0] = part1_x_expanded.flatten()

x_interval = 1 / (40 - 1)
coord_part2_in_reshaped = coord_part2_in.view(-1, 40, 2)
coord_part2_in_reshaped[:, :, 0] = torch.arange(40, device=device) * x_interval
coord_part2_in = coord_part2_in_reshaped.view(-1, 2)
part2_y_in = (dy2 + dy3 - dy1) / (dy2 + dy3)
part2_y_in = torch.from_numpy(part2_y_in.astype(np.float32)).to(device)
part2_y_in_expanded = part2_y_in.unsqueeze(1).repeat(1, 40)
coord_part2_in[:, 1] = part2_y_in_expanded.flatten()

x_interval = 1 / (40 - 1)
coord_part2_out_reshaped = coord_part2_out.view(-1, 40, 2)
coord_part2_out_reshaped[:, :, 0] = torch.arange(40, device=device) * x_interval
coord_part2_out = coord_part2_out_reshaped.view(-1, 2)
part2_y_out = dy3 / (dy2 + dy3)
part2_y_out = torch.from_numpy(part2_y_out.astype(np.float32)).to(device)
part2_y_out_expanded = part2_y_out.unsqueeze(1).repeat(1, 40)
coord_part2_out[:, 1] = part2_y_out_expanded.flatten()
###########################################################################################################################################

# initial value
domain_1_branch_3 = torch.zeros((test_num, 40)).to(device) # iter

domain_2_branch_2 = torch.zeros((test_num, 40)).to(device)
domain_2_branch_3 = torch.zeros((test_num, 40)).to(device)

domain_3_branch_2 = torch.zeros((test_num, 40)).to(device) # iter

###########################################################################################################################################
# GINOT configs
SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))

DATA_FILEBASE = f"{SCRIPT_PATH}/data"
print(DATA_FILEBASE)
PADDING_VALUE = -1000
padding_value = -1000

# load config
configs_NS = configs.NS_GINOT_configs()

filebase = configs_NS["filebase"]
trunk_args = configs_NS["trunk_args"]
branch_args = configs_NS["branch_args"]
NS_ginot = LoadModel(filebase, branch_args, trunk_args)
# print("filebase: ", filebase)
# print("trunk_args: ", trunk_args)
# print("branch_args: ", branch_args)




# # load test data
bs_test=512
seed=42
shuffle_loader=False

start = time.time()
# load data
data_file = f"{DATA_FILEBASE}/NS/inference/Z_test.pkl"
with open(data_file, "rb") as f:
    data = pickle.load(f)
vertices_all = data['vertices']
U_all = data['U']
points_cloud_all = data['points_cloud']
BC_percentage_all = data['BC_percentage']
Dirichlet_all = data['Drichlet']

test_xyt = [torch.tensor(v) for v in vertices_all]
test_S = [torch.tensor(u) for u in U_all]
test_pc = [torch.tensor(pc) for pc in points_cloud_all]
test_BC_p  = torch.tensor(np.array(BC_percentage_all))
test_Dirich = torch.tensor(np.array(Dirichlet_all))
test_ids = np.arange(test_Dirich.shape[0])

data_file = f"{DATA_FILEBASE}/NS/inference/TC_xyt_Z.pkl"
with open(data_file, "rb") as f:
    data = pickle.load(f)
vertices_all_TC = data['vertices_34']
test_xyt_TC = [torch.tensor(v) for v in vertices_all_TC]


test_dataset_TC = configs.ListDataset(
    (test_BC_p, domain_3_branch_2, test_pc, test_xyt_TC , test_S, torch.tensor(test_ids)))

# padded dataloader
def pad_collate_fn(batch):
    BC_p_batch = torch.stack([item[0] for item in batch])
    Dirich_batch = torch.stack([item[1] for item in batch])
    pc_batch = [item[2] for item in batch]  # Extract pc (variable-length)
    xyt_batch = [item[3]
                    for item in batch]  # Extract xyt (variable-length)
    S = [item[4] for item in batch]  # Extract S (variable-length)
    sample_ids = torch.stack([item[5] for item in batch])
    # y_batch = torch.stack([item[1] for item in batch])  # Extract and stack y (fixed-length)
    # Pad sequences
    pc_padded = pad_sequence(
        pc_batch, batch_first=True, padding_value=padding_value)
    xyt_padded = pad_sequence(
        xyt_batch, batch_first=True, padding_value=padding_value)
    S_padded = pad_sequence(S, batch_first=True,
                            padding_value=padding_value)
    return BC_p_batch, Dirich_batch, pc_padded, xyt_padded, S_padded, sample_ids

test_dataloader = DataLoader(test_dataset_TC, batch_size=bs_test, shuffle=False,
                                collate_fn=pad_collate_fn)

print(f"Data loading time: {time.time()-start:.2f} s")



def predict_NS(data_loader, NS_ginot, BC_3_branch_2):
    y_pred = []
    y_true = []
    NS_ginot.eval()
    id = 0
    with torch.no_grad():
        for data in data_loader:
            BC_p = data[0].to(device)
            # Dirich = data[1].to(device)
            pc = data[2].to(device)
            xyt = data[3].to(device)
            y_true_batch = data[4].to(device)
            sample_ids = data[5].to(device)

            start = id * sample_ids.shape[0]
            end   = start + sample_ids.shape[0]
            BC3   = BC_3_branch_2[start:end, :].to(device)

            mask = (y_true_batch != padding_value)
            pred = NS_ginot(BC_p, BC3, xyt, pc)
            if xyt.shape[1] == 40:
                y_pred.append(pred.detach().cpu())
            else:
                pred = [x[i].view(-1, *i.shape[1:]).cpu().detach().numpy()
                        for x, i in zip(pred, mask)]
                y_true_batch = [x[i].view(-1, *i.shape[1:]).cpu().detach().numpy()
                                for x, i in zip(y_true_batch, mask)]

            
                y_pred = y_pred+pred
                y_true = y_true+y_true_batch
            id = id + 1

    return y_pred, y_true


end_time1 = time.perf_counter()
elapsed_time1 = end_time1 - start_time1
print(f"Elapsed time(Iterative computation time): {elapsed_time1} seconds")



###########################################################################################################################################

filename_domain_1 = '/data2/yangbo/GINOT/GINOT-main/Z/part1.txt'
filename_domain_2 = '/data2/yangbo/GINOT/GINOT-main/Z/part2.txt'

with open(filename_domain_1, 'w') as f_1, open(filename_domain_2, 'w') as f_2:

    iteration = 1

    row_indices = torch.arange(test_num).repeat_interleave(BC_num).to(device)
    col_indices = torch.arange(test_num * BC_num).to(device)

    #############################################################################
    start_time2 = time.perf_counter()
    while  iteration < Iter:

        # updata Boundary Condition 
        part2_branch2 = domain_1_net((domain_1_branch_1, domain_1_branch_2, domain_1_branch_3, \
                                      domain_1_branch_4, coord_part1)) # part1中计算
        domain_2_branch_2 = part2_branch2[row_indices, col_indices].view(test_num, BC_num)
        del part2_branch2
        domain_2_branch_2 = torch.abs(domain_2_branch_2)
        domain_2_branch_2[:, [0, -1]] = 0

        domain_2_branch_3, _true = predict_NS(test_dataloader, NS_ginot, domain_3_branch_2) # part3
        domain_2_branch_3 = torch.cat(domain_2_branch_3, dim=0).to(device)
        domain_2_branch_3 = torch.abs(domain_2_branch_3)
        domain_2_branch_3[:, [0, -1]] = 0


        part1_branch3 = domain_2_net((domain_2_branch_1, domain_2_branch_2, domain_2_branch_3, \
                                            domain_2_branch_4, coord_part2_in)) # part2中计算
        domain_1_branch_3_temp = part1_branch3[row_indices, col_indices].view(test_num, BC_num)
        del part1_branch3
        domain_1_branch_3_temp= torch.abs(domain_1_branch_3_temp)
        domain_1_branch_3 = theta * domain_1_branch_3 + (1 - theta) * domain_1_branch_3_temp
        domain_1_branch_3[:, [0, -1]] = 0

        part3_branch2 = domain_2_net((domain_2_branch_1, domain_2_branch_2, domain_2_branch_3, \
                                            domain_2_branch_4, coord_part2_out)) # part2中计算
        domain_3_branch_2_temp = part3_branch2[row_indices, col_indices].view(test_num, BC_num)
        del part3_branch2
        domain_3_branch_2_temp = torch.abs(domain_3_branch_2_temp)
        domain_3_branch_2_temp = domain_3_branch_2_temp.flip(dims=[1]) 
        domain_3_branch_2 = theta * domain_3_branch_2 + (1 - theta) * domain_3_branch_2_temp
        domain_3_branch_2[:, [0, -1]] = 0
        
        iteration = iteration + 1
        # print('iteration = {i}'.format(i = iteration))

#############################################################################

    end_time2 = time.perf_counter()
    elapsed_time2 = end_time2 - start_time2
    print(f"Elapsed time(Iterative computation time): {elapsed_time2} seconds")

    start_time3 = time.perf_counter()

    # compute domain_1 
    domain_1_input = (domain_1_branch_1, domain_1_branch_2, domain_1_branch_3, domain_1_branch_4, domain_1_trunk)
    domain_1_result = domain_1_net(domain_1_input)
    # compute domain_2
    domain_2_input = (domain_2_branch_1, domain_2_branch_2, domain_2_branch_3, domain_2_branch_4, domain_2_trunk)
    domain_2_result = domain_2_net(domain_2_input)

    domain_1_result_np = domain_1_result.detach().cpu().numpy()
    np.savetxt(f_1, domain_1_result_np, fmt='%.8f', newline='\n')
    domain_2_result_np = domain_2_result.detach().cpu().numpy()
    np.savetxt(f_2, domain_2_result_np, fmt='%.8f', newline='\n')

    # compute domain_3
    test_dataset = configs.ListDataset(
            (test_BC_p, domain_3_branch_2, test_pc, test_xyt, test_S, torch.tensor(test_ids)))
    test_dataloader = DataLoader(test_dataset, batch_size=bs_test, shuffle=False,
                                collate_fn=pad_collate_fn)
    domain_3_result, true_3 = predict_NS(test_dataloader, NS_ginot, domain_3_branch_2)


    def save_ragged_1d_txt(path, ragged, sep=" ", fmt="%.8f"):
        """
        ragged: list，每个元素是一条 1D 向量（长度可不同），类型可为 list/np.ndarray/torch.Tensor
        """
        with open(path, "w") as f:
            for row in ragged:
                # torch -> cpu numpy
                if hasattr(row, "detach"):
                    row = row.detach().cpu().numpy()

                row = np.asarray(row).reshape(-1)  # 保证 1D
                if row.size == 0:
                    f.write("\n")
                    continue

                # 格式化输出
                f.write(sep.join(fmt % x for x in row.tolist()) + "\n")

    save_ragged_1d_txt("/data2/yangbo/GINOT/GINOT-main/Z/part3_pred.txt", domain_3_result, sep=" ", fmt="%.8f")
    save_ragged_1d_txt("/data2/yangbo/GINOT/GINOT-main/Z/part3_true.txt", true_3, sep=" ", fmt="%.8f")


    end_time3 = time.perf_counter()
    elapsed_time3 = end_time3 - start_time3
    print(f"Elapsed time(Saving the results): {elapsed_time3} seconds")
                  
###########################################################################################################################################
