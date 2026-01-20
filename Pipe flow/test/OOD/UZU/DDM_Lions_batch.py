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


class MIONetCartesianProd_3(dde.maps.NN):
    """MIONet with three input functions for Cartesian product format."""

    def __init__(
        self,
        layer_sizes_branch1,
        layer_sizes_branch2,
        layer_sizes_branch3,####
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
            ) = self.activation_branch3 = self.activation_trunk = get(activation)#activations.get(activation)####
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
        x_loc = inputs[3]
        # Branch net to encode the input function
        y_func1 = self.branch1(x_func1)
        y_func2 = self.branch2(x_func2)
        y_func3 = self.branch3(x_func3)####
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
                x_merger = torch.mul(torch.mul(y_func1, y_func2), y_func3) ####
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

domain_U_net = MIONetCartesianProd_4(
    [6 , 512, 512, 512],
    [40, 512, 512, 512],
    [40, 512, 512, 512],
    [2 , 512, 512, 512],
    [2 , 512, 512, 512],
    "relu",
    "Glorot normal",
).to(device)
domain_U_net.load_state_dict(torch.load('/data2/yangbo/NS_pipe_DD_reorder/train_UZ_40400/part23_U/part_model-500000.pt', 
                                         map_location=device, weights_only=False)['model_state_dict'])
domain_U_net.eval()

domain_Z_net = MIONetCartesianProd_4(
    [6 , 512, 512, 512],
    [40, 512, 512, 512],
    [40, 512, 512, 512],
    [2 , 512, 512, 512],
    [2 , 512, 512, 512],
    "relu",
    "Glorot normal",
).to(device)
domain_Z_net.load_state_dict(torch.load('/data2/yangbo/NS_pipe_DD_reorder/train_UZ_40400/part23_Z/part_model-500000.pt', 
                                         map_location=device, weights_only=False)['model_state_dict'])
domain_Z_net.eval()

domain_5_net = MIONetCartesianProd_3(
    [6 , 512],
    [40, 512, 512, 512],
    [1 , 512, 512, 512],
    [2 , 512, 512, 512],
    "relu",
    "Glorot normal",
).to(device)
domain_5_net.load_state_dict(torch.load('/data2/yangbo/NS_pipe_DD_reorder/train_UZ_40400/part4/part4_model-500000.pt', 
                                         map_location=device, weights_only=False)['model_state_dict'])
domain_5_net.eval()


###########################################################################################################################################
# load trunk.txt

# domain_1 trunk
domain_1_trunk = np.loadtxt("/data2/project_share/yangbo/NS_pipe_DD_reorder/trunk_UZ/trunk_part1.txt", dtype=np.float32)
domain_1_trunk = torch.from_numpy(domain_1_trunk).to(device)

# domain_2 trunk
domain_UZ_trunk = np.loadtxt("/data2/project_share/yangbo/NS_pipe_DD_reorder/trunk_UZ/trunk_part23.txt", dtype=np.float32)
domain_UZ_trunk = torch.from_numpy(domain_UZ_trunk).to(device)

# domain_5 trunk
domain_5_trunk = np.loadtxt("/data2/project_share/yangbo/NS_pipe_DD_reorder/trunk_UZ/trunk_part2.txt", dtype=np.float32)
domain_5_trunk = torch.from_numpy(domain_5_trunk).to(device)

###########################################################################################################################################
data_set = np.loadtxt('/data2/project_share/yangbo/NS_pipe_DD_reorder/data_Gen/UZU/data_UZU_100.txt')
data_set = np.tile(data_set, (20, 1))
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
test_num = len(dx1)
BC_num = 40

# neural networks input
domain_1_branch_1 = np.column_stack([dx1 + dx2, 1 / (dx1 + dx2), 1 / (dx1 + dx2)**2, dy1, 1 / dy1, 1 / dy1**2])
domain_1_branch_1 = torch.from_numpy(domain_1_branch_1.astype(np.float32)).to(device)
domain_1_branch_2 = np.tile(inlet_vel.reshape(-1, 1), (1, 40))
domain_1_branch_2 = torch.from_numpy(domain_1_branch_2.astype(np.float32)).to(device)
domain_1_branch_4 = dx2 / (dx1 + dx2)
domain_1_branch_4 = domain_1_branch_4.reshape(-1, 1)
domain_1_branch_4 = torch.from_numpy(domain_1_branch_4.astype(np.float32)).to(device)

domain_2_branch_1 = np.column_stack([dx2, 1 / dx2, 1 / dx2**2, dy2 + dy3, 1 / (dy2 + dy3), 1 / (dy2 + dy3)**2])
domain_2_branch_1 = torch.from_numpy(domain_2_branch_1.astype(np.float32)).to(device)
domain_2_branch_4 = [dy1 / (dy2 + dy3), dy3 / (dy2 + dy3)]
domain_2_branch_4 = np.column_stack(domain_2_branch_4)
domain_2_branch_4 = torch.from_numpy(domain_2_branch_4.astype(np.float32)).to(device)

domain_3_branch_1 = np.column_stack([dy3, 1 / dy3, 1 / dy3**2, dx3 + dx4, 1 / (dx3 + dx4), 1 / (dx3 + dx4)**2])
domain_3_branch_1 = torch.from_numpy(domain_3_branch_1.astype(np.float32)).to(device)
domain_3_branch_4 = [dx2 / (dx3 + dx4), dx4 / (dx3 + dx4)]
domain_3_branch_4 = np.column_stack(domain_3_branch_4)
domain_3_branch_4 = torch.from_numpy(domain_3_branch_4.astype(np.float32)).to(device)

domain_4_branch_1 = np.column_stack([dx4, 1 / dx4, 1 / dx4**2, dy4 + dy5, 1 / (dy4 + dy5), 1 / (dy4 + dy5)**2])
domain_4_branch_1 = torch.from_numpy(domain_4_branch_1.astype(np.float32)).to(device)
domain_4_branch_4 = [dy3 / (dy4 + dy5), dy5 / (dy4 + dy5)]
domain_4_branch_4 = np.column_stack(domain_4_branch_4)
domain_4_branch_4 = torch.from_numpy(domain_4_branch_4.astype(np.float32)).to(device)

domain_5_branch_1 = np.column_stack([dy5, 1 / dy5, 1 / dy5**2, dx5, 1 / dx5, 1 / dx5**2])
domain_5_branch_1 = torch.from_numpy(domain_5_branch_1.astype(np.float32)).to(device)
domain_5_branch_3 = dx4 / dx5
domain_5_branch_3 = domain_5_branch_3.reshape(-1, 1)
domain_5_branch_3 = torch.from_numpy(domain_5_branch_3.astype(np.float32)).to(device)

###########################################################################################################################################
# interface coordinate
coord_part1 = torch.from_numpy(np.ones([40 * test_num, 2]).astype(np.float32)).to(device)
coord_part5 = torch.from_numpy(np.ones([40 * test_num, 2]).astype(np.float32)).to(device)
coord_part2_in = torch.from_numpy(np.ones([40 * test_num, 2]).astype(np.float32)).to(device)
coord_part2_out = torch.from_numpy(np.ones([40 * test_num, 2]).astype(np.float32)).to(device)
coord_part3_in = torch.from_numpy(np.ones([40 * test_num, 2]).astype(np.float32)).to(device)
coord_part3_out = torch.from_numpy(np.ones([40 * test_num, 2]).astype(np.float32)).to(device)
coord_part4_in = torch.from_numpy(np.ones([40 * test_num, 2]).astype(np.float32)).to(device)
coord_part4_out = torch.from_numpy(np.ones([40 * test_num, 2]).astype(np.float32)).to(device)


y_interval = 1 / (40 - 1)
coord_part1_reshaped = coord_part1.view(-1, 40, 2)
coord_part1_reshaped[:, :, 1] = torch.arange(40, device=device) * y_interval
coord_part1 = coord_part1_reshaped.view(-1, 2)
part1_x = dx1 / (dx1 + dx2)
part1_x = torch.from_numpy(part1_x.astype(np.float32)).to(device)
part1_x_expanded = part1_x.unsqueeze(1).repeat(1, 40)
coord_part1[:, 0] = part1_x_expanded.flatten()


x_interval = 1 / (40 - 1)
coord_part5_reshaped = coord_part5.view(-1, 40, 2)
coord_part5_reshaped[:, :, 0] = torch.arange(40, device=device) * x_interval
coord_part5 = coord_part5_reshaped.view(-1, 2)
part5_y = (dx5 - dx4) / dx5
part5_y = torch.from_numpy(part5_y.astype(np.float32)).to(device)
part5_y_expanded = part5_y.unsqueeze(1).repeat(1, 40)
coord_part5[:, 1] = part5_y_expanded.flatten()


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


x_interval = 1 / (40 - 1)
coord_part3_in_reshaped = coord_part3_in.view(-1, 40, 2)
coord_part3_in_reshaped[:, :, 0] = torch.arange(40, device=device) * x_interval
coord_part3_in = coord_part3_in_reshaped.view(-1, 2)
part3_y_in = (dx3 + dx4 - dx2) / (dx3 + dx4)
part3_y_in = torch.from_numpy(part3_y_in.astype(np.float32)).to(device)
part3_y_in_expanded = part3_y_in.unsqueeze(1).repeat(1, 40)
coord_part3_in[:, 1] = part3_y_in_expanded.flatten()

x_interval = 1 / (40 - 1)
coord_part3_out_reshaped = coord_part3_out.view(-1, 40, 2)
coord_part3_out_reshaped[:, :, 0] = torch.arange(40, device=device) * x_interval
coord_part3_out = coord_part3_out_reshaped.view(-1, 2)
part3_y_out = dx4 / (dx3 + dx4)
part3_y_out = torch.from_numpy(part3_y_out.astype(np.float32)).to(device)
part3_y_out_expanded = part3_y_out.unsqueeze(1).repeat(1, 40)
coord_part3_out[:, 1] = part3_y_out_expanded.flatten()






x_interval = 1 / (40 - 1)
coord_part4_in_reshaped = coord_part4_in.view(-1, 40, 2)
coord_part4_in_reshaped[:, :, 0] = torch.arange(40, device=device) * x_interval
coord_part4_in = coord_part4_in_reshaped.view(-1, 2)
part4_y_in = (dy4 + dy5 - dy3) / (dy4 + dy5)
part4_y_in = torch.from_numpy(part4_y_in.astype(np.float32)).to(device)
part4_y_in_expanded = part4_y_in.unsqueeze(1).repeat(1, 40)
coord_part4_in[:, 1] = part4_y_in_expanded.flatten()

x_interval = 1 / (40 - 1)
coord_part4_out_reshaped = coord_part4_out.view(-1, 40, 2)
coord_part4_out_reshaped[:, :, 0] = torch.arange(40, device=device) * x_interval
coord_part4_out = coord_part4_out_reshaped.view(-1, 2)
part4_y_out = dy5 / (dy4 + dy5)
part4_y_out = torch.from_numpy(part4_y_out.astype(np.float32)).to(device)
part4_y_out_expanded = part4_y_out.unsqueeze(1).repeat(1, 40)
coord_part4_out[:, 1] = part4_y_out_expanded.flatten()


###########################################################################################################################################

filename_domain_1 = '/data2/yangbo/NS_pipe_DD_reorder/train_UZ_40400/DD_23_UZU/batch_2000/part1.txt'
filename_domain_2 = '/data2/yangbo/NS_pipe_DD_reorder/train_UZ_40400/DD_23_UZU/batch_2000/part2.txt'
filename_domain_3 = '/data2/yangbo/NS_pipe_DD_reorder/train_UZ_40400/DD_23_UZU/batch_2000/part3.txt'
filename_domain_4 = '/data2/yangbo/NS_pipe_DD_reorder/train_UZ_40400/DD_23_UZU/batch_2000/part4.txt'
filename_domain_5 = '/data2/yangbo/NS_pipe_DD_reorder/train_UZ_40400/DD_23_UZU/batch_2000/part5.txt'

with open(filename_domain_1, 'w') as f_1, open(filename_domain_2, 'w') as f_2, open(filename_domain_3, 'w') as f_3, \
     open(filename_domain_4, 'w') as f_4, open(filename_domain_5, 'w') as f_5:

    domain_1_branch_3 = torch.zeros((test_num, 40)).to(device) 

    domain_2_branch_2 = torch.zeros((test_num, 40)).to(device)
    domain_2_branch_3 = torch.zeros((test_num, 40)).to(device)

    domain_3_branch_2 = torch.zeros((test_num, 40)).to(device)
    domain_3_branch_3 = torch.zeros((test_num, 40)).to(device)

    domain_4_branch_2 = torch.zeros((test_num, 40)).to(device)
    domain_4_branch_3 = torch.zeros((test_num, 40)).to(device)

    domain_5_branch_2 = torch.zeros((test_num, 40)).to(device) 

    iteration = 1

    row_indices = torch.arange(test_num).repeat_interleave(BC_num).to(device)
    col_indices = torch.arange(test_num * BC_num).to(device)

    #############################################################################
    
    while  iteration < Iter:
        
        # compute, part 1 and 5
        part2_branch2 = domain_1_net((domain_1_branch_1, domain_1_branch_2, domain_1_branch_3, \
                                      domain_1_branch_4, coord_part1)) # part1中计算
        domain_2_branch_2 = part2_branch2[row_indices, col_indices].view(test_num, BC_num)
        del part2_branch2
        domain_2_branch_2 = torch.abs(domain_2_branch_2)
        domain_2_branch_2[:, [0, -1]] = 0

        part4_branch3 = domain_5_net((domain_5_branch_1, domain_5_branch_2, domain_5_branch_3, coord_part5)) # part5中计算
        domain_4_branch_3 = part4_branch3[row_indices, col_indices].view(test_num, BC_num)
        del part4_branch3
        domain_4_branch_3 = torch.abs(domain_4_branch_3)
        domain_4_branch_3[:, [0, -1]] = 0

        # compute, part 2 and 4
        part3_branch2 = domain_U_net((domain_2_branch_1, domain_2_branch_2, domain_2_branch_3, domain_2_branch_4, coord_part2_out)) # part2中计算
        domain_3_branch_2 = part3_branch2[row_indices, col_indices].view(test_num, BC_num)
        del part3_branch2
        domain_3_branch_2 = torch.abs(domain_3_branch_2)
        domain_3_branch_2[:, [0, -1]] = 0

        part3_branch3 = domain_U_net((domain_4_branch_1, domain_4_branch_2, domain_4_branch_3, domain_4_branch_4, coord_part4_in)) # part4中计算
        domain_3_branch_3 = part3_branch3[row_indices, col_indices].view(test_num, BC_num)
        del part3_branch3
        domain_3_branch_3 = torch.abs(domain_3_branch_3)
        domain_3_branch_3[:, [0, -1]] = 0

        # update, part 3
        part2_branch3 = domain_Z_net((domain_3_branch_1, domain_3_branch_2, domain_3_branch_3, domain_3_branch_4, coord_part3_in)) # part3中计算
        domain_2_branch_3_temp = part2_branch3[row_indices, col_indices].view(test_num, BC_num)
        del part2_branch3
        domain_2_branch_3_temp = torch.abs(domain_2_branch_3_temp)
        domain_2_branch_3 = theta * domain_2_branch_3 + (1 - theta) * domain_2_branch_3_temp
        domain_2_branch_3[:, [0, -1]] = 0

        part4_branch2 = domain_Z_net((domain_3_branch_1, domain_3_branch_2, domain_3_branch_3, domain_3_branch_4, coord_part3_out)) # part3中计算
        domain_4_branch_2_temp = part4_branch2[row_indices, col_indices].view(test_num, BC_num)
        del part4_branch2
        domain_4_branch_2_temp = domain_4_branch_2_temp.flip(dims=[1])
        domain_4_branch_2_temp = torch.abs(domain_4_branch_2_temp)
        domain_4_branch_2 = theta * domain_4_branch_2 + (1 - theta) * domain_4_branch_2_temp
        domain_4_branch_2[:, [0, -1]] = 0

        # update, part 2 and 4
        part1_branch3 = domain_U_net((domain_2_branch_1, domain_2_branch_2, domain_2_branch_3, domain_2_branch_4, coord_part2_in)) # part2中计算
        domain_1_branch_3_temp = part1_branch3[row_indices, col_indices].view(test_num, BC_num)
        del part1_branch3
        domain_1_branch_3_temp = torch.abs(domain_1_branch_3_temp)
        domain_1_branch_3 = theta * domain_1_branch_3 + (1 - theta) * domain_1_branch_3_temp
        domain_1_branch_3[:, [0, -1]] = 0

        part5_branch2 = domain_U_net((domain_4_branch_1, domain_4_branch_2, domain_4_branch_3, domain_4_branch_4, coord_part4_out)) # part4中计算
        domain_5_branch_2_temp = part5_branch2[row_indices, col_indices].view(test_num, BC_num)
        del part5_branch2
        domain_5_branch_2_temp = torch.abs(domain_5_branch_2_temp)
        domain_5_branch_2 = theta * domain_5_branch_2 + (1 - theta) * domain_5_branch_2_temp
        domain_5_branch_2[:, [0, -1]] = 0

        
        iteration = iteration + 1
        # print('iteration = {i}'.format(i = iteration))

    #############################################################################

    end_time1 = time.perf_counter()
    elapsed_time1 = end_time1 - start_time1
    print(f"Elapsed time(Iterative computation time): {elapsed_time1} seconds")

    start_time2 = time.perf_counter()

    # compute domain_1 
    domain_1_input = (domain_1_branch_1, domain_1_branch_2, domain_1_branch_3, domain_1_branch_4, domain_1_trunk)
    domain_1_result = domain_1_net(domain_1_input)
    # compute domain_2
    domain_2_input = (domain_2_branch_1, domain_2_branch_2, domain_2_branch_3, domain_2_branch_4, domain_UZ_trunk)
    domain_2_result = domain_U_net(domain_2_input)
    # compute domain_3
    domain_3_input = (domain_3_branch_1, domain_3_branch_2, domain_3_branch_3, domain_3_branch_4, domain_UZ_trunk)
    domain_3_result = domain_Z_net(domain_3_input)
    # compute domain_4
    domain_4_input = (domain_4_branch_1, domain_4_branch_2, domain_4_branch_3, domain_4_branch_4, domain_UZ_trunk)
    domain_4_result = domain_U_net(domain_4_input)
    # compute domain_5
    domain_5_input = (domain_5_branch_1, domain_5_branch_2, domain_5_branch_3, domain_5_trunk)
    domain_5_result = domain_5_net(domain_5_input)

    domain_1_result_np = domain_1_result.detach().cpu().numpy()
    np.savetxt(f_1, domain_1_result_np, fmt='%.8f', newline='\n')
    domain_2_result_np = domain_2_result.detach().cpu().numpy()
    np.savetxt(f_2, domain_2_result_np, fmt='%.8f', newline='\n')
    domain_3_result_np = domain_3_result.detach().cpu().numpy()
    np.savetxt(f_3, domain_3_result_np, fmt='%.8f', newline='\n')
    domain_4_result_np = domain_4_result.detach().cpu().numpy()
    np.savetxt(f_4, domain_4_result_np, fmt='%.8f', newline='\n')
    domain_5_result_np = domain_5_result.detach().cpu().numpy()
    np.savetxt(f_5, domain_5_result_np, fmt='%.8f', newline='\n')

    end_time2 = time.perf_counter()
    elapsed_time2 = end_time2 - start_time2
    print(f"Elapsed time(Saving the results): {elapsed_time2} seconds")
                  
###########################################################################################################################################
