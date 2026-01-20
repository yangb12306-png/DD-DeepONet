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
Iter = 21
theta = 0.5

domain_1_x = 160
domain_1_y = 40

domain_2_x = 40
domain_2_y = 160

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


domain_2_net = MIONetCartesianProd_3(
    [6 , 512],
    [40, 512, 512, 512],
    [1 , 512, 512, 512],
    [2 , 512, 512, 512],
    "relu",
    "Glorot normal",
).to(device)
domain_2_net.load_state_dict(torch.load('/data2/yangbo/NS_pipe_DD_reorder/train_UZ_40400/part4/part4_model-500000.pt', 
                                         map_location=device, weights_only=False)['model_state_dict'])
domain_2_net.eval()


###########################################################################################################################################
# load trunk.txt

# domain_1 trunk
domain_1_trunk = np.loadtxt("/data2/project_share/yangbo/NS_pipe_DD_reorder/trunk_UZ/trunk_part1.txt", dtype=np.float32)
domain_1_trunk = torch.from_numpy(domain_1_trunk).to(device)

# domain_2 trunk
domain_2_trunk = np.loadtxt("/data2/project_share/yangbo/NS_pipe_DD_reorder/trunk_UZ/trunk_part2.txt", dtype=np.float32)
domain_2_trunk = torch.from_numpy(domain_2_trunk).to(device)

###########################################################################################################################################
data_set = np.loadtxt('/data2/project_share/yangbo/NS_pipe_Dirichlet/data/data_Pipe_filtered.txt')
v_ref = 0.1
v = data_set[:, 4]
indices = []
for i in range(data_set.shape[0]):
    if np.abs(v[i] - v_ref) < 1e-8:
        indices.append(i)   
    else:
        continue

dx1 = data_set[indices, 0]
dy1 = data_set[indices, 1]
dx2 = data_set[indices, 2]
dy2 = data_set[indices, 3]
U_in = data_set[indices, 4]

test_num = len(indices)


# neural networks input
length = dx1 + dx2
domain_1_branch_1 = np.column_stack([length, 1 / length, 1 / length**2, dy1, 1 / dy1, 1 / dy1**2])
domain_1_branch_1 = torch.from_numpy(domain_1_branch_1.astype(np.float32)).to(device)
domain_1_branch_2 = np.tile(U_in.reshape(-1, 1), (1, 40))
domain_1_branch_2 = torch.from_numpy(domain_1_branch_2.astype(np.float32)).to(device)
domain_1_branch_4 = dx2 / (dx1 + dx2)
domain_1_branch_4 = domain_1_branch_4.reshape(-1, 1)
domain_1_branch_4 = torch.from_numpy(domain_1_branch_4.astype(np.float32)).to(device)

domain_2_branch_1 = np.column_stack([dx2, 1 / dx2, 1 / dx2**2, dy2, 1 / dy2, 1 / dy2**2])
domain_2_branch_1 = torch.from_numpy(domain_2_branch_1.astype(np.float32)).to(device)
domain_2_branch_3 = dy1 / dy2
domain_2_branch_3 = domain_2_branch_3.reshape(-1, 1)
domain_2_branch_3 = torch.from_numpy(domain_2_branch_3.astype(np.float32)).to(device)

###########################################################################################################################################
# interface coordinate
coord_1 = torch.from_numpy(np.ones([40 * test_num, 2]).astype(np.float32)).to(device)
coord_2 = torch.from_numpy(np.ones([40 * test_num, 2]).astype(np.float32)).to(device)

y_interval = 1 / (40 - 1)
coord_1_reshaped = coord_1.view(-1, 40, 2)
coord_1_reshaped[:, :, 1] = torch.arange(40, device=device) * y_interval
coord_1 = coord_1_reshaped.view(-1, 2)
part1_x = dx1 / (dx1 + dx2)
part1_x = torch.from_numpy(part1_x.astype(np.float32)).to(device)
part1_x_expanded = part1_x.unsqueeze(1).repeat(1, 40)
coord_1[:, 0] = part1_x_expanded.flatten()

x_interval = 1 / (40 - 1)
coord_2_reshaped = coord_2.view(-1, 40, 2)
coord_2_reshaped[:, :, 0] = torch.arange(40, device=device) * x_interval
coord_2 = coord_2_reshaped.view(-1, 2)
part2_y = (dy2 - dy1) / dy2
part2_y = torch.from_numpy(part2_y.astype(np.float32)).to(device)
part2_y_expanded = part2_y.unsqueeze(1).repeat(1, 40)
coord_2[:, 1] = part2_y_expanded.flatten()


###########################################################################################################################################

filename_domain_1 = '/data2/yangbo/NS_pipe_DD_reorder/train_UZ_40400/DD_L/batch/part1.txt'
filename_domain_2 = '/data2/yangbo/NS_pipe_DD_reorder/train_UZ_40400/DD_L/batch/part2.txt'

with open(filename_domain_1, 'w') as f_1, open(filename_domain_2, 'w') as f_2:

    domain_1_branch_3 = torch.zeros((test_num, 40)).to(device)
    domain_2_branch_2 = torch.zeros((test_num, 40)).to(device)

    iteration = 1

    row_indices = torch.arange(test_num).repeat_interleave(40).to(device)
    col_indices = torch.arange(test_num * 40).to(device)

    #############################################################################

    while  iteration < Iter:
        
        # updata Boundary Condition 
        part2_branch2 = domain_1_net((domain_1_branch_1, domain_1_branch_2, domain_1_branch_3, domain_1_branch_4, coord_1))
        domain_2_branch_2 = part2_branch2[row_indices, col_indices].view(test_num, 40)
        del part2_branch2
        domain_2_branch_2 = torch.abs(domain_2_branch_2)
        domain_2_branch_2[:, [0, -1]] = 0

        part1_branch3 = domain_2_net((domain_2_branch_1, domain_2_branch_2, domain_2_branch_3, coord_2))
        domain_1_branch_3 = part1_branch3[row_indices, col_indices].view(test_num, 40)
        del part1_branch3
        domain_1_branch_3_temp = torch.abs(domain_1_branch_3)
        domain_1_branch_3 = theta * domain_1_branch_3 + (1 - theta) * domain_1_branch_3_temp
        domain_1_branch_3[:, [0, -1]] = 0

        
        iteration = iteration + 1
        # print('iteration = {i}'.format(i = iteration))

    #############################################################################

    start_time3 = time.perf_counter()

    # compute domain_1 
    domain_1_input = (domain_1_branch_1, domain_1_branch_2, domain_1_branch_3, domain_1_branch_4, domain_1_trunk)
    domain_1_result = domain_1_net(domain_1_input)
    domain_1_result = torch.abs(domain_1_result)
    # compute domain_2
    domain_2_input = (domain_2_branch_1, domain_2_branch_2, domain_2_branch_3, domain_2_trunk)
    domain_2_result = domain_2_net(domain_2_input)
    domain_2_result = torch.abs(domain_2_result)

    domain_1_result_np = domain_1_result.detach().cpu().numpy()
    np.savetxt(f_1, domain_1_result_np, fmt='%.8f', newline='\n')
    domain_2_result_np = domain_2_result.detach().cpu().numpy()
    np.savetxt(f_2, domain_2_result_np, fmt='%.8f', newline='\n')

    end_time3 = time.perf_counter()
    elapsed_time3 = end_time3 - start_time3
    print(f"Elapsed time(Saving the results): {elapsed_time2} seconds")
                  
###########################################################################################################################################
