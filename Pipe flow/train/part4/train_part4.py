# 3个branch算例，对应的输入数据结构也要更改
import deepxde as dde
import matplotlib.pyplot as plt
import numpy as np
import h5py
import torch
from deepxde.data.data import Data
from deepxde.data.sampler import BatchSampler
from deepxde.nn.activations import get
from deepxde.nn.pytorch.fnn import FNN

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



class QuadrupleCartesianProd_3(Data):
    """Cartesian Product input data format for MIONet architecture.

    This dataset can be used with the network ``MIONetCartesianProd`` for operator
    learning.

    Args:
        X_train: A tuple of three NumPy arrays. The first element has the shape (`N1`,
            `dim1`), the second element has the shape (`N1`, `dim2`), and the third
            element has the shape (`N2`, `dim3`).
        y_train: A NumPy array of shape (`N1`, `N2`).
    """

    def __init__(self, X_train, y_train, X_test, y_test):
        if (
            len(X_train[0]) * len(X_train[3]) != y_train.size
            or len(X_train[1]) * len(X_train[3]) != y_train.size
            or len(X_train[2]) * len(X_train[3]) != y_train.size ####
            or len(X_train[0]) != len(X_train[1])
            or len(X_train[0]) != len(X_train[2])
        ):
            raise ValueError(
                "The training dataset does not have the format of Cartesian product."
            )
        if (
            len(X_test[0]) * len(X_test[3]) != y_test.size
            or len(X_test[1]) * len(X_test[3]) != y_test.size
            or len(X_test[2]) * len(X_test[3]) != y_test.size
            or len(X_test[0]) != len(X_test[1])
            or len(X_test[0]) != len(X_test[2])
        ):
            raise ValueError(
                "The testing dataset does not have the format of Cartesian product."
            )
        self.train_x, self.train_y = X_train, y_train
        self.test_x, self.test_y = X_test, y_test

        self.branch_sampler = BatchSampler(len(X_train[0]), shuffle=True)
        self.trunk_sampler = BatchSampler(len(X_train[2]), shuffle=True)

    def losses(self, targets, outputs, loss_fn, inputs, model, aux=None):
        return loss_fn(targets, outputs)

    def train_next_batch(self, batch_size=None):
        if batch_size is None:
            return self.train_x, self.train_y
        if not isinstance(batch_size, (tuple, list)):
            indices = self.branch_sampler.get_next(batch_size)
            return (
                self.train_x[0][indices],
                self.train_x[1][indices],
                self.train_x[2][indices],
                self.train_x[3],
            ), self.train_y[indices]
        indices_branch = self.branch_sampler.get_next(batch_size[0])
        indices_trunk = self.trunk_sampler.get_next(batch_size[1])
        return (
            self.train_x[0][indices_branch],
            self.train_x[1][indices_branch],
            self.train_x[2][indices_branch],
            self.train_x[3][indices_trunk],
        ), self.train_y[indices_branch, indices_trunk]

    def test(self):
        return self.test_x, self.test_y

###############################################################################

torch.cuda.set_device(1)

data_set_U_01 = np.loadtxt("/data2/project_share/yangbo/NS_pipe_DD_reorder/data_U_train_10100/data_10100_pipe.txt")
dx1_U_01 = data_set_U_01[:, 0]
dy1_U_01 = data_set_U_01[:, 1]
dx2_U_01 = data_set_U_01[:, 2]
dy2_U_01 = data_set_U_01[:, 3]
dx3_U_01 = data_set_U_01[:, 4]
dy3_U_01 = data_set_U_01[:, 5]
inlet_vel_U_01 = data_set_U_01[:, 6]

data_set_Z_01 = np.loadtxt("/data2/project_share/yangbo/NS_pipe_DD_reorder/data_Z_train_10100/data_10100_pipe.txt")
dx1_Z_01 = data_set_Z_01[:, 0]
dy1_Z_01 = data_set_Z_01[:, 1]
dx2_Z_01 = data_set_Z_01[:, 2]
dy2_Z_01 = data_set_Z_01[:, 3]
dx3_Z_01 = data_set_Z_01[:, 4]
dy3_Z_01 = data_set_Z_01[:, 5]
inlet_vel_Z_01 = data_set_Z_01[:, 6]


data_set_U_skew = np.loadtxt("/data2/project_share/yangbo/NS_pipe_DD_reorder/data_U_skewing/data_skewing_10032.txt")
dx1_U_skew = data_set_U_skew[:, 0]
dy1_U_skew = data_set_U_skew[:, 1]
dx2_U_skew = data_set_U_skew[:, 2]
dy2_U_skew = data_set_U_skew[:, 3]
dx3_U_skew = data_set_U_skew[:, 4]
dy3_U_skew = data_set_U_skew[:, 5]

data_set_Z_skew = np.loadtxt("/data2/project_share/yangbo/NS_pipe_DD_reorder/data_Z_skewing/data_skewing_10073.txt")
dx1_Z_skew = data_set_Z_skew[:, 0]
dy1_Z_skew = data_set_Z_skew[:, 1]
dx2_Z_skew = data_set_Z_skew[:, 2]
dy2_Z_skew = data_set_Z_skew[:, 3]
dx3_Z_skew = data_set_Z_skew[:, 4]
dy3_Z_skew = data_set_Z_skew[:, 5]

fraction_train = 0.8
sample_num_U_01 = len(inlet_vel_U_01)
N_train_U_01 = int( sample_num_U_01 * fraction_train )
train_case_U_01 = np.array(list(range(N_train_U_01))) 
test_case_U_01 =  np.array(list(range(N_train_U_01, sample_num_U_01)))

sample_num_Z_01 = len(inlet_vel_Z_01)
N_train_Z_01 = int( sample_num_Z_01 * fraction_train )
train_case_Z_01 = np.array(list(range(N_train_Z_01))) 
test_case_Z_01 =  np.array(list(range(N_train_Z_01, sample_num_Z_01))) 

sample_num_U_skew = len(dx1_U_skew)
N_train_U_skew = int( sample_num_U_skew * fraction_train )
train_case_U_skew = np.array(list(range(N_train_U_skew))) 
test_case_U_skew =  np.array(list(range(N_train_U_skew, sample_num_U_skew)))

sample_num_Z_skew = len(dx1_Z_skew)
N_train_Z_skew = int( sample_num_Z_skew * fraction_train )
train_case_Z_skew = np.array(list(range(N_train_Z_skew))) 
test_case_Z_skew =  np.array(list(range(N_train_Z_skew, sample_num_Z_skew))) 

label_U_01 = np.loadtxt("/data2/project_share/yangbo/NS_pipe_DD_reorder/data_U_train_10100/label_part4.txt")
label_U_train_01 = label_U_01[train_case_U_01, :]
label_U_test_01 = label_U_01[test_case_U_01, :]
label_Z_01 = np.loadtxt("/data2/project_share/yangbo/NS_pipe_DD_reorder/data_Z_train_10100/label_part4.txt")
label_Z_train_01 = label_Z_01[train_case_Z_01, :]
label_Z_test_01 = label_Z_01[test_case_Z_01, :]
label_U_skew = np.loadtxt("/data2/project_share/yangbo/NS_pipe_DD_reorder/data_U_skewing/label_part4.txt")
label_U_train_skew = label_U_skew[train_case_U_skew, :]
label_U_test_skew = label_U_skew[test_case_U_skew, :]
label_Z_skew = np.loadtxt("/data2/project_share/yangbo/NS_pipe_DD_reorder/data_Z_skewing/label_part4.txt")
label_Z_train_skew = label_Z_skew[train_case_Z_skew, :]
label_Z_test_skew = label_Z_skew[test_case_Z_skew, :]

label_train = np.vstack((label_U_train_01, label_Z_train_01, label_U_train_skew, label_Z_train_skew))
label_test = np.vstack((label_U_test_01, label_Z_test_01, label_U_test_skew, label_Z_test_skew))


branch_1_U_01 = [dy3_U_01, 1 / dy3_U_01, 1 / dy3_U_01, dx3_U_01, 1 / dx3_U_01, 1 / dx3_U_01**2]
branch_1_U_01 = np.column_stack(branch_1_U_01)
branch_1_train_U_01 = branch_1_U_01[train_case_U_01, :]
branch_1_test_U_01 = branch_1_U_01[test_case_U_01, :]
branch_1_Z_01 = [dy3_Z_01, 1 / dy3_Z_01, 1 / dy3_Z_01**2, dx3_Z_01, 1 / dx3_Z_01, 1 / dx3_Z_01**2]
branch_1_Z_01 = np.column_stack(branch_1_Z_01)
branch_1_train_Z_01 = branch_1_Z_01[train_case_Z_01, :]
branch_1_test_Z_01 = branch_1_Z_01[test_case_Z_01, :]
branch_1_U_skew = [dy3_U_skew, 1 / dy3_U_skew, 1 / dy3_U_skew**2, dx3_U_skew, 1 / dx3_U_skew, 1 / dx3_U_skew**2]
branch_1_U_skew = np.column_stack(branch_1_U_skew)
branch_1_train_U_skew = branch_1_U_skew[train_case_U_skew, :]
branch_1_test_U_skew = branch_1_U_skew[test_case_U_skew, :]
branch_1_Z_skew = [dy3_Z_skew, 1 / dy3_Z_skew, 1 / dy3_Z_skew**2, dx3_Z_skew, 1 / dx3_Z_skew, 1 / dx3_Z_skew**2]
branch_1_Z_skew = np.column_stack(branch_1_Z_skew)
branch_1_train_Z_skew = branch_1_Z_skew[train_case_Z_skew, :]
branch_1_test_Z_skew = branch_1_Z_skew[test_case_Z_skew, :]

branch_1_train = np.vstack((branch_1_train_U_01, branch_1_train_Z_01, branch_1_train_U_skew, branch_1_train_Z_skew))
branch_1_test = np.vstack((branch_1_test_U_01, branch_1_test_Z_01, branch_1_test_U_skew, branch_1_test_Z_skew))


branch_2_U_01 = np.loadtxt("/data2/project_share/yangbo/NS_pipe_DD_reorder/data_U_train_10100/Dirichlet_2And3And4.txt")
branch_2_train_U_01 = branch_2_U_01[train_case_U_01, :]
branch_2_test_U_01 = branch_2_U_01[test_case_U_01, :]
branch_2_Z_01 = np.loadtxt("/data2/project_share/yangbo/NS_pipe_DD_reorder/data_Z_train_10100/Dirichlet_2and3and4_part4.txt")
branch_2_train_Z_01 = branch_2_Z_01[train_case_Z_01, :]
branch_2_test_Z_01 = branch_2_Z_01[test_case_Z_01, :]
branch_2_U_skew = np.loadtxt("/data2/project_share/yangbo/NS_pipe_DD_reorder/data_U_skewing/Dirichlet_2and3and4.txt")
branch_2_train_U_skew = branch_2_U_skew[train_case_U_skew, :]
branch_2_test_U_skew = branch_2_U_skew[test_case_U_skew, :]
branch_2_Z_skew = np.loadtxt("/data2/project_share/yangbo/NS_pipe_DD_reorder/data_Z_skewing/Dirichlet_2and3and4_part4.txt")
branch_2_train_Z_skew = branch_2_Z_skew[train_case_Z_skew, :]
branch_2_test_Z_skew = branch_2_Z_skew[test_case_Z_skew, :]

branch_2_train = np.vstack((branch_2_train_U_01, branch_2_train_Z_01, branch_2_train_U_skew, branch_2_train_Z_skew))
branch_2_test = np.vstack((branch_2_test_U_01, branch_2_test_Z_01, branch_2_test_U_skew, branch_2_test_Z_skew))


branch_3_U_01 = dx2_U_01 / dx3_U_01
branch_3_U_01 = branch_3_U_01.reshape(-1, 1)
branch_3_train_U_01 = branch_3_U_01[train_case_U_01, :]
branch_3_test_U_01 = branch_3_U_01[test_case_U_01, :]
branch_3_Z_01 = dx2_Z_01 / dx3_Z_01
branch_3_Z_01 = branch_3_Z_01.reshape(-1, 1)
branch_3_train_Z_01 = branch_3_Z_01[train_case_Z_01, :]
branch_3_test_Z_01 = branch_3_Z_01[test_case_Z_01, :]
branch_3_U_skew = dx2_U_skew / dx3_U_skew
branch_3_U_skew = branch_3_U_skew.reshape(-1, 1)
branch_3_train_U_skew = branch_3_U_skew[train_case_U_skew, :]
branch_3_test_U_skew = branch_3_U_skew[test_case_U_skew, :]
branch_3_Z_skew = dx2_Z_skew / dx3_Z_skew
branch_3_Z_skew = branch_3_Z_skew.reshape(-1, 1)
branch_3_train_Z_skew = branch_3_Z_skew[train_case_Z_skew, :]
branch_3_test_Z_skew = branch_3_Z_skew[test_case_Z_skew, :]

branch_3_train = np.vstack((branch_3_train_U_01, branch_3_train_Z_01, branch_3_train_U_skew, branch_3_train_Z_skew))
branch_3_test = np.vstack((branch_3_test_U_01, branch_3_test_Z_01, branch_3_test_U_skew, branch_3_test_Z_skew))


trunk = np.loadtxt("/data2/project_share/yangbo/NS_pipe_DD_reorder/trunk_UZ/trunk_part2.txt") 

X_train = (branch_1_train.astype(np.float32), branch_2_train.astype(np.float32), 
           branch_3_train.astype(np.float32), trunk.astype(np.float32))
y_train = label_train.astype(np.float32)
 
X_test = (branch_1_test.astype(np.float32), branch_2_test.astype(np.float32), 
          branch_3_test.astype(np.float32), trunk.astype(np.float32))
y_test = label_test.astype(np.float32)

data = QuadrupleCartesianProd_3(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)

# Choose a network
m1 = 6 # branch1 input
m2 = 40 # branch2 input
m3 = 1 # branch3 input
m4 = 2 # trunk input
net = MIONetCartesianProd_3(
    [m1, 512],
    [m2, 512, 512, 512],
    [m3, 512, 512, 512],
    [m4, 512, 512, 512],
    "relu",
    "Glorot normal",
)

# Define a Model
model = dde.Model(data, net)

# Compile and Train
model.compile("adam", lr=0.0002, metrics=["MAE"], decay=("exponential", np.power(0.999990, 1/8))) 
losshistory, train_state = model.train(iterations=500000, batch_size = 64, model_save_path='/data2/yangbo/NS_pipe_DD_reorder/train_UZ_40400/part4/part4_model')

# Plot the loss trajectory
dde.utils.plot_loss_history(losshistory)
plt.savefig('/data2/yangbo/NS_pipe_DD_reorder/train_UZ_40400/part4/loss_curve.png')  

