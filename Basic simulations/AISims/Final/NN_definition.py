import numpy as np
import torch
import torch.nn as nn
import pickle
import copy
import json

##################################
## Functions to process image data
##################################
prefix_to_plot = "C:/Users/ruihe/GitHub/new-physics/results/NN_results20251003_021025"

with open(prefix_to_plot + "_parameters.json") as json_file:
    parameters = json.load(json_file)


def read_from_file_to_numpy_array(input_training_file, output_training_file):
    """Read data from multiple files into
    1) numpy array with input image with shape
            # images, # rows (24), # cols (12), 3 (object, vel x, vel y)
    2) numpy array with output image with shape
            # images, # rows (24), # cols (12), 2 (delta vel x, delta vel y)

    Args:
        filename1 (_type_): _description_
        filename2 (_type_): _description_

    Returns:
        numpy array with images
        numpy array with labels
    """
    with open(input_training_file, "rb") as filehandler:
        total_input_images = np.stack(pickle.load(filehandler), 0)
    with open(output_training_file, "rb") as filehandler:
        total_output_images = np.stack(pickle.load(filehandler), 0)

    total_output_images = (
        total_output_images[:, :, :, 1:3] - total_input_images[:, :, :, 1:3]
    )

    simulation_ids = [(int)(i / 29) for i in range(len(total_input_images))]
    simulation_times = [i % 29 for i in range(len(total_input_images))]
    return (total_input_images, total_output_images, simulation_ids, simulation_times)


def get_simulation(total_input_images, simulation_ids, simulation_times, simulation_id):

    times = [
        simulation_times[i]
        for (i, id) in enumerate(simulation_ids)
        if id == simulation_id
    ]
    images = [
        total_input_images[i]
        for (i, id) in enumerate(simulation_ids)
        if id == simulation_id
    ]
    return (times, images)


def convert_input_numpy_to_torch(input_numpy):
    """Convert input numpy array with shape
        # images, # rows (24), # cols (12), 3 (object, vel x, vel y)
    into torch input array wiuth shape
        # images, # rows (24), # cols (12), 3 (object, vel x, vel y)

    Returns:
        _type_: _description_
    """
    if isinstance(input_numpy, list) and isinstance(input_numpy[0], np.ndarray):
        # array of lists, need to stack first
        input_numpy = np.stack(input_numpy, 0)

    assert isinstance(
        input_numpy, np.ndarray
    ), "trying to convert something that is not a numpy array"
    assert input_numpy.ndim == 4, "numpy array must have 4 dimensions"
    output_torch = torch.tensor(input_numpy, dtype=torch.float32)
    return output_torch


def convert_output_torch_to_numpy_increment(output_torch):
    """Convert torch output array with shape
        # images, # rows (24), # cols (12), 2 (vel x, vel y)
    into numpy input array with shape
        # images, # rows (24), # cols (12), 3 (zeros, vel x, vel y)

    Returns:
        _type_: _description_
    """
    output2 = output_torch.detach().cpu().numpy()
    output_numpy = np.concatenate(
        [
            np.zeros(
                (
                    1,
                    parameters["image_number_rows"],
                    parameters["image_number_cols"],
                    1,
                )
            ),
            output2,
        ],
        3,
    )
    return output_numpy


def accumulate_numpy_data(previous_data, increment):
    """Accumulate image with shape
        # images, # rows (24), # cols (12), 3 (object, vel x, vel y)
    with increment with shape
        # images, # rows (24), # cols (12), 3 (zeros, vel x, vel y)
    to produce next image with shape
        # images, # rows (24), # cols (12), 3 (object, vel x, vel y)

    Args:
        previous_data (_type_): _description_
        increment (_type_): _description_

    Returns:
        _type_: _description_
    """
    return previous_data + increment


def send_numpy_data_to_image(axis, axis_img, input_numpy, title="", increment=False):
    """Convert numpy data with shape
        # rows (24), # cols (12), 3 (object, vel x, vel y)
    into an RGB image with shape
        # rows (12), # cols (24), 3 (object, vel x, vel y ; cliped to [0,1])

    Args:
        input_numpy (_type_): _description_

    Returns:
        _type_: _description_
    """
    assert isinstance(
        input_numpy, np.ndarray
    ), "trying to convert something that is not a numpy array"
    assert input_numpy.ndim == 3, "numpy array must have 3 dimensions"

    output_image = copy.deepcopy(input_numpy)
    # print(output_image[0:5, 0:5, 0:3])
    if increment:
        output_image[:, :, 1:3] = output_image[:, :, 1:3] * 10 + 0.5

    # swap  green and blue
    obstacle = output_image[:, :, 0]
    velocity_y = output_image[:, :, 1]
    velocity_x = output_image[:, :, 2]

    output_image = np.stack([obstacle, velocity_y, velocity_x], 2)

    empty_space = obstacle == 0
    output_image = output_image * np.stack([empty_space] * 3, 2)
    obstacle_color = np.array([72, 61, 139]) / 255
    output_image = output_image + np.stack(
        [
            obstacle_color[0] * obstacle,
            obstacle_color[1] * obstacle,
            obstacle_color[2] * obstacle,
        ],
        2,
    )

    output_image = np.clip(output_image, 0, 1)

    axis_img.set_data(np.swapaxes(output_image, 0, 1))
    axis.set_title(title, fontsize=5)
    axis.set_xticks([])
    axis.set_yticks([])


############
## Create NN
############


class Net_NoConv(nn.Module):
    def __init__(self, parameters):
        super(Net_NoConv, self).__init__()

        if False:
            self.fc1 = nn.Linear(
                in_features=parameters["number_nodes"][0],
                out_features=parameters["number_nodes"][1],
            )
            # Second fully connected layer
            self.fc2 = nn.Linear(
                in_features=parameters["number_nodes"][1],
                out_features=parameters["number_nodes"][2],
            )
            # Fourth fully connected layer
            self.fc3 = nn.Linear(
                in_features=parameters["number_nodes"][2],
                out_features=parameters["number_nodes"][3],
            )
            self.fc4 = nn.Linear(
                in_features=parameters["number_nodes"][3],
                out_features=parameters["number_nodes"][4],
            )
            self.fc5 = nn.Linear(
                in_features=parameters["number_nodes"][4],
                out_features=parameters["number_nodes"][5],
            )

            self.activation1 = nn.PReLU()  ### USE ACTIVATION WITHOUT VANISHING GRADIENT
            self.activation2 = nn.PReLU()  ### USE ACTIVATION WITHOUT VANISHING GRADIENT
            self.activation3 = nn.PReLU()  ### USE ACTIVATION WITHOUT VANISHING GRADIENT
            self.activation4 = nn.PReLU()  ### USE ACTIVATION WITHOUT VANISHING GRADIENT

        if True:
            # First fully connected layer
            self.fc = nn.ModuleList(
                [
                    nn.Linear(
                        in_features=parameters["number_nodes"][i],
                        out_features=parameters["number_nodes"][i + 1],
                    )
                    for i in range(len(parameters["number_nodes"]) - 1)
                ]
            )
            self.activation = nn.ModuleList(
                [
                    nn.PReLU()  ### USE ACTIVATION WITHOUT VANISHING GRADIENT
                    for i in range(len(parameters["number_nodes"]) - 2)
                ]
            )
        self.problem_parameters = parameters

    # x represents our data
    def forward(self, x):
        # Flatten x with start_dim=1
        x = torch.flatten(x, 1)
        if True:
            for i in range(len(self.activation)):
                x = self.fc[i](x)
                x = self.activation[i](x)
            output = self.fc[-1](x)
        else:
            x = self.fc1(x)
            x = self.activation1(x)
            x = self.fc2(x)
            x = self.activation2(x)
            x = self.fc3(x)
            x = self.activation3(x)
            x = self.fc4(x)
            x = self.activation4(x)
            output = self.fc5(x)

        output = torch.reshape(
            output,
            (
                len(output),
                self.problem_parameters["image_number_rows"],
                self.problem_parameters["image_number_cols"],
                2,
            ),
        )
        # output = F.log_softmax(x, dim=1)
        return output

    def get_weights(self):
        weights = copy.deepcopy(self.state_dict())
        return weights

    def save_weights(self, filename, weights):
        with open(filename, "wb") as file:
            pickle.dump(weights, file)

    def load_weights(self, filename):
        with open(filename, "rb") as filehandler:
            self.load_state_dict(pickle.load(filehandler))

    def predict_images(self, start_image):

        images = [start_image]
        for i in range(21):
            nn_output = self(convert_input_numpy_to_torch((images[i : i + 1])))
            nn_output_np = copy.deepcopy(
                convert_output_torch_to_numpy_increment(nn_output)[0]
            )
            images.append(accumulate_numpy_data(images[i], nn_output_np))
        return images
