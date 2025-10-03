import ImageLoader as im
import numpy as np
import sklearn
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
import copy
import pickle
import time
import datetime
import json

prefix_for_saves = (
    f"C:/Users/ruihe/GitHub/new-physics/results/NN_results"
    + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
)
print(f'files saved as "{prefix_for_saves}_..."')
parameters = {
    "input_training_file": f"C:/Users/ruihe/GitHub/new-physics/Training_data_pickle_compressed/total_images.pkl",
    "output_training_file": f"C:/Users/ruihe/GitHub/new-physics/Training_data_pickle_compressed/total_image_labels.pkl",
    "sample_simulation_file": f"C:/Users/ruihe/GitHub/new-physics/Training_data_pickle_compressed/Sample_sim/RGB_images_3619.pkl",
    "image_number_rows": 24,
    "image_number_cols": 12,
    "number_nodes": [864, 864 * 8, 864 * 4, 864, 864, 576],
    "percentage_training_images": 0.9,
    "starting_gamma": 1e-5,
    "starting_threshold": 1e-5,
    "n_epochs": 500,
    "batch_size": 600,
    "torch_seed": random.randint(1, 1000),
    "prefix_for_saves": prefix_for_saves,
}

matplotlib.rc("xtick", labelsize=6)
matplotlib.rc("ytick", labelsize=6)

start_time = time.time()

# sample_sim = 0
# count_gamma = 1

##################################
## Functions to process image data
##################################


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
    return (total_input_images, total_output_images)


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
    output_image = np.clip(output_image, 0, 1)
    axis_img.set_data(np.swapaxes(output_image, 0, 1))
    axis.set_title(title, fontsize=6)
    axis.set_xticks([])
    axis.set_yticks([])


##########################################
## Construct animation from NN predictions
##########################################


def run_simulator_30_times(my_nn):

    starting_image = 0
    which_images = [0, 1, 2, 3, 4, 5, 6, 10, 20]
    for i in range(len(axis_actual_images)):
        send_numpy_data_to_image(
            axis_actual_images[i],
            image_actual_images[i],
            sample_sim[starting_image + which_images[i]],
        )

    predicted_output_images = [sample_sim[starting_image]]
    predicted_increments = []
    for i in range(21):
        nn_output = my_nn(
            convert_input_numpy_to_torch(predicted_output_images[i : i + 1])
        )
        nn_output_np = copy.deepcopy(
            convert_output_torch_to_numpy_increment(nn_output)[0]
        )

        predicted_increments.append(nn_output_np)
        predicted_output_images.append(
            accumulate_numpy_data(predicted_output_images[i], nn_output_np)
        )
    mses = [
        np.mean(np.square(predicted_output_images[i] - sample_sim[starting_image + i]))
        for i in range(21)
    ]

    for i in range(len(axis_predicted_images)):
        send_numpy_data_to_image(
            axis_predicted_images[i],
            image_predicted_images[i],
            predicted_output_images[which_images[i]],
            f"mse[{which_images[i]}]={mses[which_images[i]]:.7f}",
        )
        send_numpy_data_to_image(
            axis_predicted_increments[i],
            image_predicted_increments[i],
            predicted_increments[which_images[i]],
            title="increment",
            increment=True,
        )

    fig.canvas.draw()
    fig.canvas.flush_events()
    return (predicted_output_images, mses)


###############################
## Build all axis for the plots
###############################

fig = plt.figure(figsize=(8, 6), dpi=144)  # layout="constrained")
gs = GridSpec(10, 4, hspace=0.5, figure=fig)

# fig, ax = plt.subplots(10, 4)

# images with output and prediction
axis_actual_increment = fig.add_subplot(gs[0, 0])
image_actual_increment = axis_actual_increment.imshow(
    np.zeros((parameters["image_number_cols"], parameters["image_number_rows"])),
    interpolation="nearest",
    origin="upper",
)
axis_predicted_increment = fig.add_subplot(gs[0, 1])
image_predicted_increment = axis_predicted_increment.imshow(
    np.zeros((parameters["image_number_cols"], parameters["image_number_rows"])),
    interpolation="nearest",
    origin="upper",
)

# plots with losses
axis_losses = fig.add_subplot(gs[0:4, 3])
line_losses = axis_losses.plot(np.arange(0, 1000), np.arange(0, 1000), color="g")[0]
axis_losses.set_yscale("log")
axis_losses.set_ylim(1e-7, 1e-4)
axis_losses.grid(True)
axis_losses.set_title("loss", fontsize=8)

# plots with mean-square errors
axis_mses = fig.add_subplot(gs[5:9, 3])
lines_mses = axis_mses.plot(np.arange(0, 1000), np.ones((1000, 21)), color="g")
axis_mses.set_yscale("log")
axis_mses.set_ylim(1e-7, 1e-4)
axis_mses.grid(True)
axis_mses.set_title("mean-square-errs 1-20 steps ahead", fontsize=7)

axis_actual_images = [fig.add_subplot(gs[i, 0]) for i in range(1, 10)]
image_actual_images = [
    ax.imshow(
        np.zeros((parameters["image_number_cols"], parameters["image_number_rows"])),
        interpolation="nearest",
        origin="upper",
    )
    for ax in axis_actual_images
]
axis_predicted_images = [fig.add_subplot(gs[i, 1]) for i in range(1, 10)]
image_predicted_images = [
    ax.imshow(
        np.zeros((parameters["image_number_cols"], parameters["image_number_rows"])),
        interpolation="nearest",
        origin="upper",
    )
    for ax in axis_predicted_images
]
axis_predicted_increments = [fig.add_subplot(gs[i, 2]) for i in range(1, 10)]
image_predicted_increments = [
    ax.imshow(
        np.zeros((parameters["image_number_cols"], parameters["image_number_rows"])),
        interpolation="nearest",
        origin="upper",
    )
    for ax in axis_predicted_increments
]


with open(
    parameters["sample_simulation_file"],
    "rb",
) as filehandler:
    sample_sim = pickle.load(filehandler)

count = 0
# path = "C:/Users/ruihe/GitHub/Physics-based-Machine-learning-Fluid-sim/Training_data_pickle"

gamma = parameters["starting_gamma"]
threshold = parameters["starting_threshold"]

torch.manual_seed(parameters["torch_seed"])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device= torch.device("cpu")
print("Using device:", device)


class Net_NoConv(nn.Module):
    def __init__(self):
        super(Net_NoConv, self).__init__()
        # First fully connected layer
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

    # x represents our data
    def forward(self, x):
        # Flatten x with start_dim=1
        x = torch.flatten(x, 1)
        # Pass data through ``fc1``
        x = self.fc1(x)
        x = self.activation1(x)  ### USE ACTIVATION WITHOUT VANISHING GRADIENT
        x = self.fc2(x)
        x = self.activation2(x)  ### USE ACTIVATION WITHOUT VANISHING GRADIENT
        x = self.fc3(x)
        x = self.activation3(x)  ### USE ACTIVATION WITHOUT VANISHING GRADIENT
        x = self.fc4(x)
        x = self.activation4(x)  ### USE ACTIVATION WITHOUT VANISHING GRADIENT
        x = self.fc5(x)  ### NO ACTIVATION IN LAST LAYER
        # x = F.softplus(x)

        # Apply softmax to x
        output = x
        output = torch.reshape(
            output,
            (
                len(output),
                parameters["image_number_rows"],
                parameters["image_number_cols"],
                2,
            ),
        )
        # output = F.log_softmax(x, dim=1)
        return output


(total_input_images, total_output_images) = read_from_file_to_numpy_array(
    parameters["input_training_file"],
    parameters["output_training_file"],
)

number_training = (int)(
    parameters["percentage_training_images"] * len(total_input_images)
)

input_images_training = total_input_images[:number_training]
input_images_test = total_input_images[number_training:]

output_images_training = total_output_images[:number_training]
output_images_test = total_output_images[number_training:]

print(
    f"input_images_training={input_images_training.shape}, output_images_training={output_images_training.shape}"
)
print(
    f"input_images_test={input_images_test.shape} : output_images_test={output_images_test.shape}"
)

# Pytorching data
input_tensors_training = convert_input_numpy_to_torch(input_images_training)
output_tensors_training = convert_input_numpy_to_torch(output_images_training)
input_tensors_test = convert_input_numpy_to_torch(input_images_test)
output_tensors_test = convert_input_numpy_to_torch(output_images_test)

my_nn = Net_NoConv()
with open(
    f"C:/Users/ruihe/GitHub/new-physics/Training_data_pickle_compressed/Sample_sim/Best_weights7547.pkl",
    "rb",
) as filehandler:
    my_nn.load_state_dict(pickle.load(filehandler))


my_nn.to(device)  ### SEND MODEL TO GPU
print(my_nn)

# random_data = torch.rand((1, 3, parameters["image_number_rows"], parameters["image_number_cols"]))
# result = my_nn(random_data)

# output_Img = np.concatenate(
#     (np.zeros((parameters["image_number_rows"], parameters["image_number_cols"], 1)), result.detach().numpy().reshape(parameters["image_number_rows"], parameters["image_number_cols"], 2)), axis=2
# )
# random_data_IMG = random_data.numpy().reshape(parameters["image_number_rows"], parameters["image_number_cols"], 3)

n_epochs = parameters["n_epochs"]  # number of epochs to run
batch_size = parameters["batch_size"]  # size of each batch
batch_start = torch.arange(0, len(input_images_training), batch_size)


loss_fn = nn.MSELoss()
# loss_fn = nn.HuberLoss()

# def loss_fn(output, target):
#    # output, target come as tensors for an entire batch figure out how to make it work for multiple
#   loss = torch.mean(
#        torch.sum(torch.square(output - target))
#    )  ### DOES NOT MAKE SENSE TO SUM & THEN MEAN
#    return loss


optimizer = optim.Adam(my_nn.parameters(), lr=gamma)

# Hold the best model
best_mape = np.inf  # init to infinity
best_nn = None
fig.canvas.draw()
plt.show(block=False)
losses = []
mses = []
for epoch in range(n_epochs):
    arranged_arr = np.arange(len(input_tensors_training))
    np.random.shuffle(arranged_arr)
    randomized_train = [-1] * len(input_tensors_training)
    randomized_labels = [-1] * len(input_tensors_training)
    for i in range(len(input_tensors_training)):
        randomized_train[i] = input_tensors_training.detach()[arranged_arr[i]]
        randomized_labels[i] = output_tensors_training.detach()[arranged_arr[i]]
    randomized_train = np.array(randomized_train)
    randomized_labels = np.array(randomized_labels)

    my_nn.train()
    for start in batch_start:
        # take a batch
        input_batch = convert_input_numpy_to_torch(
            randomized_train[start : start + batch_size, :, :, :]
        )
        output_batch = convert_input_numpy_to_torch(
            randomized_labels[start : start + batch_size, :, :, :]
        )
        # forward pass in GPU
        output_batch_prediction = my_nn(input_batch.to(device))  ### SEND DATA TO GPU

        loss = loss_fn(
            output_batch_prediction, output_batch.to(device)
        )  ### SEND DATA TO GPU
        output_batch_prediction = (
            output_batch_prediction.cpu()
        )  ### BRING PREDICTION TOP CPU FOR PLOT
        losses.append(loss.cpu().detach())  ### BRING LOSS TO CPU FOR PLOT

        if ((int)(start / batch_size)) % 100 == 1:  ### ONLY SHOW 1 OUT OF 100

            print(f"epoch={epoch} : loss={loss:.2e} : gamma={gamma:.2e}")
            output_Img = np.concatenate(
                (
                    np.zeros(
                        (
                            parameters["image_number_rows"],
                            parameters["image_number_cols"],
                            1,
                        )
                    ),
                    output_batch_prediction[0]
                    .detach()
                    .numpy()
                    .reshape(
                        parameters["image_number_rows"],
                        parameters["image_number_cols"],
                        2,
                    ),
                ),
                axis=2,
            )

            output_test_prediction = my_nn(input_tensors_test[0:1].to(device)).cpu()

            send_numpy_data_to_image(
                axis_predicted_increment,
                image_predicted_increment,
                convert_output_torch_to_numpy_increment(output_test_prediction[0:1])[0],
                "predicted increment",
                increment=True,
            )

            send_numpy_data_to_image(
                axis_actual_increment,
                image_actual_increment,
                convert_output_torch_to_numpy_increment(output_tensors_test[0:1])[0],
                "actual increment",
                increment=True,
            )

            if True:
                # plot losses
                line_losses.set_xdata(np.arange(0, len(losses)))
                line_losses.set_ydata(np.array(losses))
                axis_losses.set_ylim(np.min(losses), 2e-6)
                axis_losses.set_xlim(0, len(losses))
                # plot mses
                if mses:
                    x = np.arange(0, len(mses))
                    yy = np.stack(mses, 0)
                    for i, line in enumerate(lines_mses):
                        line.set_xdata(x)
                        line.set_ydata(yy[:, i])
                    # 1st mse is zero, so do not include in minimum
                    axis_mses.set_ylim(np.min(yy[:, 1:]), 1e-3)
                    axis_mses.set_xlim(0, len(mses) - 1)
            else:
                lag = 10000
                line_losses.set_xdata(np.arange(0, lag))
                line_losses.set_ydata(np.array(losses[len(losses) - lag :]))
                axis_losses.set_ylim(
                    np.min(losses[len(losses) - lag :]),
                    np.max(losses[len(losses) - lag :]),
                )

            # axis_losses.set_xlim(0, lag)

            fig.canvas.draw()
            fig.canvas.flush_events()

        count += 1
        # backward pass
        optimizer.zero_grad()
        loss.backward()
        # update weights
        optimizer.step()

        # if loss.cpu().detach() < threshold:
        # gamma = gamma / 2
        # threshold = threshold / (1 + (5 / (count_gamma**0.5)))
        # count_gamma += 1
        # for g in optimizer.param_groups:
        # g["lr"] = gamma
    # evaluate accuracy at end of each epoch
    my_nn.eval()
    output_batch_prediction = my_nn(input_tensors_test.to(device))  ### SEND DATA TO GPU
    mape = loss_fn(
        output_batch_prediction, output_tensors_test.to(device)
    )  ### SEND DATA TO GPU
    mape = mape.cpu()  ### BRING BACK TO CPU
    if mape < best_mape:
        best_mape = mape
        print(f"    MAPE: {mape:.2e}")
        best_nn = copy.deepcopy(my_nn).cpu()
        best_weights = copy.deepcopy(my_nn.state_dict())
    (predicted_images, mse) = run_simulator_30_times(copy.deepcopy(my_nn).cpu())
    mses.append(mse)

    # rgb = np

print(f"final MAPE: {mape:.2e}")

# save as png and pdf
plt.savefig(parameters["prefix_for_saves"] + "_figure.png")
plt.savefig(parameters["prefix_for_saves"] + "_figure.pdf")

# restore model and return best accuracy
my_nn.load_state_dict(best_weights)

# add final results to json file
loss = float(loss.cpu().detach())
mape = float(mape.cpu().detach())
best_mape = float(best_mape.cpu().detach())
parameters["results"] = {
    "final_loss": loss,
    "final_mape": mape,
    "best_mape": best_mape,
    "final_gamma": gamma,
    "final_mses": mse,
    "training_time_sec": time.time() - start_time,
}

with open(
    parameters["prefix_for_saves"] + "_parameters.json", "w", encoding="utf-8"
) as file:
    json.dump(parameters, file, indent=4)

with open(
    parameters["prefix_for_saves"] + "_best_weights.pkl",
    "wb",
) as file:
    pickle.dump((best_weights), file)
