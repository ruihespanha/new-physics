import ImageLoader as im
import numpy as np
import sklearn
import matplotlib.pyplot as plt
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
import copy
import pickle

sample_sim = 0
count_gamma = 1

##################################
## Functions to process image data
##################################


def read_from_file_to_numpy_array(filename1, filename2):
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
    with open(
        f"C:/Users/ruihe/GitHub/new-physics/Training_data_pickle_compressed/total_images.pkl",
        "rb",
    ) as filehandler:
        total_input_images = np.stack(pickle.load(filehandler), 0)
    with open(
        f"C:/Users/ruihe/GitHub/new-physics/Training_data_pickle_compressed/total_image_labels.pkl",
        "rb",
    ) as filehandler:
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
    output_numpy = np.concatenate([np.zeros((1, 24, 12, 1)), output2], 3)
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
    output_image = copy.deepcopy(input_numpy)
    # print(output_image[0:5, 0:5, 0:3])
    if increment:
        output_image[:, :, 1:3] = output_image[:, :, 1:3] * 10 + 0.5
    output_image = np.clip(output_image, 0, 1)
    axis_img.set_data(np.swapaxes(output_image, 0, 1))
    axis.set_title(title)


def run_simulator_30_times(my_nn):

    starting_image = 0
    send_numpy_data_to_image(ax[1, 0], k1_0, sample_sim[starting_image])
    send_numpy_data_to_image(ax[2, 0], k2_0, sample_sim[starting_image + 1])
    send_numpy_data_to_image(ax[3, 0], k3_0, sample_sim[starting_image + 2])
    send_numpy_data_to_image(ax[4, 0], k4_0, sample_sim[starting_image + 3])
    send_numpy_data_to_image(ax[5, 0], k5_0, sample_sim[starting_image + 4])
    send_numpy_data_to_image(ax[6, 0], k6_0, sample_sim[starting_image + 5])
    send_numpy_data_to_image(ax[7, 0], k7_0, sample_sim[starting_image + 6])
    send_numpy_data_to_image(ax[8, 0], k8_0, sample_sim[starting_image + 10])
    send_numpy_data_to_image(ax[9, 0], k9_0, sample_sim[starting_image + 20])

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

    send_numpy_data_to_image(
        ax[1, 1], k1_1, predicted_output_images[0], f"{mses[0]:.6f}"
    )
    send_numpy_data_to_image(
        ax[2, 1], k2_1, predicted_output_images[1], f"{mses[1]:.6f}"
    )
    send_numpy_data_to_image(
        ax[3, 1], k3_1, predicted_output_images[2], f"{mses[2]:.6f}"
    )
    send_numpy_data_to_image(
        ax[4, 1], k4_1, predicted_output_images[3], f"{mses[3]:.6f}"
    )
    send_numpy_data_to_image(
        ax[5, 1], k5_1, predicted_output_images[4], f"{mses[4]:.6f}"
    )
    send_numpy_data_to_image(
        ax[6, 1], k6_1, predicted_output_images[5], f"{mses[5]:.6f}"
    )
    send_numpy_data_to_image(
        ax[7, 1], k7_1, predicted_output_images[6], f"{mses[6]:.6f}"
    )
    send_numpy_data_to_image(
        ax[8, 1], k8_1, predicted_output_images[10], f"{mses[10]:.6f}"
    )
    send_numpy_data_to_image(
        ax[9, 1], k9_1, predicted_output_images[20], f"{mses[20]:.6f}"
    )

    send_numpy_data_to_image(ax[1, 2], k1_2, predicted_increments[0], increment=True)
    send_numpy_data_to_image(ax[2, 2], k2_2, predicted_increments[1], increment=True)
    send_numpy_data_to_image(ax[3, 2], k3_2, predicted_increments[2], increment=True)
    send_numpy_data_to_image(ax[4, 2], k4_2, predicted_increments[3], increment=True)
    send_numpy_data_to_image(ax[5, 2], k5_2, predicted_increments[4], increment=True)
    send_numpy_data_to_image(ax[6, 2], k6_2, predicted_increments[5], increment=True)
    send_numpy_data_to_image(ax[7, 2], k7_2, predicted_increments[6], increment=True)
    send_numpy_data_to_image(ax[8, 2], k8_2, predicted_increments[10], increment=True)
    send_numpy_data_to_image(ax[9, 2], k9_2, predicted_increments[20], increment=True)

    fig.canvas.draw()
    fig.canvas.flush_events()
    return predicted_output_images


with open(
    f"C:/Users/ruihe/GitHub/new-physics/Training_data_pickle_compressed/Sample_sim/RGB_images_3619.pkl",
    "rb",
) as filehandler:
    sample_sim = pickle.load(filehandler)

count = 0
path = "C:/Users/ruihe/GitHub/Physics-based-Machine-learning-Fluid-sim/Training_data_pickle"

gamma = 1e-05
threshold = 1e-5

torch.manual_seed(random.randint(1, 1000))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device= torch.device("cpu")
print("Using device:", device)

fig, ax = plt.subplots(10, 3)

k3 = ax[0, 0].imshow(np.zeros((12, 24)), interpolation="nearest", origin="upper")
k2 = ax[0, 1].imshow(np.zeros((12, 24)), interpolation="nearest", origin="upper")
k1 = ax[0, 2].plot(np.arange(0, 1000), np.arange(0, 1000), color="g")[0]

k1_0 = ax[1, 0].imshow(np.zeros((12, 24)), interpolation="nearest", origin="upper")
k1_1 = ax[1, 1].imshow(np.zeros((12, 24)), interpolation="nearest", origin="upper")
k1_2 = ax[1, 2].imshow(np.zeros((12, 24)), interpolation="nearest", origin="upper")

k2_0 = ax[2, 0].imshow(np.zeros((12, 24)), interpolation="nearest", origin="upper")
k2_1 = ax[2, 1].imshow(np.zeros((12, 24)), interpolation="nearest", origin="upper")
k2_2 = ax[2, 2].imshow(np.zeros((12, 24)), interpolation="nearest", origin="upper")

k3_0 = ax[3, 0].imshow(np.zeros((12, 24)), interpolation="nearest", origin="upper")
k3_1 = ax[3, 1].imshow(np.zeros((12, 24)), interpolation="nearest", origin="upper")
k3_2 = ax[3, 2].imshow(np.zeros((12, 24)), interpolation="nearest", origin="upper")

k4_0 = ax[4, 0].imshow(np.zeros((12, 24)), interpolation="nearest", origin="upper")
k4_1 = ax[4, 1].imshow(np.zeros((12, 24)), interpolation="nearest", origin="upper")
k4_2 = ax[4, 2].imshow(np.zeros((12, 24)), interpolation="nearest", origin="upper")

k5_0 = ax[5, 0].imshow(np.zeros((12, 24)), interpolation="nearest", origin="upper")
k5_1 = ax[5, 1].imshow(np.zeros((12, 24)), interpolation="nearest", origin="upper")
k5_2 = ax[5, 2].imshow(np.zeros((12, 24)), interpolation="nearest", origin="upper")

k6_0 = ax[6, 0].imshow(np.zeros((12, 24)), interpolation="nearest", origin="upper")
k6_1 = ax[6, 1].imshow(np.zeros((12, 24)), interpolation="nearest", origin="upper")
k6_2 = ax[6, 2].imshow(np.zeros((12, 24)), interpolation="nearest", origin="upper")

k7_0 = ax[7, 0].imshow(np.zeros((12, 24)), interpolation="nearest", origin="upper")
k7_1 = ax[7, 1].imshow(np.zeros((12, 24)), interpolation="nearest", origin="upper")
k7_2 = ax[7, 2].imshow(np.zeros((12, 24)), interpolation="nearest", origin="upper")

k8_0 = ax[8, 0].imshow(np.zeros((12, 24)), interpolation="nearest", origin="upper")
k8_1 = ax[8, 1].imshow(np.zeros((12, 24)), interpolation="nearest", origin="upper")
k8_2 = ax[8, 2].imshow(np.zeros((12, 24)), interpolation="nearest", origin="upper")

k9_0 = ax[9, 0].imshow(np.zeros((12, 24)), interpolation="nearest", origin="upper")
k9_2 = ax[9, 2].imshow(np.zeros((12, 24)), interpolation="nearest", origin="upper")
k9_1 = ax[9, 1].imshow(np.zeros((12, 24)), interpolation="nearest", origin="upper")

ax[0, 2].set_yscale("log")
ax[0, 2].set_ylim(1e-7, 1e-4)
ax[0, 2].grid(True)


class Net_NoConv(nn.Module):
    def __init__(self):
        super(Net_NoConv, self).__init__()
        # First fully connected layer
        self.fc1 = nn.Linear(in_features=(int)(864), out_features=(int)(864 * 8))
        # Second fully connected layer
        self.fc2 = nn.Linear(in_features=(int)(864 * 8), out_features=(int)(864 * 4))
        # Fourth fully connected layer
        self.fc3 = nn.Linear((int)(864 * 4), 864)
        self.fc4 = nn.Linear((int)(864), 864)
        self.fc5 = nn.Linear((int)(864), 576)
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
        output = torch.reshape(output, (len(output), 24, 12, 2))
        # output = F.log_softmax(x, dim=1)
        return output


(total_input_images, total_output_images) = read_from_file_to_numpy_array(
    f"C:/Users/ruihe/GitHub/new-physics/Training_data_pickle_compressed/total_images.pkl",
    f"C:/Users/ruihe/GitHub/new-physics/Training_data_pickle_compressed/total_image_labels.pkl",
)

number_training = (int)(9 * len(total_input_images) / 10)

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

# random_data = torch.rand((1, 3, 24, 12))
# result = my_nn(random_data)

# output_Img = np.concatenate(
#     (np.zeros((24, 12, 1)), result.detach().numpy().reshape(24, 12, 2)), axis=2
# )
# random_data_IMG = random_data.numpy().reshape(24, 12, 3)

n_epochs = 400  # number of epochs to run
batch_size = 600  # size of each batch
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
        X_batch = convert_input_numpy_to_torch(
            randomized_train[start : start + batch_size, :, :, :]
        )
        y_batch = convert_input_numpy_to_torch(
            randomized_labels[start : start + batch_size, :, :, :]
        )
        # forward pass in GPU
        y_pred = my_nn(X_batch.to(device))  ### SEND DATA TO GPU

        loss = loss_fn(y_pred, y_batch.to(device))  ### SEND DATA TO GPU
        y_pred = y_pred.cpu()  ### BRING PREDICTION TOP CPU FOR PLOT
        losses.append(loss.cpu().detach())  ### BRING LOSS TO CPU FOR PLOT

        if ((int)(start / batch_size)) % 100 == 1:  ### ONLY SHOW 1 OUT OF 100

            print(f"epoch={epoch} : loss={loss:.2e} : gamma={gamma:.2e}")
            output_Img = np.concatenate(
                (np.zeros((24, 12, 1)), y_pred[0].detach().numpy().reshape(24, 12, 2)),
                axis=2,
            )

            y_pred_test = my_nn(input_tensors_test[0:1].to(device)).cpu()

            send_numpy_data_to_image(
                ax[0, 1],
                k2,
                convert_output_torch_to_numpy_increment(y_pred_test[0:1])[0],
                "predicted increment",
                increment=True,
            )

            send_numpy_data_to_image(
                ax[0, 0],
                k3,
                convert_output_torch_to_numpy_increment(output_tensors_test[0:1])[0],
                "actual increment",
                increment=True,
            )

            lag = 10000
            if True:
                k1.set_xdata(np.arange(0, len(losses)))
                k1.set_ydata(np.array(losses))
                ax[0, 2].set_ylim(np.min(losses), 1e-5)
                ax[0, 2].set_xlim(0, len(losses))
            else:
                k1.set_xdata(np.arange(0, lag))
                k1.set_ydata(np.array(losses[len(losses) - lag :]))
                ax[0, 2].set_ylim(
                    np.min(losses[len(losses) - lag :]),
                    np.max(losses[len(losses) - lag :]),
                )

            # ax[0, 2].set_xlim(0, lag)

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
    y_pred = my_nn(input_tensors_test.to(device))  ### SEND DATA TO GPU
    mape = loss_fn(y_pred, output_tensors_test.to(device))  ### SEND DATA TO GPU
    mape = mape.cpu()  ### BRING BACK TO CPU
    if mape < best_mape:
        best_mape = mape
        print(f"    MAPE: {mape:.2e}")
        best_nn = copy.deepcopy(my_nn).cpu()
        best_weights = copy.deepcopy(my_nn.state_dict())
    run_simulator_30_times(copy.deepcopy(my_nn).cpu())

    # rgb = np

plt.savefig("fugure after training with staring image of 10")


# restore model and return best accuracy
my_nn.load_state_dict(best_weights)

with open(
    f"C:/Users/ruihe/GitHub/new-physics/Training_data_pickle_compressed/Sample_sim/Best_weights{random.randint(0,10000)}.pkl",
    "wb",
) as file:
    pickle.dump((best_weights), file)

print(f"final MAPE: {mape:.2e}")
