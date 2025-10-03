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


def convert_result_to_torch_format(result):
    # result = result.reshape((1, 24, 12, 3))
    temp = np.zeros((1, 24, 12, 3))
    temp[0] = result
    result = torch.tensor(temp, dtype=torch.float32)
    return result


def convert_nn_result_to_numpy(result):

    temp = np.concatenate(
        (np.zeros((24, 12, 1)), result.detach().numpy().reshape(24, 12, 2)),
        axis=2,
    )
    result = temp
    return result


def run_simulator_30_times(my_nn):
    # print(f"X_test shape: {X_test[0].shape}")

    starting_image = 0
    print(np.swapaxes(sample_sim[starting_image], 0, 1))
    k1_0.set_data(np.swapaxes(sample_sim[starting_image], 0, 1))
    k2_0.set_data(np.swapaxes(sample_sim[starting_image + 1], 0, 1))
    k3_0.set_data(np.swapaxes(sample_sim[starting_image + 2], 0, 1))
    k4_0.set_data(np.swapaxes(sample_sim[starting_image + 3], 0, 1))
    k5_0.set_data(np.swapaxes(sample_sim[starting_image + 4], 0, 1))
    k6_0.set_data(np.swapaxes(sample_sim[starting_image + 5], 0, 1))
    k7_0.set_data(np.swapaxes(sample_sim[starting_image + 6], 0, 1))
    k8_0.set_data(np.swapaxes(sample_sim[starting_image + 10], 0, 1))
    k9_0.set_data(np.swapaxes(sample_sim[starting_image + 20], 0, 1))
    temp_y_pred_addative = [np.swapaxes(X_test[0].detach().numpy(), 0, 2)]
    temp_y_pred = []
    for i in range(21):
        nn_temp_result = my_nn(convert_result_to_torch_format(temp_y_pred_addative[i]))
        # print(f"X_test shape: {X_test[0].shape}")
        nn_result = convert_nn_result_to_numpy(nn_temp_result)
        temp_y_pred_addative.append(temp_y_pred_addative[i] + nn_result)
        temp_y_pred.append(copy.deepcopy(nn_result))
    mses = [
        np.mean(np.square(temp_y_pred_addative[i] - sample_sim[starting_image + i]))
        for i in range(21)
    ]

    k0.set_data(np.swapaxes(np.swapaxes(X_test[0].detach().numpy(), 0, 2), 0, 1))

    k1_1.set_data(np.clip(np.swapaxes(temp_y_pred_addative[0], 0, 1), 0, 1))
    ax[1, 1].set_title(f"{mses[0]:.6f}")
    k2_1.set_data(np.clip(np.swapaxes(temp_y_pred_addative[1], 0, 1), 0, 1))
    ax[2, 1].set_title(f"{mses[1]:.6f}")
    k3_1.set_data(np.clip(np.swapaxes(temp_y_pred_addative[2], 0, 1), 0, 1))
    ax[3, 1].set_title(f"{mses[2]:.6f}")
    k4_1.set_data(np.clip(np.swapaxes(temp_y_pred_addative[3], 0, 1), 0, 1))
    ax[4, 1].set_title(f"{mses[3]:.6f}")
    k5_1.set_data(np.clip(np.swapaxes(temp_y_pred_addative[4], 0, 1), 0, 1))
    ax[5, 1].set_title(f"{mses[4]:.6f}")
    k6_1.set_data(np.clip(np.swapaxes(temp_y_pred_addative[5], 0, 1), 0, 1))
    ax[6, 1].set_title(f"{mses[5]:.6f}")
    k7_1.set_data(np.clip(np.swapaxes(temp_y_pred_addative[6], 0, 1), 0, 1))
    ax[7, 1].set_title(f"{mses[6]:.6f}")
    k8_1.set_data(np.clip(np.swapaxes(temp_y_pred_addative[10], 0, 1), 0, 1))
    ax[8, 1].set_title(f"{mses[10]:.6f}")
    k9_1.set_data(np.clip(np.swapaxes(temp_y_pred_addative[20], 0, 1), 0, 1))
    ax[9, 1].set_title(f"{mses[20]:.6f}")

    k1_2.set_data(np.clip(np.swapaxes((temp_y_pred[0] * 10) + 0.5, 0, 1), 0, 1))
    k2_2.set_data(np.clip(np.swapaxes((temp_y_pred[1] * 10) + 0.5, 0, 1), 0, 1))
    k3_2.set_data(np.clip(np.swapaxes((temp_y_pred[2] * 10) + 0.5, 0, 1), 0, 1))
    k4_2.set_data(np.clip(np.swapaxes((temp_y_pred[3] * 10) + 0.5, 0, 1), 0, 1))
    k5_2.set_data(np.clip(np.swapaxes((temp_y_pred[4] * 10) + 0.5, 0, 1), 0, 1))
    k6_2.set_data(np.clip(np.swapaxes((temp_y_pred[5] * 10) + 0.5, 0, 1), 0, 1))
    k7_2.set_data(np.clip(np.swapaxes((temp_y_pred[6] * 10) + 0.5, 0, 1), 0, 1))
    k8_2.set_data(np.clip(np.swapaxes((temp_y_pred[10] * 10) + 0.5, 0, 1), 0, 1))
    k9_2.set_data(np.clip(np.swapaxes((temp_y_pred[20] * 10) + 0.5, 0, 1), 0, 1))

    fig.canvas.draw()
    fig.canvas.flush_events()
    return temp_y_pred_addative


with open(
    f"C:/Users/ruihe/GitHub/new-physics/Training_data_pickle_compressed/Sample_sim/RGB_images_3619.pkl",
    "rb",
) as filehandler:
    sample_sim = pickle.load(filehandler)

count = 0
path = "C:/Users/ruihe/GitHub/Physics-based-Machine-learning-Fluid-sim/Training_data_pickle"

gamma = 0.00003
threshold = 1e-5

torch.manual_seed(random.randint(1, 1000))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device= torch.device("cpu")
print("Using device:", device)

fig, ax = plt.subplots(11, 3)

k3 = ax[0, 0].imshow(np.zeros((12, 24)), interpolation="nearest", origin="upper")
k2 = ax[0, 1].imshow(np.zeros((12, 24)), interpolation="nearest", origin="upper")
k1 = ax[0, 2].plot(np.arange(0, 1000), np.arange(0, 1000), color="g")[0]
k0 = ax[10, 1].imshow(np.zeros((12, 24)), interpolation="nearest", origin="upper")

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

# total_images, total_image_labels = im.ImageLoaderPKL(path)  # 1000 trials

# with open(
#     f"C:/Users/ruihe/GitHub/new-physics/Training_data_pickle_compressed/total_images.pkl",
#     "wb",
# ) as file:
#     pickle.dump(total_images, file)

# with open(
#     f"C:/Users/ruihe/GitHub/new-physics/Training_data_pickle_compressed/total_image_labels.pkl",
#     "wb",
# ) as file:
#     pickle.dump(total_image_labels, file)

with open(
    f"C:/Users/ruihe/GitHub/new-physics/Training_data_pickle_compressed/total_images.pkl",
    "rb",
) as filehandler:
    total_images = pickle.load(filehandler)
with open(
    f"C:/Users/ruihe/GitHub/new-physics/Training_data_pickle_compressed/total_image_labels.pkl",
    "rb",
) as filehandler:
    total_image_labels = pickle.load(filehandler)

total_images = np.swapaxes(total_images, 1, 3)
total_image_labels = np.swapaxes(total_image_labels, 1, 3)
total_image_labels = total_image_labels - total_images

images = np.stack(total_images[: (int)(9 * len(total_images) / 10)], 0)
image_test = np.stack(total_images[(int)(9 * len(total_images) / 10) :], 0)

image_labels = np.stack(total_image_labels[: (int)(9 * len(total_images) / 10)], 0)
image_labels = image_labels[:, 1:3, :, :]
image_labels_test = np.stack(total_image_labels[(int)(9 * len(total_images) / 10) :], 0)
image_labels_test = image_labels_test[:, 1:3, :, :]

print(
    f"{images.shape} : {image_test.shape} : {image_labels.shape} : {image_labels_test.shape}"
)

# Pytorching data
X_train = torch.tensor(images, dtype=torch.float32)
y_train = torch.tensor(image_labels, dtype=torch.float32)
X_test = torch.tensor(image_test, dtype=torch.float32)
y_test = torch.tensor(image_labels_test, dtype=torch.float32)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # First 2D convolutional layer, taking in 3 input channels (RGB),
        # outputting 32 convolutional features, with a square kernel size of 7
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=1
        )
        # Second 2D convolutional layer, taking in the 64 input layers,
        # outputting 64 convolutional features, with a square kernel size of 3
        self.conv2 = nn.Conv2d(
            in_channels=6,
            out_channels=12,
            kernel_size=5,
            stride=1,
            padding=2,
        )
        # First fully connected layer
        self.fc1 = nn.Linear(in_features=(int)(3456), out_features=(int)(3456))
        # Second fully connected layer that outputs our 10 labels
        self.fc2 = nn.Linear(in_features=(int)(3456), out_features=(int)(864))
        self.fc3 = nn.Linear((int)(864), 576)
        self.activation = nn.PReLU()  ### USE ACTIVATION WITHOUT VANISHING GRADIENT

    # x represents our data
    def forward(self, x):
        # Pass data through conv1
        x = self.conv1(x)
        # Use the rectified-linear activation function over x
        # x = F.softplus(x)
        x = self.activation(x)  ### USE ACTIVATION WITHOUT VANISHING GRADIENT

        x = self.conv2(x)
        # x = F.softplus(x)
        x = self.activation(x)  ### USE ACTIVATION WITHOUT VANISHING GRADIENT

        # Flatten x with start_dim=1
        x = torch.flatten(x, 1)
        # Pass data through ``fc1``
        x = self.fc1(x)
        # x = F.softplus(x)
        x = self.activation(x)  ### USE ACTIVATION WITHOUT VANISHING GRADIENT
        x = self.fc2(x)
        x = self.activation(x)  ### USE ACTIVATION WITHOUT VANISHING GRADIENT
        # x = F.softplus(x)
        x = self.fc3(x)  ### NO ACTIVATION IN LAST LAYER
        # x = F.softplus(x)

        # Apply softmax to x
        output = x
        output = torch.reshape(output, (len(output), 2, 12, 24))
        # output = F.log_softmax(x, dim=1)
        return output


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
        output = torch.reshape(output, (len(output), 2, 12, 24))
        # output = F.log_softmax(x, dim=1)
        return output


my_nn = Net_NoConv()
my_nn.to(device)  ### SEND MODEL TO GPU
print(my_nn)

# random_data = torch.rand((1, 3, 24, 12))
# result = my_nn(random_data)

# output_Img = np.concatenate(
#     (np.zeros((24, 12, 1)), result.detach().numpy().reshape(24, 12, 2)), axis=2
# )
# random_data_IMG = random_data.numpy().reshape(24, 12, 3)

# print(np.sum(np.absolute(random_data_IMG - output_Img)))

n_epochs = 300  # number of epochs to run
batch_size = 600  # size of each batch
batch_start = torch.arange(0, len(images), batch_size)


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
    arranged_arr = np.arange(len(X_train))
    np.random.shuffle(arranged_arr)
    randomized_train = [-1] * len(X_train)
    randomized_labels = [-1] * len(X_train)
    for i in range(len(X_train)):
        randomized_train[i] = X_train.detach()[arranged_arr[i]]
        randomized_labels[i] = y_train.detach()[arranged_arr[i]]
    randomized_train = np.array(randomized_train)
    randomized_labels = np.array(randomized_labels)

    my_nn.train()
    for start in batch_start:
        # take a batch
        X_batch = torch.tensor(randomized_train[start : start + batch_size, :, :, :])
        y_batch = torch.tensor(randomized_labels[start : start + batch_size, :, :, :])
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

            y_pred_test = my_nn(X_test.to(device)).cpu()
            rgb = np.zeros((12, 24, 3))
            rgb[:, :, 1] = np.clip(
                y_pred_test[0].detach().numpy()[0, :, :] * 10 + 0.5, 0, 1
            )
            rgb[:, :, 2] = np.clip(
                y_pred_test[0].detach().numpy()[1, :, :] * 10 + 0.5, 0, 1
            )
            # rgb = np.zeros((12, 24, 3))
            # rgb[:, :, 1] = np.clip(X_batch[0].detach().numpy()[1, :, :], 0, 1)
            # rgb[:, :, 2] = np.clip(X_batch[0].detach().numpy()[2, :, :], 0, 1)

            k2.set_data(rgb)
            rgb = np.zeros((12, 24, 3))
            rgb[:, :, 1] = np.clip(y_test[0].detach().numpy()[0, :, :] * 10 + 0.5, 0, 1)
            rgb[:, :, 2] = np.clip(y_test[0].detach().numpy()[1, :, :] * 10 + 0.5, 0, 1)

            k3.set_data(rgb)
            # k3.set_data(
            #     np.swapaxes(np.swapaxes(X_test[0].detach().numpy(), 0, 2), 0, 1)
            # )
            lag = 10000
            if True:
                k1.set_xdata(np.arange(0, len(losses)))
                k1.set_ydata(np.array(losses))
                ax[0, 2].set_ylim(np.min(losses), np.max(losses))
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

        if loss.cpu().detach() < threshold:
            gamma = gamma / 2
            threshold = threshold / (1 + (5 / (count_gamma**0.5)))
            count_gamma += 1
            for g in optimizer.param_groups:
                g["lr"] = gamma
    # evaluate accuracy at end of each epoch
    my_nn.eval()
    y_pred = my_nn(X_test.to(device))  ### SEND DATA TO GPU
    mape = loss_fn(y_pred, y_test.to(device))  ### SEND DATA TO GPU
    mape = mape.cpu()  ### BRING BACK TO CPU
    if mape < best_mape:
        best_mape = mape
        print(f"    MAPE: {mape:.2e}")
        best_nn = copy.deepcopy(my_nn).cpu()
    run_simulator_30_times(copy.deepcopy(my_nn).cpu())

    # rgb = np

plt.savefig("fugure after training with staring image of 10")


# restore model and return best accuracy
# my_nn.load_state_dict(best_nn) # no longer works

with open(
    f"C:/Users/ruihe/GitHub/new-physics/Training_data_pickle_compressed/Sample_sim/Best_model{random.randint(0,10000)}.pkl",
    "wb",
) as file:
    pickle.dump((best_nn), file)

print(f"final MAPE: {mape:.2e}")
