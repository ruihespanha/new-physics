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

count = 0
path = "C:/Users/ruihe/GitHub/Physics-based-Machine-learning-Fluid-sim/Training_data_pickle"

gamma = 0.00003
threshold = 1e-5

torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device= torch.device("cpu")
print("Using device:", device)

fig, ax = plt.subplots(3)

k3 = ax[0].imshow(np.zeros((12, 24)), interpolation="nearest", origin="upper")
k2 = ax[1].imshow(np.zeros((12, 24)), interpolation="nearest", origin="upper")
k1 = ax[2].plot(np.arange(0, 1000), np.arange(0, 1000), color="g")[0]
ax[2].set_yscale("log")
ax[2].set_ylim(1e-8, 1e-3)
ax[2].grid(True)


total_images, total_image_labels = im.ImageLoaderPKL(path)  # 50 trials

total_images = np.swapaxes(total_images, 1, 3)
total_image_labels = np.swapaxes(total_image_labels, 1, 3)
total_image_labels = total_image_labels - total_images

images = np.stack(total_images[: (int)(7 * len(total_images) / 10)], 0)
image_test = np.stack(total_images[(int)(7 * len(total_images) / 10) :], 0)

image_labels = np.stack(total_image_labels[: (int)(7 * len(total_images) / 10)], 0)
image_labels = image_labels[:, 1:3, :, :]
image_labels_test = np.stack(total_image_labels[(int)(7 * len(total_images) / 10) :], 0)
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
        self.fc1 = nn.Linear(in_features=(int)(864), out_features=(int)(864 * 2))
        # Second fully connected layer
        self.fc2 = nn.Linear(in_features=(int)(864 * 2), out_features=(int)(864))
        # Third fully connected layer that outputs our 10 labels
        self.fc3 = nn.Linear(in_features=(int)(864), out_features=(int)(576))
        # Fourth fully connected layer
        self.fc4 = nn.Linear((int)(576), 576)
        self.activation = nn.PReLU()  ### USE ACTIVATION WITHOUT VANISHING GRADIENT

    # x represents our data
    def forward(self, x):
        # Flatten x with start_dim=1
        x = torch.flatten(x, 1)
        # Pass data through ``fc1``
        x = self.fc1(x)
        x = self.activation(x)  ### USE ACTIVATION WITHOUT VANISHING GRADIENT
        x = self.fc2(x)
        x = self.activation(x)  ### USE ACTIVATION WITHOUT VANISHING GRADIENT
        x = self.fc3(x)
        x = self.activation(x)  ### USE ACTIVATION WITHOUT VANISHING GRADIENT
        x = self.fc4(x)  ### NO ACTIVATION IN LAST LAYER
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

n_epochs = 1000  # number of epochs to run
batch_size = 30  # size of each batch
batch_start = torch.arange(0, len(images), batch_size)


loss_fn = nn.MSELoss()

# def loss_fn(output, target):
#    # output, target come as tensors for an entire batch figure out how to make it work for multiple
#   loss = torch.mean(
#        torch.sum(torch.square(output - target))
#    )  ### DOES NOT MAKE SENSE TO SUM & THEN MEAN
#    return loss


optimizer = optim.Adam(my_nn.parameters(), lr=gamma)

# Hold the best model
best_mape = np.inf  # init to infinity
best_weights = None
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

        if ((int)(start / 30)) % 87 == 1:  ### ONLY SHOW 1 OUT OF 100
            print(f"{epoch} : {loss:.2e} : {gamma:.2e}")
            output_Img = np.concatenate(
                (np.zeros((24, 12, 1)), y_pred[0].detach().numpy().reshape(24, 12, 2)),
                axis=2,
            )

            rgb = np.zeros((12, 24, 3))
            rgb[:, :, 1] = np.clip(y_pred[0].detach().numpy()[0, :, :] * 10 + 0.5, 0, 1)
            rgb[:, :, 2] = np.clip(y_pred[0].detach().numpy()[1, :, :] * 10 + 0.5, 0, 1)
            # rgb = np.zeros((12, 24, 3))
            # rgb[:, :, 1] = np.clip(X_batch[0].detach().numpy()[1, :, :], 0, 1)
            # rgb[:, :, 2] = np.clip(X_batch[0].detach().numpy()[2, :, :], 0, 1)

            k2.set_data(rgb)
            rgb = np.zeros((12, 24, 3))
            rgb[:, :, 1] = np.clip(
                y_batch[0].detach().numpy()[0, :, :] * 10 + 0.5, 0, 1
            )
            rgb[:, :, 2] = np.clip(
                y_batch[0].detach().numpy()[1, :, :] * 10 + 0.5, 0, 1
            )

            k3.set_data(rgb)
            k1.set_xdata(np.arange(0, len(losses)))
            k1.set_ydata(np.array(losses))

            ax[2].set_xlim(0, len(losses))

            fig.canvas.draw()
            fig.canvas.flush_events()

        count += 1
        # backward pass
        optimizer.zero_grad()
        loss.backward()
        # update weights
        optimizer.step()

        if loss.cpu().detach() < threshold:
            gamma = gamma / 4
            threshold = threshold / 5
            for g in optimizer.param_groups:
                g["lr"] = gamma
    # evaluate accuracy at end of each epoch
    my_nn.eval()
    y_pred = my_nn(X_test.to(device))  ### SEND DATA TO GPU
    mape = loss_fn(y_pred, y_test.to(device))  ### SEND DATA TO GPU
    mape = mape.cpu()  ### BRING BACK TO CPU
    if mape < best_mape:
        best_mape = mape
        print(f"                        MAPE: {mape:.2e}")
        best_weights = copy.deepcopy(my_nn.state_dict())

# restore model and return best accuracy
my_nn.load_state_dict(best_weights)
print(f"final MAPE: {mape:.2e}")

# my_nn.eval()
# with torch.no_grad():
#     # Test out inference with 5 samples
#     for i in range(5):
#         X_sample = image_test[i : i + 1]
#         X_sample = torch.tensor(X_sample, dtype=torch.float32)
#         y_pred = my_nn(X_sample)
#         print(
#             f"{image_test[i]} -> {y_pred[0].numpy()} (expected {image_labels_test[i].numpy()})"
#         )

# plt.show()

"""
# Equates to one random 24 x 12 image
random_data = torch.rand((1, 3, 24, 12))

result = my_nn(random_data)


print(f"{images[0].shape} : {result.detach().numpy().reshape(24,12,2).shape}")

result.detach().numpy().reshape(24, 12, 2)

output_Img = np.concatenate(
    (np.zeros((24, 12, 1)), result.detach().numpy().reshape(24, 12, 2)), axis=2
)

print(output_Img.shape)
print(output_Img)
# print(random_data)
ax.imshow(
    np.asarray(output_Img * 40),
    # random_data.reshape(24, 12, 3),
    vmin=0,
    vmax=1,
    interpolation="none",
)
plt.show()

fig, ax = plt.subplots(1)
ax.imshow(0)
k3 = ax.imshow(np.zeros((3, 12, 24)), interpolation="nearest", origin="upper")
rgb = np.zeros((3, 12, 24))
print(random_data.numpy().shape)
rgb[:, :, 0] = random_data.numpy()[0, :, :]
rgb[:, :, 1] = random_data.numpy()[1, :, :]

k3.set_data(rgb)
print(result)
"""


# n_epochs = 100   # number of epochs to run
# batch_size = 10  # size of each batch
# batch_start = torch.arange(0, len(X_train), batch_size)

# # Hold the best model
# best_mape = np.inf   # init to infinity
# best_weights = None
