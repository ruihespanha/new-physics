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

count = 0
path = "C:/Users/ruihe/GitHub/Physics-based-Machine-learning-Fluid-sim/Training_data_pickle"

fig, ax = plt.subplots(2)

k3 = ax[0].imshow(np.zeros((12, 24)), interpolation="nearest", origin="upper")
k2 = ax[1].imshow(np.zeros((12, 24)), interpolation="nearest", origin="upper")

total_images, total_image_labels = im.ImageLoaderPKL(path)  # 50 trials

total_images = np.swapaxes(total_images, 1, 3)
total_image_labels = np.swapaxes(total_image_labels, 1, 3)

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
        nn.init.xavier_uniform_(self.conv1.weight)

        # First fully connected layer
        self.fc1 = nn.Linear(in_features=(int)(3456), out_features=(int)(3456))
        # Second fully connected layer that outputs our 10 labels
        self.fc2 = nn.Linear(in_features=(int)(3456), out_features=(int)(864))
        self.fc3 = nn.Linear((int)(864), 576)

    # x represents our data
    def forward(self, x):
        # Pass data through conv1
        x = self.conv1(x)
        # Use the rectified-linear activation function over x
        x = F.softplus(x)

        x = self.conv2(x)
        x = F.softplus(x)

        # Flatten x with start_dim=1
        x = torch.flatten(x, 1)
        # Pass data through ``fc1``
        x = self.fc1(x)
        x = F.softplus(x)
        x = self.fc2(x)
        x = F.softplus(x)
        x = self.fc3(x)
        x = F.softplus(x)

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
        self.fc2 = nn.Linear(in_features=(int)(864 * 2), out_features=(int)(864))
        # Second fully connected layer that outputs our 10 labels
        self.fc3 = nn.Linear(in_features=(int)(864), out_features=(int)(576))
        self.fc4 = nn.Linear((int)(576), 576)

    # x represents our data
    def forward(self, x):
        # Flatten x with start_dim=1
        # Pass data through ``fc1``
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.softplus(x)
        x = self.fc2(x)
        x = F.softplus(x)
        x = self.fc3(x)
        x = F.softplus(x)
        x = self.fc4(x)
        x = F.softplus(x)

        # Apply softmax to x
        output = x

        output = torch.reshape(output, (len(output), 2, 12, 24))
        # output = F.log_softmax(x, dim=1)
        return output


my_nn = Net_NoConv()
print(my_nn)

# random_data = torch.rand((1, 3, 24, 12))
# result = my_nn(random_data)

# output_Img = np.concatenate(
#     (np.zeros((24, 12, 1)), result.detach().numpy().reshape(24, 12, 2)), axis=2
# )
# random_data_IMG = random_data.numpy().reshape(24, 12, 3)

# print(np.sum(np.absolute(random_data_IMG - output_Img)))

n_epochs = 1000  # number of epochs to run
batch_size = 10  # size of each batch
batch_start = torch.arange(0, len(images), batch_size)


def loss_fn(output, target):

    # output, target come as tensors for an entire batch figure out how to make it work for multiple
    loss = torch.mean(torch.sum(torch.absolute(output - target)))
    if count % 100 == 0:
        print(float(loss))
    return loss


optimizer = optim.Adam(my_nn.parameters(), lr=0.00001)

# Hold the best model
best_mape = np.inf  # init to infinity
best_weights = None
fig.canvas.draw()
plt.show(block=False)
for epoch in range(n_epochs):
    print(epoch)
    my_nn.train()
    for start in batch_start:
        # take a batch
        X_batch = X_train[start : start + batch_size, :, :, :]
        y_batch = y_train[start : start + batch_size, :, :, :]
        # forward pass
        y_pred = my_nn(X_batch)
        if ((int)(start / 10)) % 10 == 1:
            output_Img = np.concatenate(
                (np.zeros((24, 12, 1)), y_pred[0].detach().numpy().reshape(24, 12, 2)),
                axis=2,
            )

            rgb = np.zeros((12, 24, 3))
            rgb[:, :, 1] = y_pred[0].detach().numpy()[0, :, :]
            rgb[:, :, 2] = y_pred[0].detach().numpy()[1, :, :]

            k2.set_data(rgb)
            rgb = np.zeros((12, 24, 3))
            rgb[:, :, 1] = y_batch[0].detach().numpy()[0, :, :]
            rgb[:, :, 2] = y_batch[0].detach().numpy()[1, :, :]

            k3.set_data(rgb)
            fig.canvas.draw()
            fig.canvas.flush_events()

        # print(y_pred.shape)
        loss = loss_fn(y_pred, y_batch)
        count += 1
        # backward pass
        optimizer.zero_grad()
        loss.backward()
        # update weights
        optimizer.step()
    # evaluate accuracy at end of each epoch
    my_nn.eval()
    y_pred = my_nn(X_test)
    mape = float(loss_fn(y_pred, y_test))
    if mape < best_mape:
        best_mape = mape
        best_weights = copy.deepcopy(my_nn.state_dict())

# restore model and return best accuracy
my_nn.load_state_dict(best_weights)
print("MAPE: %.2f" % best_mape)

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
