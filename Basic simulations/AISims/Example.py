# %pip install --quiet phiflow
from phi.torch.flow import *

import matplotlib.pyplot as plt

# from phi.flow import *  # If JAX is not installed. You can use phi.torch or phi.tf as well.
from tqdm.notebook import trange

gabagabagoo = 1


def potential(pos):
    return pos[0] + pos[1]


landscape = CenteredGrid(potential, x=200, y=200, bounds=Box(x=(-50, 50), y=(-50, 50)))
plot(landscape)

math.seed(0)
net = dense_net(2, 1, [32, 64, 128, 32])
optimizer = adam(net)


def loss_function(x, label):
    prediction = math.native_call(net, x)
    return math.l2_loss(prediction - label), prediction


input_data = rename_dims(landscape.points, spatial, batch)
labels = rename_dims(landscape.values, spatial, batch)
loss_function(input_data, labels)[0]

loss_trj = []
pred_trj = []
for i in range(5000):
    gabagabagoo += 1
    landscape = CenteredGrid(
        potential, x=200, y=200, bounds=Box(x=(-50, 50), y=(-50, 50))
    )
    if i % 100 == 99:
        print(i)
    loss, pred = update_weights(net, optimizer, loss_function, input_data, labels)
    loss_trj.append(loss)
    pred_trj.append(pred)
loss_trj = stack(loss_trj, spatial("iteration"))
pred_trj = stack(pred_trj, batch("iteration"))
plot(math.mean(loss_trj, "x,y"), err=math.std(loss_trj, "x,y"), size=(4, 3))

pred_grid = rename_dims(pred_trj.iteration[::40], "x,y", spatial)
plot(pred_grid, animate="iteration", size=(6, 5))

"""
def update(frame):
    ax.clear()

    voxels = changeArrToBools3D(3, stepUpdate(frame))
    ##voxels = np.rot90(voxels, k=1, axes=(1, 2))

    ax.voxels(voxels, facecolors="blue", alpha=0.6)

fig = plt.figure()
ax = fig.add_subplot(111, projection="2d")

ani = FuncAnimation(fig, update, interval=0)
"""
plt.show()
