import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from tqdm import tqdm

from NN_definition import Net_NoConv, get_simulation, read_from_file_to_numpy_array

# prefix_to_plot = "C:/Users/ruihe/GitHub/new-physics/results/NN_results20251003_021025"
prefix_to_plot = "C:/Users/ruihe/GitHub/new-physics/results/NN_results20251004_173504"

with open(prefix_to_plot + "_parameters.json") as json_file:
    parameters = json.load(json_file)

my_nn = Net_NoConv(parameters)
my_nn.load_weights(prefix_to_plot + "_best_weights.pkl")

#########################
## Get data for animation
#########################

(total_input_images, total_output_images, simulation_ids, simulation_times) = (
    read_from_file_to_numpy_array(
        parameters["input_training_file"],
        parameters["output_training_file"],
    )
)

number_first_test = (int)(
    parameters["percentage_training_images"] * len(total_input_images)
)

# single simulation
training_id = simulation_ids[10]
testing_id = simulation_ids[number_first_test + 10]

(times_training, images_training) = get_simulation(
    total_input_images, simulation_ids, simulation_times, training_id
)

(times_testing, images_testing) = get_simulation(
    total_input_images, simulation_ids, simulation_times, testing_id
)

predicted_training = my_nn.predict_images(images_training[0])
predicted_testing = my_nn.predict_images(images_testing[0])


def plot_animation(ax, times, images, which_images):
    """Plot one animation

    Args:
        ax (_type_): _description_
        times (_type_): _description_
        images (_type_): _description_
        which_images (_type_): _description_
    """
    for i, which in enumerate(which_images):
        ax[i].imshow(images[which])
        ax[i].set_ylabel(f"t={times[which]}", fontsize=8)
        ax[i].set_xticks([])
        ax[i].set_yticks([])
        ax[i].invert_yaxis()


###################
## Plot simulations
###################

fig1 = plt.figure(figsize=(8, 4), dpi=144)  # layout="constrained")
gs = GridSpec(3, 7, hspace=0, wspace=0, figure=fig1)

axis_training = [
    fig1.add_subplot(gs[0, 0]),
    fig1.add_subplot(gs[1, 0]),
    fig1.add_subplot(gs[2, 0]),
    fig1.add_subplot(gs[:, 1:3]),
]
plot_animation(axis_training, times_training, images_training, [0, 5, 10, 20])

axis_test = [
    fig1.add_subplot(gs[0, 6]),
    fig1.add_subplot(gs[1, 6]),
    fig1.add_subplot(gs[2, 6]),
    fig1.add_subplot(gs[:, 4:6]),
]
plot_animation(axis_test, times_testing, images_testing, [0, 5, 10, 20])

for ext in ["png", "pdf", "svg"]:
    fig1.savefig(prefix_to_plot + "_plot_simulation_animation." + ext)

###################
## Plot simulations
###################

fig2 = plt.figure(figsize=(8, 4), dpi=144)  # layout="constrained")
gs = GridSpec(3, 7, hspace=0, wspace=0, figure=fig2)

axis_training = [
    fig2.add_subplot(gs[0, 0]),
    fig2.add_subplot(gs[1, 0]),
    fig2.add_subplot(gs[2, 0]),
    fig2.add_subplot(gs[:, 1:3]),
]
plot_animation(axis_training, times_training, predicted_training, [0, 5, 10, 20])

axis_test = [
    fig2.add_subplot(gs[0, 6]),
    fig2.add_subplot(gs[1, 6]),
    fig2.add_subplot(gs[2, 6]),
    fig2.add_subplot(gs[:, 4:6]),
]
plot_animation(axis_test, times_testing, predicted_testing, [0, 5, 10, 20])

for ext in ["png", "pdf", "svg"]:
    fig2.savefig(prefix_to_plot + "_plot_prediction_animation." + ext)


#####################
## Plot RMSEs vs Time
#####################


def compute_RMSEs(
    my_nn, total_input_images, simulation_ids, simulation_times, image_ids
):
    RMSEs = []
    max_value = 0
    for id in tqdm(image_ids):
        times, images = get_simulation(
            total_input_images, simulation_ids, simulation_times, id
        )
        max_image = max([np.max(np.abs(image - 0.5)) for image in images])
        if max_image > max_value:
            max_value = max_image
        predicted_images = my_nn.predict_images(images[0])
        rmse = [
            np.sqrt(
                np.mean(
                    np.square(images[t][:, :, 1:3] - predicted_images[t][:, :, 1:3])
                )
            )
            for t in range(21)
        ]
        RMSEs.append(rmse)
    RMSEs = np.stack(RMSEs, 0)
    RMSEs = RMSEs / max_value * 100
    return RMSEs


# all "full" simulations
# training_ids = range(simulation_ids[number_first_test] - 1)
# testing_ids = range(simulation_ids[number_first_test] + 1, simulation_ids[-1] + 1)
training_ids = range(99)
testing_ids = range(
    simulation_ids[number_first_test] + 1, simulation_ids[number_first_test] + 1 + 99
)

training_RMSEs = compute_RMSEs(
    my_nn, total_input_images, simulation_ids, simulation_times, training_ids
)
testing_RMSEs = compute_RMSEs(
    my_nn, total_input_images, simulation_ids, simulation_times, testing_ids
)

fig3 = plt.figure(figsize=(10, 5), dpi=144)  # layout="constrained")
gs = GridSpec(1, 1, hspace=0, wspace=0, figure=fig3)

ax = fig3.add_subplot(gs[0, 0])
times = np.arange(20)
training_mean = np.mean(training_RMSEs, 0)
testing_mean = np.mean(testing_RMSEs, 0)
training_std = np.std(training_RMSEs, 0)
testing_std = np.std(testing_RMSEs, 0)
ax.fill_between(
    times,
    training_mean[0:20] - training_std[0:20],
    training_mean[0:20] + training_std[0:20],
    color="blue",
    alpha=0.2,
)
ax.fill_between(
    times,
    testing_mean[0:20] - testing_std[0:20],
    testing_mean[0:20] + testing_std[0:20],
    color="green",
    alpha=0.2,
)
ax.plot(times, training_mean[0:20], label="training set", color="blue")
ax.plot(times, testing_mean[0:20], label="testing set", color="green")
ax.legend()
# ax.set_yscale("log")
ax.set_ylabel("% of rmse")
ax.set_xlabel("number of time step")
ax.set_xlim(0, 20)
ax.grid()

fig4 = plt.figure(figsize=(10, 5), dpi=144)  # layout="constrained")
gs = GridSpec(1, 1, hspace=0, wspace=0, figure=fig4)

ax = fig4.add_subplot(gs[0, 0])
ax.violinplot(
    testing_RMSEs[:, 1:20:2],
    np.arange(1, 20, 2),
    showmeans=True,
    showmedians=False,
    showextrema=True,
)
ax.set_ylabel("% of rmse")
ax.set_xlabel("number of time step")
ax.set_xlim(0, 20)

for ext in ["png", "pdf", "svg"]:
    fig3.savefig(prefix_to_plot + "_plot_rmse_vs_time." + ext)
    fig4.savefig(prefix_to_plot + "_plot_rmse_vs_time_violin." + ext)


########################
## Plot RMSEs vs Network
########################

prefixes_to_plot = [
    ("C:/Users/ruihe/GitHub/new-physics/results/NN_results20251004_173504", "nn 1"),
    ("C:/Users/ruihe/GitHub/new-physics/results/NN_results20251003_021025", "nn 2"),
    ("C:/Users/ruihe/GitHub/new-physics/results/NN_results20251003_071440", "nn 3"),
]

rmses = []
labels = [label for prefix, label in prefixes_to_plot]
time = 20
for prefix, label in prefixes_to_plot:
    my_nn.load_weights(prefix + "_best_weights.pkl")
    testing_RMSEs = compute_RMSEs(
        my_nn, total_input_images, simulation_ids, simulation_times, testing_ids
    )
    rmses.append(testing_RMSEs[:, 10])


fig5 = plt.figure(figsize=(10, 5), dpi=144)  # layout="constrained")
gs = GridSpec(1, 1, hspace=0, wspace=0, figure=fig5)

ax = fig5.add_subplot(gs[0, 0])
ax.violinplot(
    rmses,
    showmeans=True,
    showmedians=False,
    showextrema=True,
)
ax.set_ylabel("% of rmse")
ax.set_xticks(np.arange(1, len(labels) + 1), labels=labels)
ax.set_xlim(0.25, len(labels) + 0.75)

for ext in ["png", "pdf", "svg"]:
    fig5.savefig(prefix_to_plot + "_plot_rmse_vs_network_violin." + ext)

plt.show()
