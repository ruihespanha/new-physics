# import readline

import torch
import math
import numpy as np
import random
import matplotlib.pyplot as plt
import dash
import time
from phi.torch.flow import *
from phi.flow import *
import random
from PIL import Image
import pickle

backend.default_backend().list_devices("GPU")
backend.default_backend().set_default_device("GPU")
COUNT = 100


for i in range(900):
    simHeight = 24
    simWidth = 12
    resolution = 1
    nrows, ncols = simHeight * resolution, simWidth * resolution
    radius = 3 * resolution
    inflow_size = (4 * resolution, 8 * resolution)
    dt = 0.05 * resolution

    settings = batch(setting=3)

    def potential(pos):
        return pos[0] + pos[1]

    smoke = CenteredGrid(
        0, extrapolation.BOUNDARY, x=ncols, y=nrows, bounds=Box(x=ncols, y=nrows)
    )  # sampled at cell centers

    velocity = StaggeredGrid(
        0, extrapolation.BOUNDARY, x=ncols, y=nrows, bounds=Box(x=ncols, y=nrows)
    )  # sampled in staggered form at face centers

    INFLOW_LOCATION = tensor([inflow_size], batch("inflow_loc"), channel(vector="x,y"))
    INFLOW = 1 * CenteredGrid(
        Box(
            x=(2, simWidth * resolution - 2),
            y=(1, 2),
        ),
        extrapolation.BOUNDARY,
        x=ncols,
        y=nrows,
        bounds=Box(x=ncols, y=nrows),
    )

    INFLOWVELOCITY = 1 * StaggeredGrid(
        Box(
            x=(2, simWidth * resolution - 2),
            y=(1, 2),
        ),
        extrapolation.BOUNDARY,
        x=ncols,
        y=nrows,
        bounds=Box(x=ncols, y=nrows),
    )

    objects_numpy = np.zeros((simHeight * resolution, simWidth * resolution))
    OBJECTS = [
        # these are the left and right boundaries
        Obstacle(
            Box(
                x=(0, 1),
                y=(0, simHeight * resolution),
            )
        ),
        Obstacle(
            Box(
                x=((simWidth * resolution) - 1, simWidth * resolution),
                y=(0, simHeight * resolution),
            )
        ),
        # Obstacle(
        #     Box(x=(4 * resolution, 8 * resolution), y=(12 * resolution, 14 * resolution))
        # ),
        # Obstacle(Box(x=(0, 8 * resolution), y=(9 * resolution, 11 * resolution))),
        # Obstacle(
        #     Box(
        #         x=(10 * resolution, simLength * resolution),
        #         y=(5 * resolution, 6 * resolution),
        #     )
        # ),
        # Obstacle(
        #     Box(x=(2 * resolution, 15 * resolution), y=(18 * resolution, 19 * resolution))
        # ),
    ]
    objects_numpy[:, 0:1] = 1
    objects_numpy[:, simWidth * resolution - 1 : simWidth * resolution] = 1
    # objects_numpy[10:20, 10:20] = 1

    for i in range(10):
        randx = random.randint(0, simWidth - 2)
        randy = random.randint(4, simHeight - 2)
        OBJECTS.append(
            Obstacle(
                Box(
                    x=(randx * resolution, (randx + 2) * resolution),
                    y=(randy * resolution, (randy + 2) * resolution),
                )
            )
        )

        objects_numpy[
            randy * resolution : (randy + 2) * resolution,
            randx * resolution : (randx + 2) * resolution,
        ] = 1

    amountObstacles = 3
    # for i in range(amountObstacles):
    #     # this is a box
    #     OBJECTS.append(
    #         Obstacle(
    #             Box(
    #                 x=(
    #                     (random.randint(1, simLength) * resolution),
    #                     (random.randint(1, simLength) * resolution),
    #                 ),
    #                 y=(
    #                     (random.randint(1, simWidth) * resolution),
    #                     (random.randint(1, simWidth) * resolution),
    #                 ),
    #             )
    #         )
    #     )
    # OBJECTS_BOOL = np.zeros(shape=(ncols, nrows))
    # OBJECTS_BOOL[3:4, 6:10] = 1
    # OBJECTS_BOOL[0:2, 6:10] = 1

    print(f"Smoke: {smoke.shape}")
    print(f"Velocity: {velocity.shape}")
    goobagoobagoo = rename_dims(smoke.points, spatial, batch)
    print(f"goo: {goobagoobagoo}")
    print(f"Inflow: {INFLOW.shape}")
    print(f"Inflow, spatial only: {INFLOW.shape.spatial}")

    print(smoke.values)
    print(velocity.values)
    print(INFLOW.values)

    smoke += INFLOW
    velocity += INFLOWVELOCITY

    buoyancy_force = smoke * (0, 0) @ velocity

    velocity = buoyancy_force

    velocity, _ = fluid.make_incompressible(velocity)

    fig = plt.figure(1, figsize=(10, 10))
    fig.clear()
    ax = fig.subplots(1, 5)
    ax[1].imshow(objects_numpy, interpolation="nearest", origin="upper")

    smoke_np = smoke.numpy().reshape(ncols, nrows).T
    print(f"range = {np.min(smoke_np)}, {np.max(smoke_np)}")
    # blurs https://matplotlib.org/stable/gallery/images_contours_and_fields/interpolation_methods.html
    g = ax[0].imshow(smoke_np, vmin=0, vmax=15, cmap="viridis", interpolation="none")
    k1 = ax[2].imshow(smoke_np, vmin=0, vmax=15, cmap="twilight", interpolation="none")
    k2 = ax[3].imshow(smoke_np, vmin=0, vmax=15, cmap="twilight", interpolation="none")
    k3 = ax[4].imshow(smoke_np, vmin=0, vmax=15, interpolation="none")
    ax[0].invert_yaxis()
    ax[1].invert_yaxis()
    ax[2].invert_yaxis()
    ax[3].invert_yaxis()
    ax[4].invert_yaxis()
    fig.canvas.draw()
    plt.show(block=False)
    t0 = time.time()
    trajectory = [smoke]

    @jit_compile
    def step(smoke, velocity, INFLOW, dt):
        smoke = advect.mac_cormack(smoke, velocity, dt=dt) + INFLOW
        buoyancy_force = smoke * (0, 0) @ velocity
        INFLOW_VELOCITY = INFLOWVELOCITY * (0, 5) @ velocity
        velocity = (
            advect.semi_lagrangian(velocity, velocity, dt=dt)
            + buoyancy_force
            + INFLOW_VELOCITY
        )
        velocity, _ = fluid.make_incompressible(
            velocity, OBJECTS, Solve("auto", 1e-5, 0, x0=None)
        )
        return smoke, velocity

    rand_index = COUNT
    COUNT += 1

    rgb_start = np.zeros((resolution * simHeight, resolution * simWidth, 3))
    rgb_start[:, :, 0] = objects_numpy
    rgb_start[:, :, 1] = np.ones((resolution * simHeight, resolution * simWidth)) / 2
    rgb_start[:, :, 2] = np.ones((resolution * simHeight, resolution * simWidth)) / 2
    # PIL_image = Image.fromarray(np.uint8(rgb_start * 255)).convert("RGB")
    # img = Image.fromarray(rgb)
    with open(
        f"C:/Users/ruihe/GitHub/Physics-based-Machine-learning-Fluid-sim/Training_data_pickle/RGB_image0_{rand_index}.pkl",
        "wb",
    ) as file:
        pickle.dump((rgb_start), file)
    # print(res)
    # PIL_image.save(
    #     f"C:/Users/ruihe/GitHub/Physics-based-Machine-learning-Fluid-sim/Training_data/RGB_image0_{rand_index}.png",
    #     "PNG",
    # )
    for v in range(29):
        n = v + 1
        print(n, end=" ")

        (
            smoke,
            velocity,
        ) = step(smoke, velocity, INFLOW, dt)
        smoke_np = smoke.numpy().reshape(ncols, nrows).T

        maxval = 0
        # print(f"length:{len(velocity.numpy())}")
        # print(f"zero:{len(velocity.numpy()[1][0])}")
        # tempArrX = numpy.zeros((resolution * simHeight, resolution * simWidth + 1))
        # tempArrY = numpy.zeros((resolution * simHeight + 1, resolution * simWidth))
        # for j in range(resolution * simWidth + 1):
        #     for i in range(resolution * simHeight):
        #         tempArrX[i, j] = ((velocity.numpy()[0][j][i] + 10) / 20) * 15
        # for j in range(resolution * simWidth):
        #     for i in range(resolution * simHeight + 1):
        #         tempArrY[i, j] = ((-velocity.numpy()[1][j][i] + 10) / 20) * 15

        tempArrX = numpy.zeros((resolution * simHeight, resolution * simWidth))
        tempArrY = numpy.zeros((resolution * simHeight, resolution * simWidth))
        rgb = np.zeros((resolution * simHeight, resolution * simWidth, 3))
        rgb[:, :, 0] = objects_numpy
        for j in range(resolution * simWidth):
            for i in range(resolution * simHeight):
                temp = (
                    (((velocity.numpy()[0][j][i] + 10) / 20))
                    + (((velocity.numpy()[0][j + 1][i] + 10) / 20))
                ) / 2
                tempArrX[i, j] = temp * 15
                rgb[i, j, 2] = temp
                # if temp < 0.5:
                #     rgb[i, j, 2] = 1 - temp * 2
                # else:
                #     rgb[i, j, 1] = (temp - 0.5) * 2
        for j in range(resolution * simWidth):
            for i in range(resolution * simHeight):
                temp = (
                    (((-velocity.numpy()[1][j][i] + 10) / 20))
                    + (((-velocity.numpy()[1][j][i + 1] + 10) / 20))
                ) / 2
                tempArrY[i, j] = temp * 15
                rgb[i, j, 1] = 1 - temp

        print(f"time = {time.time()-t0} range = {np.min(smoke_np)}, {np.max(smoke_np)}")
        t0 = time.time()

        # PIL_image = Image.fromarray(np.uint8(rgb * 255)).convert("RGB")
        # res = pickle.dumps(np.uint8(rgb))
        # print(res)
        # img = Image.fromarray(rgb)
        # PIL_image.save(
        #     f"C:/Users/ruihe/GitHub/Physics-based-Machine-learning-Fluid-sim/Training_data/RGB_image{n}_{rand_index}.png",
        #     "PNG",
        # )
        with open(
            f"C:/Users/ruihe/GitHub/Physics-based-Machine-learning-Fluid-sim/Training_data_pickle/RGB_image{n}_{rand_index}.pkl",
            "wb",
        ) as file:
            pickle.dump((rgb), file)

        g.set_data(smoke_np)
        k1.set_data(tempArrX)
        k2.set_data(tempArrY)
        k3.set_data(rgb)
        fig.canvas.draw()
        fig.canvas.flush_events()

    plt.show(block=False)
