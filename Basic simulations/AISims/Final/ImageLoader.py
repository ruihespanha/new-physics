import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pickle
from numba import jit

# working on this one


def ImageLoader(folder):
    IMG_SIZE = 64
    images = []
    labels = []
    indexes = []  # all the sorted images (2D)
    indexOrderLabels = []  # the order in the simulation of arrays in indexes (2D)
    indexLabels = []  # the id of the simulatin for the sub arrays (1d)
    label_num = 0

    for file in os.listdir(folder):
        temp_found = False
        for i in range(len(indexLabels)):
            if (
                int(
                    (
                        file[file.find("_") + 1 :][
                            file[file.find("_") + 1 :].find("_") + 1 :
                        ]
                    )[
                        : len(
                            (
                                file[file.find("_") + 1 :][
                                    file[file.find("_") + 1 :].find("_") + 1 :
                                ]
                            )
                        )
                        - 4
                    ]
                )
                == indexLabels[i]
            ):
                temp_found = True
                try:
                    img = Image.open(f"{folder}" + "\\" + f"{file}").resize((12, 24))
                    indexes[i].append(np.array(img) / 255.0)
                    indexOrderLabels[i].append(
                        int(
                            (file[file.find("e") + 1 :])[
                                : file[file.find("e") + 1 :].find("_")
                            ]
                        )
                    )
                except NameError:
                    print(f"Error loading image: {file} : {NameError}")

        if temp_found == False:
            indexLabels.append(
                int(
                    (
                        file[file.find("_") + 1 :][
                            file[file.find("_") + 1 :].find("_") + 1 :
                        ]
                    )[
                        : len(
                            (
                                file[file.find("_") + 1 :][
                                    file[file.find("_") + 1 :].find("_") + 1 :
                                ]
                            )
                        )
                        - 4
                    ]
                )
            )
            indexes.append([])
            indexOrderLabels.append([])
            try:
                img = Image.open(f"{folder}" + "\\" + f"{file}").resize((12, 24))
                indexes[len(indexes) - 1].append(np.array(img) / 255.0)
                indexOrderLabels[len(indexOrderLabels) - 1].append(
                    int(
                        (file[file.find("e") + 1 :])[
                            : file[file.find("e") + 1 :].find("_")
                        ]
                    )
                )
            except NameError:
                print(f"Error loading image: {file} : {NameError}")

    for i in range(len(indexes)):
        for j in range(len(indexes[i])):
            tempLabels = indexOrderLabels[i][j]
            if tempLabels != 29:
                images.append(indexes[i][j])
                for u in range(len(indexes[i])):
                    if tempLabels == indexOrderLabels[i][u] - 1:
                        labels.append(indexes[i][u])

    return images, labels


def ImageLoaderPKL(folder):
    images = []
    labels = []
    indexes = []  # all the sorted images (2D)
    indexOrderLabels = []  # the order in the simulation of arrays in indexes (2D)
    indexLabels = []  # the id of the simulatin for the sub arrays (1d)
    label_num = 0
    COUNT = 0

    for file in os.listdir(folder):
        COUNT += 1
        if COUNT % 25 == 0:
            print(f"{COUNT}")
        temp_found = False
        for i in range(len(indexLabels)):
            if (
                int(
                    (
                        file[file.find("_") + 1 :][
                            file[file.find("_") + 1 :].find("_") + 1 :
                        ]
                    )[
                        : len(
                            (
                                file[file.find("_") + 1 :][
                                    file[file.find("_") + 1 :].find("_") + 1 :
                                ]
                            )
                        )
                        - 4
                    ]
                )
                == indexLabels[i]
            ):
                temp_found = True
                try:

                    with open(f"{folder}" + "\\" + f"{file}", "rb") as filehandler:
                        loaded_variable = pickle.load(filehandler)

                    indexes[i].append(loaded_variable)
                    indexOrderLabels[i].append(
                        int(
                            (file[file.find("e") + 1 :])[
                                : file[file.find("e") + 1 :].find("_")
                            ]
                        )
                    )
                except NameError:
                    print(f"Error loading image: {file} : {NameError}")

        if temp_found == False:
            indexLabels.append(
                int(
                    (
                        file[file.find("_") + 1 :][
                            file[file.find("_") + 1 :].find("_") + 1 :
                        ]
                    )[
                        : len(
                            (
                                file[file.find("_") + 1 :][
                                    file[file.find("_") + 1 :].find("_") + 1 :
                                ]
                            )
                        )
                        - 4
                    ]
                )
            )
            indexes.append([])
            indexOrderLabels.append([])
            try:
                with open(f"{folder}" + "\\" + f"{file}", "rb") as filehandler:
                    loaded_variable = pickle.load(filehandler)

                indexes[len(indexes) - 1].append(loaded_variable)
                indexOrderLabels[len(indexOrderLabels) - 1].append(
                    int(
                        (file[file.find("e") + 1 :])[
                            : file[file.find("e") + 1 :].find("_")
                        ]
                    )
                )
            except NameError:
                print(f"Error loading image: {file} : {NameError}")

    for i in range(len(indexes)):
        for j in range(len(indexes[i])):
            tempLabels = indexOrderLabels[i][j]
            if tempLabels != 29:
                images.append(indexes[i][j])
                for u in range(len(indexes[i])):
                    if tempLabels == indexOrderLabels[i][u] - 1:
                        labels.append(indexes[i][u])

    return images, labels
