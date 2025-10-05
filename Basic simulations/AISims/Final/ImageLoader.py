import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pickle
from numba import jit
import re
from tqdm import tqdm

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

    images_by_id = {}
    for file in tqdm(os.listdir(folder)):
        m = re.search("RGB_image(?P<time>[0-9]+)_(?P<id>[0-9]+).pkl", file)
        time = (int)(m["time"])
        id = (int)(m["id"])

        if not (id in images_by_id):
            images_by_id[id] = [None] * 30

        with open(f"{folder}" + "\\" + f"{file}", "rb") as filehandler:
            images_by_id[id][time] = pickle.load(filehandler)

    inputs = []
    labels = []
    for key in tqdm(sorted(images_by_id.keys())):
        for t in range(29):
            inputs.append(images_by_id[key][t])
            labels.append(images_by_id[key][t + 1])

    return inputs, labels

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


data_folder = "C:/Users/ruihe/GitHub/new-physics/Training_data_pickle"
input_training_file = "C:/Users/ruihe/GitHub/new-physics/Training_data_pickle_compressed/total_images_ordered.pkl"
output_training_file = "C:/Users/ruihe/GitHub/new-physics/Training_data_pickle_compressed/total_image_labels_ordered.pkl"
images, labels = ImageLoaderPKL(data_folder)

with open(input_training_file, "wb") as file:
    pickle.dump(images, file)
with open(output_training_file, "wb") as file:
    pickle.dump(labels, file)
