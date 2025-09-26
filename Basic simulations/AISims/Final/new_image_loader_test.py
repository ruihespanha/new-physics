import numpy as np
import ImageLoader
import os
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pickle


path = "C:/Users/ruihe/GitHub/Physics-based-Machine-learning-Fluid-sim/Training_data_pickle"

path2 = "C:/Users/ruihe/GitHub/Physics-based-Machine-learning-Fluid-sim/Training_data"
# ims, labels = ImageLoader.ImageLoader(path2)

ims, labels = ImageLoader.ImageLoaderPKL(path)

print(ims[1000][:, :, 0])

fig, ax = plt.subplots(2)

ax[0].imshow(ims[1000][:, :, 0], vmin=0, vmax=1, cmap="plasma", interpolation="none")
ax[1].imshow(labels[1000][:, :, 0], vmin=0, vmax=1, cmap="plasma", interpolation="none")

plt.show()
# loaded_variable = 0
# testfile = np.random.rand(12, 24) * 2
# print(testfile)

# with open(
#     f"C:/Users/ruihe/Downloads/" + f"test.pkl",
#     "wb",
# ) as file:
#     pickle.dump(np.uint8(testfile), file)


# with open(f"C:/Users/ruihe/Downloads/" + f"test.pkl", "rb") as filehandler:
#     loaded_variable = pickle.load(filehandler)

# print(loaded_variable)
