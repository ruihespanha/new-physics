import time
import copy
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class Methods(object):
    @staticmethod
    def UpdateSimArr(size, arr):
        tempArr = copy.copy(arr)
        for x in range(size):
            for y in range(size):
                tempSum = 0
                for i in range(3):
                    for j in range(3):
                        if i == 1 and j == 1:
                            tempSum += 0
                        else:
                            if (
                                x + i - 1 < 0
                                or x + i - 1 >= size
                                or y + j - 1 < 0
                                or y + j - 1 >= size
                            ):
                                tempSum += 0
                            else:
                                if arr[x + i - 1, y + j - 1] != 0:
                                    tempSum += 1
                                else:
                                    tempSum += 0
                if tempSum == 3:
                    tempArr[x, y] = 1
                elif tempSum == 2 and arr[x, y] == 1:
                    tempArr[x, y] = 1
                else:
                    tempArr[x, y] = 0
        return tempArr

    @staticmethod
    def PrintArr(size, arr):
        tempStr = ""
        for y in range(size):
            for x in range(size):
                if arr[x, y] == 1:
                    tempStr += "■"
                else:
                    tempStr += " "
            tempStr += "\n"
        return tempStr


size = 100
running = True
randArr = np.random.rand(size, size)
arr = np.zeros((size, size))
for x in range(size):
    for y in range(size):
        if randArr[x, y] > 0.3:
            arr[x, y] = 1

while True:
    arr = Methods.UpdateSimArr(size, arr)
    os.system("cls" if os.name == "nt" else "clear")
    print(Methods.PrintArr(size, arr))
    time.sleep(0.1)
