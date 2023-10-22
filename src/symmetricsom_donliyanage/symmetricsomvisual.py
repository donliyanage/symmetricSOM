# Author Don SMA Liyanage (don.liyanage@live.com)


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys

import datetime

np.set_printoptions(threshold=sys.maxsize)

SOM_MAP = None
SOM_LABEL_MAP = None


def closestNode(data, t, map, m_rows, m_cols):
    result = (0, 0)
    small_dist = 1.0e20
    for i in range(m_rows):
        for j in range(m_cols):
            ed = euclideanDistance(map[i][j], data[t])
            if ed < small_dist:
                small_dist = ed
                result = (i, j)
    return result


def euclideanDistance(v1, v2):
    return np.linalg.norm(v1 - v2)


def manhattonDistance(r1, c1, r2, c2):
    return np.abs(r1 - r2) + np.abs(c1 - c2)


def mostRelevece(lst, n):
    if len(lst) == 0: return -1
    counts = np.zeros(shape=n, dtype=np.int)
    for i in range(len(lst)):
        counts[lst[i]] += 1
    return np.argmax(counts)


def pointerValue(r, c, lstmap):
    rtnVal = 0
    try:
        rtnVal = lstmap[r][c]
        if rtnVal == -1:
            rtnVal = 0
    except:
        rtnVal = 0

    return rtnVal


def Features(dsFEATURESET, dsLABEL, nDimentions, nRows=30, nCols=30, nLearnMax=0.5, nStepsMax=175000, fSAVESOM=True,
             nSAVESOM="SOM_MAP.DAT", nSAVESOMLABEL="SOM_LABEL.DAT"):
    print(datetime.datetime.now())
    np.random.seed(1)
    dimensions = nDimentions  # Per Transformed features
    rows = nRows  # Per Oriantation and Symmetry
    cols = nCols  # Per Oriantation and Symmetry
    rangeMax = rows + cols
    learnMax = nLearnMax  # 0.5
    stepsMax = nStepsMax

    # Data
    _data_SET = dsFEATURESET
    _data_LABEL = dsLABEL

    # Construct SOM
    map = np.random.random_sample(size=(rows, cols, dimensions))
    for s in range(stepsMax):
        if s % (stepsMax / 10) == 0:
            print("step = ", str(s))
        pct_left = 1.0 - ((s * 1.0) / stepsMax)
        curr_range = (int)(pct_left * rangeMax)
        curr_rate = pct_left * learnMax

        t = np.random.randint(len(_data_SET))
        (bmu_row, bmu_col) = closestNode(_data_SET, t, map, rows, cols)
        for i in range(rows):
            for j in range(cols):
                if manhattonDistance(bmu_row, bmu_col, i, j) < curr_range:
                    map[i][j] = map[i][j] + curr_rate * (_data_SET[t] - map[i][j])

    # Construct U-Matrix
    u_matrix = np.zeros(shape=(rows, cols), dtype=np.float64)
    for i in range(rows):
        for j in range(cols):
            v = map[i][j]  # single vector
            sumDiststance = 0.0
            ct = 0

            if i - 1 >= 0:  # above
                sumDiststance += euclideanDistance(v, map[i - 1][j]);
                ct += 1
            if i + 1 <= rows - 1:  # below
                sumDiststance += euclideanDistance(v, map[i + 1][j]);
                ct += 1
            if j - 1 >= 0:  # left
                sumDiststance += euclideanDistance(v, map[i][j - 1]);
                ct += 1
            if j + 1 <= cols - 1:  # right
                sumDiststance += euclideanDistance(v, map[i][j + 1]);
                ct += 1

            u_matrix[i][j] = sumDiststance / ct

    global SOM_MAP
    SOM_MAP = map
    if fSAVESOM is True:
        np.save(nSAVESOM, SOM_MAP)

    # Show U-Matrix - Close Clusters
    plt.imshow(u_matrix, cmap='gray')
    plt.show()

    # Show Labled Visualisation
    mapping = np.empty(shape=(rows, cols), dtype=object)
    for i in range(rows):
        for j in range(cols):
            mapping[i][j] = []

    for t in range(len(_data_SET)):
        (m_row, m_col) = closestNode(_data_SET, t, map, rows, cols)
        mapping[m_row][m_col].append(_data_LABEL[t])

    label_map = np.zeros(shape=(rows, cols), dtype=np.int)
    for i in range(rows):
        for j in range(cols):
            label_map[i][j] = mostRelevece(mapping[i][j], 3)

    plt.imshow(label_map, cmap=plt.cm.get_cmap('terrain_r', 4))
    plt.colorbar()
    plt.show()

    global SOM_LABEL_MAP
    SOM_LABEL_MAP = label_map
    if fSAVESOM is True:
        np.save(nSAVESOMLABEL, SOM_LABEL_MAP)

    print(datetime.datetime.now())


def predictValidate(dsFeatureSet, dsLabel, nRows=30, nCols=30):
    (m_row, m_col) = closestNode(dsFeatureSet, 0, SOM_MAP, nRows, nCols)

    matching_l = pointerValue(m_row, m_col, SOM_LABEL_MAP)

    u_l = pointerValue(m_row - 1, m_col, SOM_LABEL_MAP)
    b_l = pointerValue(m_row + 1, m_col, SOM_LABEL_MAP)
    r_l = pointerValue(m_row, m_col - 1, SOM_LABEL_MAP)
    l_l = pointerValue(m_row, m_col + 1, SOM_LABEL_MAP)

    ur_l = pointerValue(m_row - 1, m_col + 1, SOM_LABEL_MAP)
    ul_l = pointerValue(m_row - 1, m_col - 1, SOM_LABEL_MAP)
    br_l = pointerValue(m_row + 1, m_col + 1, SOM_LABEL_MAP)
    bl_l = pointerValue(m_row + 1, m_col - 1, SOM_LABEL_MAP)

    average_l_0 = (u_l + b_l + r_l + l_l) / 4
    average_l_1 = (ur_l + ul_l + br_l + bl_l) / 4
    average_l = (average_l_0 + average_l_1) / 2

    (matching_l, average_l)


if __name__ == '__main__':
    Features()



