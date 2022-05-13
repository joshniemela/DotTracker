#!/usr/bin/python
# Created By  :  Josh Niemelä
# Created Date:  04/05/2022
# version = "1.0.0"
# ---------------------------------------------------------------------------
""" Program to get energy, momentum, position and velocity of 2 red dots """
# ---------------------------------------------------------------------------
import numpy as np
import cv2
import os
from multiprocessing.pool import Pool
import matplotlib.pyplot as plt
import pandas as pd
import logging

def calcpos(contour):
    M = cv2.moments(contour)
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    return(cX, cY)


def compareAB(img, reference):
    img = img.astype(np.int16)

    A = img[:, :, 1]
    B = img[:, :, 2]

    A_diff = A-reference[1]
    B_diff = B-reference[2]

    diff = np.sqrt(np.square(A_diff)+np.square(B_diff))
    diff = diff.astype(np.uint8)

    return(diff)


def color_diff(frame, reference, threshold, dilate_size, dilate_iterations):
    LAB = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    diffed = compareAB(LAB, reference)
    thresh = cv2.threshold(diffed, thresh_val, 255, cv2.THRESH_BINARY_INV)[1]
    dilated = cv2.dilate(thresh, kernel=kernel, iterations=its)
    return(dilated)


def process_frame(img):
    thresh = color_diff(img, reference, thresh_val, kernel, its)

    contours, hierarchy = cv2.findContours(
        thresh,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_NONE)
    if hierarchy is None:
        return(np.zeros(4)*np.nan)  # return nan values
    if hierarchy.shape[1] == ob_num:
        x1, y1 = calcpos(contours[0])
        x2, y2 = calcpos(contours[1])
        if x1 > x2:
            return(np.array([y2, x2, y1, x1]))
        else:
            return(np.array([y1, x1, y2, x2]))
    else:
        print(f"found {hierarchy.shape[1]} obs")
        return(np.zeros(4)*np.nan)  # return nan values

logger = logging.getLogger("Logger")
programpath = os.getcwd()

file_name = f"{programpath}/experiment.mp4"  # file name
reference = [110, 165, 145]  # LAB color to find
kernel = np.ones((5, 5), 'uint8')  # params for dilation
its = 15  # iterations for dilate
thresh_val = 8  # threshold value, lower for less sensitivity
ob_num = 2  # expected objects
fps = 50  # frames per second [Hz]
ppm = 663  # pixels per meter [m], derived by the total lenght of ruler on the video
tstart = 0  # start time in seconds
tend = 3 # end time in seconds
processes = 12  # number of processes to be made by multiprocessing
# masses [kg]
m1 = 0.300
m2 = 0.300

if __name__ == "__main__":
    cap = cv2.VideoCapture(file_name)
    frames = []
    ret = True
    while ret:
        ret, frame = cap.read()
        frames.append(frame)
    frames.pop()
    print("done getting frames")
    with Pool(processes) as p:
        data = p.map(process_frame, frames)
        data = np.array(data).reshape(-1, 4)
    df = pd.DataFrame(data)
    df.index = df.index/fps
    df = df/ppm  # calculates SI unit for displacement
    df = df.interpolate(method="linear")  # remove to get true data
    df = df[tstart:tend]

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, sharex=True)
    ax1.plot(df.index, df[1], label="x1")
    ax1.plot(df.index, df[3], label="x2")

    # calculate momenta
    df = df.diff()
    df = df.rolling(3).mean()
    ax2.plot(df.index, df[1], label="v1")
    ax2.plot(df.index, df[3], label="v2")

    p1 = m1*df[1]
    p2 = m2*df[3]
    ax3.plot(df.index, p1, label="p1")
    ax3.plot(df.index, p2, label="p2")
    ax3.plot(df.index, abs(p1+p2), label="psum")

    e1 = 1/2*m1*df[1]**2
    e2 = 1/2*m2*df[3]**2

    ax4.plot(df.index, e1, label="e1")
    ax4.plot(df.index, e2, label="e2")
    ax4.plot(df.index, e1+e2, label="esum")

    ax1.legend()
    ax2.legend()
    ax3.legend()
    ax4.legend()
    ax1.set_ylabel("x[m]")
    ax2.set_ylabel("v[m/s]")
    ax3.set_ylabel("p[kg*m/s]")
    ax4.set_ylabel("E[J]")
    ax4.set_xlabel("t[s]")
    plt.savefig(f"{file_name}.png")
