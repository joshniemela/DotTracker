 #!/usr/bin/python
# Created By  :  Josh Niemelä
# Created Date:  04/05/2022
# version = "1.0.0"
# ---------------------------------------------------------------------------
""" Program to make a position scatter plot of 2 red dots """
# ---------------------------------------------------------------------------
import numpy as np
import cv2
import os
from multiprocessing.pool import Pool
import matplotlib.pyplot as plt
import pandas as pd


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
    ob_list = []
    contours, hierarchy = cv2.findContours(
        thresh,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_NONE)
    for contour in contours:
        ob_list.append((calcpos(contour)))
    return(ob_list)

programpath = os.getcwd()

file_name = f"{programpath}/done/experiment.mp4"  # file name
reference = [110, 165, 145]  # LAB color to find
kernel = np.ones((5, 5), 'uint8')  # params for dilation
its = 15  # iterations for dilate
thresh_val = 8  # threshold value, lower for less sensitivity
ob_num = 2  # expected objects
fps = 50  # frames per second [Hz]
ppm = 663  # pixels per meter [m]
tstart = 8  # start time in seconds
tend = 60  # end time in seconds
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
    x = []
    y = []
    for obs in data:
        for ob in obs:
            x.append(ob[0])
            y.append(ob[1])
    plt.scatter(x, y)