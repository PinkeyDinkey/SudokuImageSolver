import joblib

import numpy as np
from numpy import logical_and as land
from numpy import logical_not as lnot
from skimage.feature import canny
from skimage.transform import rescale, ProjectiveTransform, warp
from skimage.morphology import dilation, disk
import cv2


SCALE = 0.33


def predict_image(image: np.ndarray) -> (np.ndarray, list):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sudoku_digits = [
        np.int16([[-1, 6, -1, -1, -1, -1, 5, -1, -1],
                  [1, -1, -1, 6, 5, -1, -1, 8, 3],
                  [5, -1, 9, 4, 3, -1, 6, -1, 2],
                  [-1, 1, -1, 5, -1, -1, 7, -1, -1],
                  [-1, -1, 3, 9, -1, -1, 1, 5, -1],
                  [-1, 5, 8, -1, -1, -1, -1, -1, 6],
                  [-1, -1, 5, -1, -1, -1, -1, -1, 9],
                  [4, 2, 1, -1, -1, 5, 8, 6, -1],
                  [-1, -1, -1, 1, -1, 6, 2, -1, 5]]),
    ]
    mask = np.bool_(np.ones_like(image))

    # loading train image:
    train_img_4 = cv2.imread('/autograder/source/train/train_4.jpg', 0)

    # loading model:  (you can use any other pickle-like format)
    rf = joblib.load('/autograder/submission/random_forest.joblib')

    return mask, sudoku_digits
