import numpy as np
import cv2 as cv
from skimage import io
import matplotlib.pyplot as plt
from imutils.perspective import four_point_transform
from skimage.segmentation import clear_border
import imutils
from skimage.morphology import disk
from skimage.filters import threshold_otsu, threshold_local
from tensorflow import keras

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.metrics import f1_score,accuracy_score
#import joblib


def find_tables(img):
    blur = cv.GaussianBlur(img, (15, 15), 3)
    # ret3,th3 = cv.threshold(blur,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)

    th2 = cv.adaptiveThreshold(blur, 255,cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)

    thresh = cv.bitwise_not(th2)

    cnts = cv.findContours(thresh.copy(), cv.RETR_EXTERNAL,
                           cv.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv.contourArea, reverse=True)
    puzzleCnt = None

    for c in cnts:

        peri = cv.arcLength(c, True)
        approx = cv.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            puzzleCnt = approx
        break
    if puzzleCnt is None:
        raise Exception(("Could not find Sudoku puzzle outline. "
                         "Try debugging your thresholding and contour steps."))

    output = thresh.copy()
    output[:] = (0)
    cv.drawContours(output, [puzzleCnt], -1, (255, 0, 0), 5)
    cv.fillPoly(output, pts=[puzzleCnt], color=(255, 255, 255))

    ret3, mask = cv.threshold(output, 0, 255, cv.THRESH_OTSU)
    # mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel,iterations=2)

    puzzle = four_point_transform(thresh, puzzleCnt.reshape(4, 2))
    warped = four_point_transform(img, puzzleCnt.reshape(4, 2))
    return puzzle, mask


def find_tables2(img):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    # blur = cv2.GaussianBlur(gray, (15, 15), 0)
    # # ret3,th3 = cv.threshold(blur,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
    #
    # th2 = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
    #                             cv2.THRESH_BINARY, 17, 2)
    #
    # thresh = cv2.bitwise_not(th2)
    #
    # cnts = cv2.findContours(thresh.copy(), cv2.RETR_TREE,
    #                         cv2.CHAIN_APPROX_SIMPLE)
    # cnts = imutils.grab_contours(cnts)
    # cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    #
    # puzzleCnt = []
    # areaArray = []
    # count = 0
    #
    # for c in cnts:
    #
    #     peri = cv2.arcLength(c, True)
    #     approx = cv2.approxPolyDP(c, 0.04 * peri, True)
    #     area = cv2.contourArea(c)
    #     areaArray.append(area)
    #     # if(area>1000000) & (area<10000000) &len(approx)==4 :
    #     if len(approx) == 4 and area > 600000:
    #         puzzleCnt.append(approx)
    #         count += 1
    #     output = thresh.copy()
    #     output[:] = (0)
    #     puzzle = []
    #     warped = []
    # for k in range(0, count):
    #     cv2.drawContours(output, [puzzleCnt[k]], -1, (255, 0, 0), 5)
    #     cv2.fillPoly(output, pts=[puzzleCnt[k]], color=(255, 255, 255))
    #     mask = cv2.threshold(output, 0, 255, cv2.THRESH_OTSU)
    #     puzzle.append(four_point_transform(thresh, puzzleCnt[k].reshape(4, 2)))
    #     warped.append(four_point_transform(gray, puzzleCnt[k].reshape(4, 2)))
    # return puzzle, warped, mask

    blurred = cv.GaussianBlur(gray, (15, 15), 3)
    thresh = cv.adaptiveThreshold(blurred, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)
    thresh = cv.bitwise_not(thresh)
    cnts = cv.findContours(thresh.copy(), cv.RETR_EXTERNAL,
                            cv.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv.contourArea, reverse=True)
    puzzleCnt = []
    areaArray = []
    count = 0
    for c in cnts:
        # approximate the contour
        peri = cv.arcLength(c, True)
        approx = cv.approxPolyDP(c, 0.02 * peri, True)
        # if our approximated contour has four points, then we can
        # assume we have found the outline of the puzzle
        area = cv.contourArea(c)
        if len(approx) == 4 and area > 600000:
            puzzleCnt.append(approx)
            count += 1
    output = thresh.copy()
    output[:] = (0)
    puzzle = []
    warped = []
    mask = 0
    for k in range(0, count):
        cv.drawContours(output, [puzzleCnt[k]], -1, (255, 0, 0), 5)
        cv.fillPoly(output, pts=[puzzleCnt[k]], color=(255, 255, 255))
        ret3, mask = cv.threshold(output, 0, 255, cv.THRESH_OTSU)
        # cv2.imshow('dsad',mask)
        # cv2.waitKey(0)
        puzzle.append(four_point_transform(image, puzzleCnt[k].reshape(4, 2)))
        warped.append(four_point_transform(gray, puzzleCnt[k].reshape(4, 2)))
    return (puzzle, warped, mask)


def nn_preproccess(img):
    resized_image = cv.resize(img, (28, 28))
    #pred = pred.reshape(784,)
    pred = (resized_image / 255.0)
    return pred

def number_detection_NN(image,model,finalgrid):
    image = image.reshape(1, 28, 28, 1).astype('float32')
    percentFilled = cv.countNonZero(finalgrid) / float(finalgrid.shape[0]* finalgrid.shape[1])
    if percentFilled < 0.03:
        return -1
    else:
        return model.predict(image).argmax()


def recognize_digits(puzzle):
    kernel = np.ones((4, 4))
    puzzle2 = cv.morphologyEx(puzzle, cv.MORPH_OPEN, kernel, iterations=1)
    puzzle3 = cv.morphologyEx(puzzle2, cv.MORPH_CLOSE, kernel, iterations=2)

    thresh = cv.threshold(puzzle3, 0, 255,
                          cv.THRESH_BINARY | cv.THRESH_OTSU)[1]
    thresh = clear_border(thresh)

    edge_h = np.shape(thresh)[0]
    edge_w = np.shape(thresh)[1]
    celledge_h = edge_h // 9
    celledge_w = edge_w // 9

    tempgrid = []
    for i in range(celledge_h, edge_h + 1, celledge_h):
        for j in range(celledge_w, edge_w + 1):
            rows = thresh[i - celledge_h:i]
            tempgrid.append([rows[k][j - celledge_w:j] for k in range(len(rows))])

    finalgrid = []
    for i in range(0, len(tempgrid) - 8, 1):
        finalgrid.append(tempgrid[i:i + 9])
    for i in range(9):
        for j in range(9):
            finalgrid[i][j] = np.array(finalgrid[i][j])
            finalgrid[i][j] = cv.threshold(finalgrid[i][j], 0, 255,
                                           cv.THRESH_BINARY | cv.THRESH_OTSU)[1]
            finalgrid[i][j] = clear_border(finalgrid[i][j])
    return finalgrid, tempgrid

def resizing(image):
    resized_img = cv.resize(image, (28,28))
    image_resize_2 = resized_img.reshape(784,)
    return image_resize_2.reshape(1,-1)

def number_detection(image,model):
    percentFilled = cv.countNonZero(image) / float(image.shape[0]* image.shape[1])
    if percentFilled < 0.03:
        return -1
    else:
        return model.predict(image)[0]

SCALE = 0.33


def predict_image(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    puzzle, mask = find_tables(gray)

    finalgrid, tempgrid = recognize_digits(puzzle[0])
    #model = joblib.load('/autograder/submission/random_forest_ovo.joblib')
    model = keras.models.load_model('C:/Users/PinkeyDinkey/Desktop/sudoku/Model.h5')#/autograder/submission/model.h5

    sudoku_digits = [
        np.array([[-1, -1, -1, -1, -1, -1, -1, -1, -1],
                  [-1, -1, -1, 8, 9, 4, -1, -1, -1],
                  [-1, -1, -1, 6, -1, 1, -1, -1, -1],
                  [-1, 6, 5, 1, -1, 9, 7, 8, -1],
                  [-1, 1, -1, -1, -1, -1, -1, 3, -1],
                  [-1, 3, 9, 4, -1, 5, 6, 1, -1],
                  [-1, -1, -1, 8, -1, 2, -1, -1, -1],
                  [-1, -1, -1, 9, 1, 3, -1, -1, -1],
                  [-1, -1, -1, -1, -1, -1, -1, -1, -1]]),
    ]

    answers = []
    for i in range(0, len(tempgrid) - 8, 9):
        answers.append(tempgrid[i:i + 9])
    # Converting all the cell images to np.array
    for i in range(9):
        for j in range(9):
            answers[i][j] = nn_preproccess(finalgrid[i][j])
            answers[i][j] = number_detection_NN(answers[i][j], model, finalgrid[i][j])
            sudoku_digits[0][i, j] = answers[i][j]


    return mask, sudoku_digits

image = cv.imread('train/1111.png')
mask , digits = predict_image(image)
