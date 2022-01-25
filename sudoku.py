from imutils.perspective import four_point_transform
import imutils
from imutils import contours
from skimage.segmentation import clear_border
import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import mnist
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report

from random import shuffle, seed as random_seed, randrange
import sys


class _SudokuSolver:
    def __init__(self, sudoku):
        self.width = sudoku.width
        self.height = sudoku.height
        self.size = sudoku.size
        self.sudoku = sudoku

    def _solve(self):
        blanks = self.__get_blanks()
        blank_count = len(blanks)
        are_blanks_filled = [False for _ in range(blank_count)]
        blank_fillers = self.__calculate_blank_cell_fillers(blanks)
        solution_board = self.__get_solution(
            Sudoku._copy_board(self.sudoku.board), blanks, blank_fillers, are_blanks_filled)
        solution_difficulty = 0
        if not solution_board:
            solution_board = Sudoku.empty(self.width, self.height).board
            solution_difficulty = -2
        return Sudoku(self.width, self.height, board=solution_board, difficulty=solution_difficulty)

    def __calculate_blank_cell_fillers(self, blanks):
        sudoku = self.sudoku
        valid_fillers = [[[True for _ in range(self.size)] for _ in range(self.size)] for _ in range(self.size)]
        for row, col in blanks:
            for i in range(self.size):
                same_row = sudoku.board[row][i]
                same_col = sudoku.board[i][col]
                if same_row and i != col:
                    valid_fillers[row][col][same_row - 1] = False
                if same_col and i != row:
                    valid_fillers[row][col][same_col - 1] = False
            grid_row, grid_col = row // sudoku.height, col // sudoku.width
            grid_row_start = grid_row * sudoku.height
            grid_col_start = grid_col * sudoku.width
            for y_offset in range(sudoku.height):
                for x_offset in range(sudoku.width):
                    if grid_row_start + y_offset == row and grid_col_start + x_offset == col:
                        continue
                    cell = sudoku.board[grid_row_start + y_offset][grid_col_start + x_offset]
                    if cell:
                        valid_fillers[row][col][cell - 1] = False
        return valid_fillers

    def __get_blanks(self):
        blanks = []
        for i, row in enumerate(self.sudoku.board):
            for j, cell in enumerate(row):
                if cell == Sudoku._empty_cell_value:
                    blanks += [(i, j)]
        return blanks

    def __valid_fillers_for(self, board, pos):
        row, col = pos
        fillers = list(range(1, self.size + 1))

        # Check same row and column
        for i in range(self.size):
            if board[row][i]:
                fillers[board[row][i] - 1] = Sudoku._empty_cell_value
            if board[i][col]:
                fillers[board[i][col] - 1] = Sudoku._empty_cell_value

        # Check same grid
        grid_row, grid_col = row // self.height, col // self.width
        row_start = grid_row * self.height
        col_start = grid_col * self.width
        for y_offset in range(self.height):
            for x_offset in range(self.width):
                if board[row_start + y_offset][col_start + x_offset]:
                    fillers[board[row_start + y_offset]
                            [col_start + x_offset] - 1] = Sudoku._empty_cell_value
        return [f for f in fillers if f != Sudoku._empty_cell_value]

    def __is_neighbor(self, blank1, blank2):
        row1, col1 = blank1
        row2, col2 = blank2
        if row1 == row2 or col1 == col2:
            return True
        grid_row1, grid_col1 = row1 // self.height, col1 // self.width
        grid_row2, grid_col2 = row2 // self.height, col2 // self.width
        return grid_row1 == grid_row2 and grid_col1 == grid_col2

    # Optimized version of above
    def __get_solution(self, board, blanks, blank_fillers, are_blanks_filled):
        min_filler_count = None
        chosen_blank = None
        for i, blank in enumerate(blanks):
            x, y = blank
            if are_blanks_filled[i]:
                continue
            valid_filler_count = sum(blank_fillers[x][y])
            if valid_filler_count == 0:
                # Blank cannot be filled with any number, no solution
                return None
            if not min_filler_count or valid_filler_count < min_filler_count:
                min_filler_count = valid_filler_count
                chosen_blank = blank
                chosen_blank_index = i

        if not chosen_blank:
            # All blanks have been filled with valid values, return this board as the solution
            return board

        row, col = chosen_blank

        # Declare chosen blank as filled
        are_blanks_filled[chosen_blank_index] = True

        # Save list of neighbors affected by the filling of current cell
        revert_list = [False for _ in range(len(blanks))]

        for number in range(self.size):
            # Only try filling this cell with numbers its neighbors aren't already filled with
            if not blank_fillers[row][col][number]:
                continue

            # Test number in this cell, number + 1 is used because number is zero-indexed
            board[row][col] = number + 1

            for i, blank in enumerate(blanks):
                blank_row, blank_col = blank
                if blank == chosen_blank:
                    continue
                if self.__is_neighbor(blank, chosen_blank) and blank_fillers[blank_row][blank_col][number]:
                    blank_fillers[blank_row][blank_col][number] = False
                    revert_list[i] = True
                else:
                    revert_list[i] = False
            solution_board = self.__get_solution(board, blanks, blank_fillers, are_blanks_filled)

            if solution_board:
                return solution_board

            # No solution found by having tested number in this cell
            # So we reallow neighbor cells to have this number filled in them
            for i, blank in enumerate(blanks):
                if revert_list[i]:
                    blank_row, blank_col = blank
                    blank_fillers[blank_row][blank_col][number] = True

        # If this point is reached, there is no solution with the initial board state,
        # a mistake must have been made in earlier steps

        # Declare chosen cell as empty once again
        are_blanks_filled[chosen_blank_index] = False
        board[row][col] = Sudoku._empty_cell_value

        return None


class Sudoku:
    _empty_cell_value = None

    def __init__(self, width, height=None, board=None, difficulty=-1, seed=randrange(sys.maxsize)):
        self.board = board
        self.width = width
        self.height = height
        if not height:
            self.height = width
        self.size = self.width * self.height
        self.__difficulty = difficulty

        assert self.width > 0, 'Width cannot be less than 1'
        assert self.height > 0, 'Height cannot be less than 1'
        assert self.size > 1, 'Board size cannot be 1 x 1'

        if board:
            blank_count = 0
            for row in self.board:
                for i in range(len(row)):
                    if type(row[i]) is not int or not 1 <= row[i] <= self.size:
                        row[i] = Sudoku._empty_cell_value
                        blank_count += 1
            if difficulty == -1:
                self.__difficulty = blank_count / self.size / self.size
        else:
            positions = list(range(self.size))
            random_seed(seed)
            shuffle(positions)
            self.board = [[(i + 1) if i == positions[j] else Sudoku._empty_cell_value for i in range(self.size)] for j
                          in range(self.size)]

    def solve(self):
        return _SudokuSolver(self)._solve()

    def validate(self):
        row_numbers = [[False for _ in range(self.size)] for _ in range(self.size)]
        col_numbers = [[False for _ in range(self.size)] for _ in range(self.size)]
        box_numbers = [[False for _ in range(self.size)] for _ in range(self.size)]

        for row in range(self.size):
            for col in range(self.size):
                cell = self.board[row][col]
                box = (row // self.height) * self.height + (col // self.width)
                if cell == Sudoku._empty_cell_value:
                    continue
                if row_numbers[row][cell - 1]:
                    return False
                if col_numbers[col][cell - 1]:
                    return False
                if box_numbers[box][cell - 1]:
                    return False
                row_numbers[row][cell - 1] = True
                col_numbers[col][cell - 1] = True
                box_numbers[box][cell - 1] = True
        return True

    @staticmethod
    def _copy_board(board):
        return [[cell for cell in row] for row in board]

    @staticmethod
    def empty(width, height):
        size = width * height
        board = [[Sudoku._empty_cell_value] * size] * size
        return Sudoku(width, height, board, 0)

    def difficulty(self, difficulty):
        assert 0 < difficulty < 1, 'Difficulty must be between 0 and 1'
        indices = list(range(self.size * self.size))
        shuffle(indices)
        problem_board = self.solve().board
        for index in indices[:int(difficulty * self.size * self.size)]:
            row_index = index // self.size
            col_index = index % self.size
            problem_board[row_index][col_index] = Sudoku._empty_cell_value
        return Sudoku(self.width, self.height, problem_board, difficulty)

    def show(self):
        if self.__difficulty == -2:
            pass#print('Puzzle has no solution')
        if self.__difficulty == -1:
            pass#print('Invalid puzzle. Please solve the puzzle (puzzle.solve()), or set a difficulty (puzzle.difficulty())')
        if not self.board:
            pass#print('No solution')
        #print(self.__format_board_ascii())

    def show_full(self):
        pass
        #print(self.__str__())

    def __format_board_ascii(self):
        table = ''
        cell_length = len(str(self.size))
        format_int = '{0:0' + str(cell_length) + 'd}'
        for i, row in enumerate(self.board):
            if i == 0:
                table += ('+-' + '-' * (cell_length + 1) * self.width) * self.height + '+' + '\n'
            table += (('| ' + '{} ' * self.width) * self.height + '|').format(
                *[format_int.format(x) if x != Sudoku._empty_cell_value else ' ' * cell_length for x in row]) + '\n'
            if i == self.size - 1 or i % self.height == self.height - 1:
                table += ('+-' + '-' * (cell_length + 1) * self.width) * self.height + '+' + '\n'
        return table

    def __str__(self):
        if self.__difficulty == -2:
            difficulty_str = 'INVALID PUZZLE (GIVEN PUZZLE HAS NO SOLUTION)'
        elif self.__difficulty == -1:
            difficulty_str = 'INVALID PUZZLE'
        elif self.__difficulty == 0:
            difficulty_str = 'SOLVED'
        else:
            difficulty_str = '{:.2f}'.format(self.__difficulty)
        return '''
---------------------------
{}x{} ({}x{}) SUDOKU PUZZLE
Difficulty: {}
---------------------------
{}
        '''.format(self.size, self.size, self.width, self.height, difficulty_str, self.__format_board_ascii())

# class SudokuNet:
#     @staticmethod
#     def build(width, height, depth, classes):
#         # initialize the model
#         model = Sequential()
#         inputShape = (height, width, depth)
#         model.add(Conv2D(32, (5, 5), padding="same",
#                          input_shape=inputShape))
#         model.add(Activation("relu"))
#         model.add(MaxPooling2D(pool_size=(2, 2)))
#         # second set of CONV => RELU => POOL layers
#         model.add(Conv2D(32, (3, 3), padding="same"))
#         model.add(Activation("relu"))
#         model.add(MaxPooling2D(pool_size=(2, 2)))
#         # first set of FC => RELU layers
#         model.add(Flatten())
#         model.add(Dense(64))
#         model.add(Activation("relu"))
#         model.add(Dropout(0.5))
#         # second set of FC => RELU layers
#         model.add(Dense(64))
#         model.add(Activation("relu"))
#         model.add(Dropout(0.5))
#         # softmax classifier
#         model.add(Dense(classes))
#         model.add(Activation("softmax"))
#         # return the constructed network architecture
#         return model
# # Load image, grayscale, and adaptive threshold
#
# def model_creator():
#     INIT_LR = 1e-3
#     EPOCHS = 10
#     BS = 128
#     # grab the MNIST dataset
#     print("[INFO] accessing MNIST...")
#     ((trainData, trainLabels), (testData, testLabels)) = mnist.load_data()
#     # add a channel (i.e., grayscale) dimension to the digits
#     trainData = trainData.reshape((trainData.shape[0], 28, 28, 1))
#     testData = testData.reshape((testData.shape[0], 28, 28, 1))
#     # scale data to the range of [0, 1]
#     trainData = trainData.astype("float32") / 255.0
#     testData = testData.astype("float32") / 255.0
#     # convert the labels from integers to vectors
#     le = LabelBinarizer()
#     trainLabels = le.fit_transform(trainLabels)
#     testLabels = le.transform(testLabels)
#     print("[INFO] compiling model...")
#     opt = Adam(lr=INIT_LR)
#     model = SudokuNet.build(width=28, height=28, depth=1, classes=10)
#     model.compile(loss="categorical_crossentropy", optimizer=opt,
#                   metrics=["accuracy"])
#     # train the network
#     print("[INFO] training network...")
#     H = model.fit(
#         trainData, trainLabels,
#         validation_data=(testData, testLabels),
#         batch_size=BS,
#         epochs=EPOCHS,
#         verbose=1)
#     print("[INFO] evaluating network...")
#     predictions = model.predict(testData)
#     print(classification_report(
#         testLabels.argmax(axis=1),
#         predictions.argmax(axis=1),
#         target_names=[str(x) for x in le.classes_]))
#     # serialize the model to disk
#     print("[INFO] serializing digit model...")
#     #model.save('net1', save_format="h5")
#     return model

def find_puzzle(image):
    #'train/train_5.jpg'
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(gray, (15, 15), 3)
    thresh = cv2.adaptiveThreshold(blurred, 255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    thresh = cv2.bitwise_not(thresh)
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    puzzleCnt = None
    for c in cnts:
            # approximate the contour
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            # if our approximated contour has four points, then we can
            # assume we have found the outline of the puzzle
            if len(approx) == 4:
                puzzleCnt = approx
                break
    output = image.copy()
    cv2.drawContours(output, [puzzleCnt], -1, (0, 255, 0), 2)
    image_binary = np.zeros((image.shape[1],
                             image.shape[0], 1),
                            np.uint8)[0]
    #mask =cv2.drawContours(image, [puzzleCnt], -1, (255,255,255),1)
    # mask = cv2.drawContours(image_binary , [max([puzzleCnt], key=cv2.contourArea)],
    #                 -1, (255, 255, 255), -1)
    output = thresh.copy()
    output[:] = (0)
    #cv2.drawContours(output, [puzzleCnt], -1, (255, 0, 0), 5)
    #cv2.fillPoly(output, pts=[puzzleCnt], color=(255, 255, 255))
    mask = cv2.fillPoly(output, pts=[puzzleCnt], color=(255, 255, 255))
    #ret3, mask = cv2.threshold(output, 0, 255, cv2.THRESH_OTSU)
    resized = cv2.resize(mask, (600,500), interpolation=cv2.INTER_AREA)
    # cv2.imshow('sad',resized)
    # cv2.waitKey(0)
    puzzle = four_point_transform(image, puzzleCnt.reshape(4, 2))
    warped = four_point_transform(gray, puzzleCnt.reshape(4, 2))
    return (puzzle, warped,mask)

def extract_digit(cell, debug=False):
    # apply automatic thresholding to the cell and then clear any
    # connected borders that touch the border of the cell
    thresh = cv2.threshold(cell, 0, 255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    thresh = clear_border(thresh)
    # check to see if we are visualizing the cell thresholding step
    # if debug:
    #     cv2.imshow("Cell Thresh", thresh)
    #     cv2.waitKey(0)
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_TREE,
                             cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    # if no contours were found than this is an empty cell
    if len(cnts) == 0:
        return None
    # otherwise, find the largest contour in the cell and create a
    # mask for the contour
    c = max(cnts, key=cv2.contourArea)
    mask = np.zeros(thresh.shape, dtype="uint8")
    cv2.drawContours(mask, [c], -1, 255, -1)
    # compute the percentage of masked pixels relative to the total
    # area of the image
    (h, w) = thresh.shape
    percentFilled = cv2.countNonZero(mask) / float(w * h)
    # if less than 3% of the mask is filled then we are looking at
    # noise and can safely ignore the contour
    if percentFilled < 0.001:
        return None
    # apply the mask to the thresholded cell
    digit = cv2.bitwise_and(thresh, thresh, mask=mask)
    # check to see if we should visualize the masking step
    if debug:
        resize = cv2.resize(digit,(300,300),interpolation = cv2.INTER_AREA)
        # cv2.imshow("Digit", resize)
        # cv2.waitKey(0)
    # return the digit to the calling function
    return digit

def predict_i(img):
    model = keras.models.load_model('net1.h5')#/autograder/submission/model.h5
    #print('jep')
    #img = cv2.imread("train/train_4.jpg")
    #img = imutils.resize(img, width=600)
    (puzzleImage, warped,mask) = find_puzzle(img)
    board = np.zeros((9, 9), dtype="int")

    stepX = warped.shape[1] // 9
    stepY = warped.shape[0] // 9

    cellLocs = []

    for y in range(0, 9):
        row = []
        for x in range(0, 9):
            startX = x * stepX
            startY = y * stepY
            endX = (x + 1) * stepX
            endY = (y + 1) * stepY
            # add the (x, y)-coordinates to our cell locations list
            row.append((startX, startY, endX, endY))
            cell = warped[startY:endY, startX:endX]
            digit = extract_digit(cell, debug=True)
            # verify that the digit is not empty
            sought = [255, 255, 255]

            # Find all pixels where the 3 NoTE ** BGR not RGB  values match "sought", and count

            if digit is not None:
                # resize the cell to 28x28 pixels and then prepare the
                # cell for classification
                n_white_pix = np.sum(digit == 255)
                if n_white_pix < 60:
                    board[y,x] = -1
                else:
                    roi = cv2.resize(digit, (28, 28))
                    roi = roi.astype("float") / 255.0
                    roi = img_to_array(roi)
                    roi = np.expand_dims(roi, axis=0)
                    percentFilled = cv2.countNonZero(digit) / float(digit.shape[0] * digit.shape[1])
                    if percentFilled < 0.03:
                        board[y,x] = -1
                    else:
                        #return model.predict(image).argmax()
                        pred = model.predict(roi).argmax(axis=1)[0]
                        board[y, x] = pred
                    # classify the digit and update the Sudoku board with the
                    # prediction
                    pred = model.predict(roi).argmax(axis=1)[0]
                    #print(pred)

        #print(board)
        cellLocs.append(row)
        #print("[INFO] OCR'd Sudoku board:")
        #print(board)

    puzzle = Sudoku(3, 3, board=board.tolist())
    puzzle.show()
    # solve the Sudoku puzzle
    #print("[INFO] solving Sudoku puzzle...")
    solution = puzzle.solve()
    solution.show_full()
    for (cellRow, boardRow) in zip(cellLocs, solution.board):
        # loop over individual cell in the row
        for (box, digit) in zip(cellRow, boardRow):
            # unpack the cell coordinates
            startX, startY, endX, endY = box
            # compute the coordinates of where the digit will be drawn
            # on the output puzzle image
            textX = int((endX - startX) * 0.33)
            textY = int((endY - startY) * -0.2)
            textX += startX
            textY += endY
            # draw the result digit on the Sudoku puzzle image
            cv2.putText(puzzleImage, str(digit), (textX, textY),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        # show the output image

    resized = cv2.resize(puzzleImage, (600,600), interpolation = cv2.INTER_AREA)
    # cv2.imshow("Sudoku Result", resized)
    # cv2.waitKey(0)
    for y in range(0, 9):
        for x in range(0, 9):
            if board[y,x]== 0:
                board[y,x] = -1


    return (mask, board)

def predict_image(image):
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
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
    #mask = np.bool_(np.ones_like(image))
    mask , sudoku_digitss = predict_i(image)
    #sudoku_digits = [np.int16(sudoku_digitss)]
    return mask,sudoku_digits
image = cv2.imread('train/train_3.jpg')
mask , digits = predict_image(image)
print(digits)
# gray = cv2.cvtColor(puzzle, cv2.COLOR_BGR2GRAY)
# thresh = cv2.threshold(gray, 0, 255,
# 		cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
# thresh = clear_border(thresh)
# # Filter out all numbers and noise to isolate only boxes
# cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
# 		cv2.CHAIN_APPROX_SIMPLE)
# 	cnts = imutils.grab_contours(cnts)
# for c in cnts:
#     area = cv2.contourArea(c)
#     if area < 1000:
#         cv2.drawContours(thresh, [c], -1, (0,0,0), -1)
#
# # Fix horizontal and vertical lines
# vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,5))
# thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, vertical_kernel, iterations=9)
# horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,1))
# thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, horizontal_kernel, iterations=4)
#
# # Sort by top to bottom and each row by left to right
# invert = 255 - thresh
# cnts = cv2.findContours(invert, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# cnts = cnts[0] if len(cnts) == 2 else cnts[1]
# (cnts, _) = contours.sort_contours(cnts, method="top-to-bottom")
#
#
#
# sudoku_rows = []
# row = []
# for (i, c) in enumerate(cnts, 1):
#     area = cv2.contourArea(c)
#     if area < 50000:
#         row.append(c)
#         if i % 9 == 0:
#             (cnts, _) = contours.sort_contours(row, method="left-to-right")
#             sudoku_rows.append(cnts)
#             row = []
#
# # Iterate through each box
# for row in sudoku_rows:
#     for c in row:
#         mask = np.zeros(image.shape, dtype=np.uint8)
#         cv2.drawContours(mask, [c], -1, (255,255,255), -1)
#         result = cv2.bitwise_and(image, mask)
#         result[mask==0] = 255
#         dim = (600, 600)
#
#         # resize image
#         resized = cv2.resize(invert, dim, interpolation=cv2.INTER_AREA)
#         cv2.imshow('thresh', resized)
#         resized = cv2.resize(result, dim, interpolation=cv2.INTER_AREA)
#         cv2.imshow('result', resized)
#         if cv2.waitKey(1000) & 0xff == ord('q'):
#             break
# dim = (600, 300)
#
#         # resize image
# resized = cv2.resize(thresh, dim, interpolation=cv2.INTER_AREA)
# cv2.imshow('thresh', resized)
# cv2.waitKey(0)
# resized = cv2.resize(invert, dim, interpolation=cv2.INTER_AREA)
# cv2.imshow('invert', resized)
# cv2.waitKey(0)