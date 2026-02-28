import cv2
import numpy as np
import math

def HoughTransform(edge_map):
    theta_values = np.deg2rad(np.arange(-90.0, 90.0))
    height, width = edge_map.shape
    diagonal_length = int(round(math.sqrt(width * width + height * height)))
    rho_values = np.linspace(-diagonal_length, diagonal_length, diagonal_length * 2 + 1)

    accumulator = np.zeros((len(rho_values), len(theta_values)), dtype=int)
    y_coordinates, x_coordinates = np.nonzero(edge_map)

    for edge_idx in range(len(x_coordinates)):
        x = x_coordinates[edge_idx]
        y = y_coordinates[edge_idx]
        for theta_idx in range(len(theta_values)):
            theta = theta_values[theta_idx]
            rho = int(round(x * np.cos(theta) + y * np.sin(theta)))
            accumulator[rho + diagonal_length, theta_idx] += 1

        # print("%d out of %d edges have voted" % (edge_idx+1, len(x_coordinates)))
        # cv2.imshow("Accumulator", (accumulator * 255 / accumulator.max()).astype(np.uint8))
        # cv2.waitKey(0)
    return accumulator, theta_values, rho_values


im = cv2.imread('images\paper.bmp')

im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

edge_map = cv2.Canny(im_gray, 70, 150)

accumulator, theta_values, rho_values = HoughTransform(edge_map)

lines = np.argwhere(accumulator > 30)

height, width = im_gray.shape
for line in lines:
    rho = rho_values[line[0]]
    theta = theta_values[line[1]]
    slope = -np.cos(theta)/np.sin(theta)
    intercept = rho/np.sin(theta)
    x1, x2 = 0, width
    y1 = int(slope*x1 + intercept)
    y2 = int(slope*x2 + intercept)
    cv2.line(im, (x1, y1), (x2, y2), (0, 0, 255), 2)
    # cv2.imshow("Output", im)
    # cv2.waitKey(0)

cv2.imshow("Edges", edge_map)
cv2.imshow("Hough Transform", (accumulator*255/accumulator.max()).astype(np.uint8))
cv2.imshow("Output", im)
cv2.waitKey(0)
cv2.destroyAllWindows()