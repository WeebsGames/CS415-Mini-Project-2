# code for hysteresis thresholding in Canny edge detector

import cv2
import numpy as np
import matplotlib.pyplot as plt


def convolution(im, kernel):
    kernel = np.flip(np.flip(kernel, axis=0), axis=1)
    im_height, im_width = im.shape
    kernel_size = kernel.shape[0]
    pad_size = int((kernel_size - 1) / 2)
    im_padded = np.zeros((im_height + pad_size * 2, im_width + pad_size * 2))
    im_padded[pad_size:-pad_size, pad_size:-pad_size] = im

    im_out = np.zeros_like(im)
    for x in range(im_width):
        for y in range(im_height):
            im_patch = im_padded[y:y + kernel_size, x:x + kernel_size]
            new_value = np.sum(kernel * im_patch)
            im_out[y, x] = new_value
    return im_out


def get_gaussian_kernel(kernel_size, sigma):
    kernel_x = np.linspace(-(kernel_size // 2), kernel_size // 2, kernel_size)
    for i in range(kernel_size):
        kernel_x[i] = np.exp(-(kernel_x[i] / sigma) ** 2 / 2)
    kernel = np.outer(kernel_x.T, kernel_x.T)

    kernel *= 1.0 / kernel.sum()
    return kernel


def compute_gradient(im):
    sobel_filter_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    sobel_filter_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    gradient_x = convolution(im, sobel_filter_x)
    gradient_y = convolution(im, sobel_filter_y)

    magnitude = np.sqrt(np.square(gradient_x) + np.square(gradient_y))
    magnitude *= 255.0 / magnitude.max()
    direction = np.arctan2(gradient_y, gradient_x)
    direction *= 180 / np.pi
    return magnitude, direction


def nms(magnitude, direction):
    height, width = magnitude.shape
    res = np.zeros(magnitude.shape)
    direction[direction < 0] += 180  # (-180, 180) -> (0, 180)

    for y in range(1, height - 1):
        for x in range(1, width - 1):
            current_direction = direction[y, x]
            current_magnitude = magnitude[y, x]
            if (0 <= current_direction < 22.5) or (157.5 <= current_direction <= 180):
                p = magnitude[y, x - 1]
                r = magnitude[y, x + 1]

            elif 22.5 <= current_direction < 67.5:
                p = magnitude[y + 1, x + 1]
                r = magnitude[y - 1, x - 1]

            elif 67.5 <= current_direction < 112.5:
                p = magnitude[y - 1, x]
                r = magnitude[y + 1, x]

            else:
                p = magnitude[y - 1, x + 1]
                r = magnitude[y + 1, x - 1]

            if current_magnitude >= p and current_magnitude >= r:
                res[y, x] = current_magnitude

    return res

def hysteresis(magnitude):
    height, width = magnitude.shape
    res = np.zeros(magnitude.shape)
    high = .05*255
    low = .024*255

    for y in range(1, height - 1):
        for x in range(1, width - 1):
            if(magnitude[x, y] > high):
                res[x,y] = 255
                res = checkneighbors(magnitude, res, x, y, high, low)
    return res

def checkneighbors(magnitude, res, x, y, high, low):
    for i in range(-1, 1):
        for j in range(-1, 1):
            if(magnitude[x+j, y+i] > low and magnitude[x+j, y+i] < high and i != 0 and j != 0 ):
                res[x+j, y+i] = 255
                res = checkneighbors(magnitude, res, x+j, y+i, high, low)
    return res


im = cv2.imread("images\lena-1.png", 0)
im = im.astype(float)

gaussian_kernel = get_gaussian_kernel(9, 3)
im_smoothed = convolution(im, gaussian_kernel)

# cv2.imshow("Original image", im.astype(np.uint8))
# cv2.imshow("Smoothed image", im_smoothed.astype(np.uint8))
# cv2.waitKey()
# cv2.destroyAllWindows()

gradient_magnitude, gradient_direction = compute_gradient(im_smoothed)

edge_nms = nms(gradient_magnitude, gradient_direction)

# plt.hist(edge_nms[edge_nms>0].flatten(), bins=50)
# plt.show()

# cv2.imshow("Before NMS", gradient_magnitude.astype(np.uint8))
cv2.imshow("After NMS", edge_nms.astype(np.uint8))
# cv2.imwrite("images\imgnms.png", edge_nms)
cv2.imshow("after hysteresis", hysteresis(edge_nms).astype(np.uint8))
cv2.waitKey()
cv2.destroyAllWindows()

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

def find_local_maxima(accumulator, threshold=30):
    height, width = accumulator.shape
    local_maxima = np.zeros_like(accumulator, dtype=bool)
    
    for i in range(1, height-1):
        for j in range(1, width-1):
            if accumulator[i, j] > threshold:
                # Check if it's a local maximum
                if accumulator[i, j] >= np.max(accumulator[i-1:i+2, j-1:j+2]):
                    local_maxima[i, j] = True
    
    return np.argwhere(local_maxima)


im = cv2.imread('images\paper.bmp')

im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

edge_map = cv2.Canny(im_gray, 70, 150)

accumulator, theta_values, rho_values = HoughTransform(edge_map)
# print(np.argwhere(accumulator))
# print(rho_values)
# lines = np.argwhere(accumulator > 30)
lines = find_local_maxima(accumulator, 30)

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