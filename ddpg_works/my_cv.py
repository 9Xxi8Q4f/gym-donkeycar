import cv2
import numpy as np
import skimage as skimage
import cv2

img_rows , img_cols = 80, 80
# Convert image into Black and white
# img_channels = 4 # We stack 4 frames

def remove_noise(image, kernel_size):
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)


def discard_colors(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def detect_edges(image, low_threshold, high_threshold):
    return cv2.Canny(image, low_threshold, high_threshold)


def draw_lines(image, lines, color=[255, 0, 0], thickness=2):
    for line in lines:
        for x1,y1,x2,y2,slope in line:
            x1, y1, x2, y2 = int(x1),int(y1),int(x2),int(y2)
            cv2.line(image, (x1, y1), (x2, y2), color, thickness)


def hough_lines(image, rho, theta, threshold, min_line_len, max_line_gap):
    lines = cv2.HoughLinesP(image, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    return lines


def slope(x1, y1, x2, y2):
    try:
        return (y1 - y2) / (x1 - x2)
    except:
        return 0
        

def separate_lines(lines):
    right = []
    left = []

    if lines is not None:
        for x1,y1,x2,y2 in lines[:, 0]:
            m = slope(x1,y1,x2,y2)
            if m >= 0:
                right.append([x1,y1,x2,y2,m])
            else:
                left.append([x1,y1,x2,y2,m])
    return left, right


def reject_outliers(data, cutoff, threshold=0.08, lane='left'):
    data = np.array(data)
    data = data[(data[:, 4] >= cutoff[0]) & (data[:, 4] <= cutoff[1])]
    try:
        if lane == 'left':
            return data[np.argmin(data,axis=0)[-1]]
        elif lane == 'right':
            return data[np.argmax(data,axis=0)[-1]]
    except:
        return []


def extend_point(x1, y1, x2, y2, length):
    line_len = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    x = x2 + (x2 - x1) / line_len * length
    y = y2 + (y2 - y1) / line_len * length
    return x, y


def process_image(obs):

            obs = cv2.cvtColor(obs, cv2.COLOR_BGR2GRAY)
            obs = cv2.resize(obs, (img_rows, img_cols))
            edges = detect_edges(obs, low_threshold=150, high_threshold=200)
 
            rho = 0.8
            theta = np.pi/180
            threshold = 25
            min_line_len = 5
            max_line_gap = 10

            hough_line = hough_lines(edges, rho, theta, threshold, min_line_len, max_line_gap)

            left_lines, right_lines = separate_lines(hough_line)

            filtered_right, filtered_left = [],[]
            if len(left_lines):
                filtered_left = reject_outliers(left_lines, cutoff=(-30.0, -0.1), lane='left')
            if len(right_lines):
                filtered_right = reject_outliers(right_lines,  cutoff=(0.1, 30.0), lane='right')

            lines = []
            if len(filtered_left) and len(filtered_right):
                lines = np.expand_dims(np.vstack((np.array(filtered_left),np.array(filtered_right))),axis=0).tolist()
            elif len(filtered_left):
                lines = np.expand_dims(np.expand_dims(np.array(filtered_left),axis=0),axis=0).tolist()
            elif len(filtered_right):
                lines = np.expand_dims(np.expand_dims(np.array(filtered_right),axis=0),axis=0).tolist()

            ret_img = np.zeros((80,80))

            if len(lines):
                try:
                    draw_lines(ret_img, lines, thickness=1)
                except:
                    pass

            ret_img = ret_img[30:,:]
            return ret_img.flatten()


# img = cv2.imread("rr.png")
# imag = process_image(img)

# import matplotlib.pyplot as plt

# plt.imshow(imag, cmap="gray")
# plt.show()
