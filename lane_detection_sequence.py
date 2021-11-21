import cv2
import numpy as np

def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

def gaussian_blur(img, kernel_size):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def canny(img, low_threshold, high_threshold):
    return cv2.Canny(img, low_threshold, high_threshold)

origin_img = cv2.imread('./lane_detection_sample.jpg')
gray_img = grayscale(origin_img)
blur_img = gaussian_blur(gray_img, 5)
canny_img = canny(blur_img, 50, 200)

cv2.imwrite('./canny_img.jpg',canny_img)
cv2.imshow('canny',canny_img)
cv2.waitKey(0)
cv2.destroyAllWindows()