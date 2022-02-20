import cv2
import numpy as np

def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

def gaussian_blur(img, kernel_size):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def canny(img, low_threshold, high_threshold):
    return cv2.Canny(img, low_threshold, high_threshold)

def roi(img):
    mask = np.zeros_like(img)
    h,w = mask.shape
    vertices = np.array([[(w/10,h), (w/10,h*3/4), (w*9/10,h*3/4), (w*9/10,h)]], dtype=np.int32)
    cv2.fillPoly(mask, vertices, 255)
    roi_img = cv2.bitwise_and(img, mask)
    return roi_img

if __name__ == '__main__':
    origin_img = cv2.imread('./lane_detection_sample.jpg')
    gray_img = grayscale(origin_img)
    blur_img = gaussian_blur(gray_img, 5)
    canny_img = canny(blur_img, 50, 200)
    roi_img = roi(canny_img)

    cv2.imwrite('./roi_img.jpg',roi_img)
    cv2.imshow('roi',roi_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()