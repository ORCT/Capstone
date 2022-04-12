import cv2
import numpy as np

def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

def gaussian_blur(img, kernel_size):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def canny(img, low_threshold, high_threshold):
    return cv2.Canny(img, low_threshold, high_threshold)

def roi(img,h,w):
    mask = np.zeros_like(img)
    vertices = np.array([[(w/10,h), (w/10,h*3/4), (w*9/10,h*3/4), (w*9/10,h)]], dtype=np.int32)
    cv2.fillPoly(mask, vertices, 255)
    roi_img = cv2.bitwise_and(img, mask)
    return roi_img

def hough(img,h,w,min_line_len):
    lines = cv2.HoughLinesP(img, rho=1, theta=np.pi/180, threshold=30, minLineLength=min_line_len, maxLineGap=30)
    line_img = np.zeros((h, w, 3), dtype=np.uint8)
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(line_img, (x1, y1), (x2, y2), color=[255,0,0], thickness=2)
    return line_img

def lane_detection(min_line_len):
    origin_img = cv2.imread('./slope_test.jpg')
    h,w = origin_img.shape[:2]
    gray_img = grayscale(origin_img)
    blur_img = gaussian_blur(gray_img, 5)
    canny_img = canny(blur_img, 50, 200)
    roi_img = roi(canny_img,h,w)
    hough_img = hough(roi_img,h,w,min_line_len)
    return hough_img

def nothing(pos):
    pass

if __name__ == '__main__':
    cv2.namedWindow(winname='Lane Detection')
    cv2.createTrackbar('minLine', 'Lane Detection', 0, 200, nothing)#don't write keword
    while cv2.waitKey(1) != ord('q'):
        min_line_len = cv2.getTrackbarPos(trackbarname='minLine', winname='Lane Detection')
        hough_img = lane_detection(min_line_len)
        cv2.imshow('Lane Detection',hough_img)

    cv2.imwrite('./hough_img1.jpg',hough_img)
    cv2.destroyAllWindows()