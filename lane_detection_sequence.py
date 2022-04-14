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
    vertices = np.array([[(0,h), (0,h*2/3), (w,h*2/3), (w,h)]], dtype=np.int32)
    cv2.fillPoly(mask, vertices, 255)
    roi_img = cv2.bitwise_and(img, mask)
    return roi_img

def restrict_deg(lines,min_slope,max_slope):
    slope_deg = np.rad2deg(np.arctan2(lines[:,1]-lines[:,3],lines[:,0]-lines[:,2]))
    lines = lines[np.abs(slope_deg)<max_slope]#cannot use and & index true catch
    slope_deg = slope_deg[np.abs(slope_deg)<max_slope]
    lines = lines[np.abs(slope_deg)>min_slope]
    slope_deg = slope_deg[np.abs(slope_deg)>min_slope]#where can i use slope
    return lines, slope_deg

def separate_line(lines,slope_deg):
    l_lines, r_lines = lines[(slope_deg>0),:], lines[(slope_deg<0),:]
    l_line = [sum(l_lines[:,0])/len(l_lines),sum(l_lines[:,1])/len(l_lines),sum(l_lines[:,2])/len(l_lines),sum(l_lines[:,3])/len(l_lines)]
    r_line = [sum(r_lines[:,0])/len(r_lines),sum(r_lines[:,1])/len(r_lines),sum(r_lines[:,2])/len(r_lines),sum(r_lines[:,3])/len(r_lines)]
    return l_line, r_line

def hough(img,h,w,min_line_len,min_slope,max_slope):
    lines = cv2.HoughLinesP(img, rho=1, theta=np.pi/180, threshold=30, minLineLength=min_line_len, maxLineGap=30)#return = [[x1,y1,x2,y2],[...],...]
    lines = np.squeeze(lines)#one time ok
    lanes, slopes = restrict_deg(lines,min_slope,max_slope)
    l_lane, r_lane = separate_line(lanes,slopes)
    #lane_img = np.zeros((h, w, 3), dtype=np.uint8)
    #for x1,y1,x2,y2 in l_lanes:
    #cv2.line(lane_img, (int(l_lane[0]), int(l_lane[1])), (int(l_lane[2]), int(l_lane[3])), color=[0,0,255], thickness=2)
    #for x1,y1,x2,y2 in r_lanes:
    #cv2.line(lane_img, (int(r_lane[0]), int(r_lane[1])), (int(r_lane[2]), int(r_lane[3])), color=[255,0,0], thickness=2)
    return l_lane,r_lane



def lane_detection(min_line_len,min_slope,max_slope):
    origin_img = cv2.imread('./left_right.jpg')
    h,w = origin_img.shape[:2]
    gray_img = grayscale(origin_img)
    blur_img = gaussian_blur(gray_img, 5)
    canny_img = canny(blur_img, 50, 200)
    roi_img = roi(canny_img,h,w)
    l_lane,r_lane = hough(roi_img,h,w,min_line_len,min_slope,max_slope)
    cv2.line(origin_img, (int(l_lane[0]), int(l_lane[1])), (int(l_lane[2]), int(l_lane[3])), color=[0,0,255], thickness=5)
    cv2.line(origin_img, (int(r_lane[0]), int(r_lane[1])), (int(r_lane[2]), int(r_lane[3])), color=[255,0,0], thickness=5)
    return origin_img

def nothing(pos):
    pass

if __name__ == '__main__':
    cv2.namedWindow(winname='Lane Detection')
    cv2.createTrackbar('houghMinLine', 'Lane Detection', 20, 200, nothing)#don't write keyword
    cv2.createTrackbar('slopeMinDeg', 'Lane Detection', 100, 180, nothing)
    cv2.createTrackbar('slopeMaxDeg', 'Lane Detection', 160, 180, nothing)
    while cv2.waitKey(1) != ord('q'):
        min_line_len = cv2.getTrackbarPos(trackbarname='houghMinLine', winname='Lane Detection')
        min_slope = cv2.getTrackbarPos('slopeMinDeg','Lane Detection')
        max_slope = cv2.getTrackbarPos('slopeMaxDeg','Lane Detection')
        result_img = lane_detection(min_line_len,min_slope,max_slope)
        cv2.imshow('Lane Detection',result_img)

    cv2.imwrite('./hough_img3.jpg',result_img)
    cv2.destroyAllWindows()
    
#It will be great that we can select the instant roi region using click when we run the code.