import serial
import time
from collections import deque
import cv2
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
#start = time.time()  # start time

def nothing(x):
    pass

def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

def gaussian_blur(img, kernel_size=3):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def canny(img, low_threshold, high_threshold):
    return cv2.Canny(img, low_threshold, high_threshold)

def region_of_interest(img, vertices, color3=(255,255,255), color1=255): # ROI

    mask = np.zeros_like(img) # mask that same size of img
    
    if len(img.shape) > 2: # Color if 3 channel :
        color = color3
    else: # gray img if 1 channel :
        color = color1    
    # fill inner space of 4 point(vertices)(ROI) 
    cv2.fillPoly(mask, vertices, color)
    #cv2.imshow('mask',mask)

    # mask&origin img change to one img
    ROI_image = cv2.bitwise_and(img, mask)
    return ROI_image

def draw_lines(img, lines, color=[255, 0, 0], thickness=2): # draw line
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def hough_lines(img, rho=1, theta=np.pi/180, threshold=30, min_line_len=10, max_line_gap=20):
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)

    return lines, line_img

def weighted_img(img, initial_img, α=1, β=1., λ=0.): # origin & hough overlap
    return cv2.addWeighted(initial_img, α, img, β, λ)

def get_fitline(img, f_lines): #make representation line   
    lines = np.squeeze(f_lines)
    lines = lines.reshape(lines.shape[0]*2,2)
    rows,cols = img.shape[:2]
    output = cv2.fitLine(lines,cv2.DIST_L2,0, 0.01, 0.01)
    vx, vy, x, y = output[0], output[1], output[2], output[3]
    x1, y1 = int(((img.shape[0]-1)-y)/vy*vx + x) , img.shape[0]-1
    x2, y2 = int(((img.shape[0]/2+100)-y)/vy*vx + x) , int(img.shape[0]/2+100)
    
    result = [x1,y1,x2,y2]
    return result

def draw_fit_line(img, lines, color=[255, 0, 0], thickness=10): # draw representation line & color order is BGR
        cv2.line(img, (lines[0], lines[1]), (lines[2], lines[3]), color, thickness)

def get_vanishing_point(left_line, right_line):
    left_slope = (left_line[1]-left_line[3])/(left_line[0]-left_line[2])
    left_intercept = (left_line[1]-(left_line[1]-left_line[3])/(left_line[0]-left_line[2])*left_line[0])
    right_slope = (right_line[1]-right_line[3])/(right_line[0]-right_line[2])
    right_intercept = (right_line[1]-(right_line[1]-right_line[3])/(right_line[0]-right_line[2])*right_line[0])
    x = -(left_intercept-right_intercept)/(left_slope-right_slope)
    y = left_slope*x + left_intercept

    point = [x,y]
    return point
    #eqn_left_line = (left_line[1]-left_line[3])/(left_line[0]-left_line[2])*left_line[0]+(left_line[1]-(left_line[1]-(left_line[1]-left_line[3])/(left_line[0]-left_line[2])*left_line[0]))-left_line[1]
    #eqn_right_line = (right_line[1]-right_line[3])/(right_line[0]-right_line[2])*right_line[0]+(right_line[1]-(right_line[1]-(right_line[1]-right_line[3])/(right_line[0]-right_line[2])*right_line[0]))-right_line[1]

def draw_vp_circle(img,center_point):
    cv2.circle(img, center=tuple(np.int0(center_point)), radius=20, color=(255, 255, 0), thickness=3)

def get_steering_value(vp, height, width, steer_max=50, offset=0):#bigger height value is low position in the result picture that we saw
    vpx = (vp[0]-width/2)/(width/2) #degree control, -width/2 : -1 = width/2 = 1
    vpy = (vp[1]-height/2)/(height/2) #weight control, ,but exponent, -height/2 : -1 = height/2 = 1

    delta = 0.5*(np.sign(vpx)+1) * (vpx*steer_max) * (2**vpy) -0.5*(np.sign(vpx)-1) * (vpx*steer_max) * (2**vpy) + offset
    return delta

def interact_ser(_str, _ard):
    _ard.write(_str.encode())
    tmp = _ard.readline()
    print(tmp.decode())

def steering_process(delta_filter,delta,ard):
    delta_filter.append(delta)
    if len(delta_filter)>10:
        delta_filter.popleft()
    delta = int(sum(delta_filter)/len(delta_filter))
    if len(delta_filter)>=10:
        print('steer value', delta)
        serial_deque = deque([])
        if delta < 0:
            delta = abs(delta)
            str_delta = list(str(delta))
            serial_deque = deque(['-']+str_delta+['`'])
        elif delta == 0:
            str_delta = list(str(delta))
            serial_deque = deque(['f']+str_delta+['`'])
        else:
            str_delta = list(str(delta))
            serial_deque = deque(['+']+str_delta+['`'])
        for i in serial_deque:
            interact_ser(i,ard)

def apply_hsv_filter(frame):
    hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    lower_blue = np.array([0,0,0])
    upper_blue = np.array([360,100,20])
    mask = cv2.inRange(hsv,lower_blue,upper_blue)
    res = cv2.bitwise_and(frame,frame, mask=mask)
    return res

def conv_img_to_delta(image,low,high):
    height, width = image.shape[:2]# shape is numpy array

    gray_img = grayscale(image)
        
    blur_img = gaussian_blur(gray_img)

    canny_img = canny(blur_img,low,high)

    vertices = np.array([[(0,height),(0, height/2), (width, height/2), (width,height)]], dtype=np.int32) # half of image size divided by center horizontal line
    ROI_img = region_of_interest(canny_img, vertices) # ROI

    line_arr, line_img = hough_lines(ROI_img) # hough
    line_arr = np.squeeze(line_arr)
        
    # find slope
    slope_degree = (np.arctan2(line_arr[:,1] - line_arr[:,3], line_arr[:,0] - line_arr[:,2]) * 180) / np.pi

    # restrict the horizon
    line_arr = line_arr[np.abs(slope_degree)<160]
    slope_degree = slope_degree[np.abs(slope_degree)<160]

    # restrict the vertical
    line_arr = line_arr[np.abs(slope_degree)>95]
    slope_degree = slope_degree[np.abs(slope_degree)>95]

    # filtering line
    L_lines, R_lines = line_arr[(slope_degree>0),:], line_arr[(slope_degree<0),:]
    temp = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    L_lines, R_lines = L_lines[:,None], R_lines[:,None]

    # get points of rep lines
    left_fit_line = get_fitline(image,L_lines)
    right_fit_line = get_fitline(image,R_lines)

    # draw rep line
    draw_fit_line(temp, left_fit_line)
    draw_fit_line(temp, right_fit_line)

    # draw vp circle
    vp = get_vanishing_point(left_fit_line, right_fit_line)
    temp1 = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    draw_vp_circle(temp1,vp)

    #weight img
    temp = weighted_img(temp,temp1)
    result = weighted_img(image,temp)
    #result = weighted_img(image,hough_img)
    #result1 = weighted_img(temp,ROI_img)

    # get steering value(delta)
    row_delta = get_steering_value(vp,image.shape[0],image.shape[1])
    return result,ROI_img,int(row_delta)

def main():
    capture = cv2.VideoCapture(1)#If your pc has own camera already, additional webcam number will be 1, else 0.
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    port = 'COM15'
    ard = serial.Serial(port,9600)
    cv2.namedWindow('Lane Detection')
    cv2.createTrackbar('threshold1', 'Lane Detection', 0, 1000, nothing)
    cv2.createTrackbar('threshold2', 'Lane Detection', 0, 1000, nothing)
    cv2.setTrackbarPos('threshold1', 'Lane Detection', 300)
    cv2.setTrackbarPos('threshold2', 'Lane Detection', 500)
    delta_filter=deque([])
    while cv2.waitKey(33) != ord('q'):
        try:
            ret, frame = capture.read()
            low = cv2.getTrackbarPos('threshold1','Lane Detection')
            high = cv2.getTrackbarPos('threshold2','Lane Detection')
            res = apply_hsv_filter(frame)
            img,ROI_img,delta = conv_img_to_delta(res,low,high)
            steering_process(delta_filter,delta,ard)
            cv2.imshow("Lane Detection", img)
            cv2.imshow('ROI',ROI_img)
            #cv2.imshow('hsv',res)
            #cv2.imshow('origin',frame)
        except:
            continue
    capture.release()
    cv2.destroyAllWindows()
    ard.close()

main()