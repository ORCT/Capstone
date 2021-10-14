import cv2
import numpy as np

def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

def gaussian_blur(img, kernel_size):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

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

def canny(img, low_threshold, high_threshold):
    return cv2.Canny(img, low_threshold, high_threshold)

def draw_lines(img, lines, color=[0, 0, 255], thickness=2): # 선 그리기
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    #line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    #draw_lines(line_img, lines)
    return lines

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

def weighted_img(img, initial_img, α=1, β=1., λ=0.): # origin & hough overlap
    return cv2.addWeighted(initial_img, α, img, β, λ)

def nothing():
    pass

image = cv2.imread('lane sample/sample5.jpg') # import image, be careful the path!!
#cv2.imshow('origin',image)
#cv2.waitKey(0)

height, width = image.shape[:2]

gray_img = grayscale(image)
#cv2.imshow('gray',gray_img)
#cv2.waitKey(0)

blur_img = gaussian_blur(gray_img,3)

canny_img = canny(blur_img, 200, 350)
#cv2.imshow('canny',canny_img)
#cv2.waitKey(0)

vertices = np.array([[(0,height),(0, height/2), (width, height/2), (width,height)]], dtype=np.int32) # half of image size divided by center horizontal line
ROI_img = region_of_interest(canny_img, vertices) # ROI
#cv2.imshow('ROI',ROI_img)
#cv2.waitKey(0)

line_arr = hough_lines(ROI_img, 1, 1*np.pi/180, 30, 10, 20)
line_arr = np.squeeze(line_arr)
#cv2.imshow('hough', line_arr)
#cv2.waitKey(0)

slope_degree = (np.arctan2(line_arr[:,1] - line_arr[:,3], line_arr[:,0] - line_arr[:,2]) * 180) / np.pi
#print(slope_degree)

# restrict the horizon
line_arr = line_arr[np.abs(slope_degree)<160]
slope_degree = slope_degree[np.abs(slope_degree)<160]

# restrict the vertical
line_arr = line_arr[np.abs(slope_degree)>95]
slope_degree = slope_degree[np.abs(slope_degree)>95]

L_lines, R_lines = line_arr[(slope_degree>0),:], line_arr[(slope_degree<0),:]
temp = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
L_lines, R_lines = L_lines[:,None], R_lines[:,None]

left_fit_line = get_fitline(image,L_lines)
right_fit_line = get_fitline(image,R_lines)

draw_fit_line(temp, left_fit_line)
draw_fit_line(temp, right_fit_line)

result = weighted_img(temp, image)
#cv2.imshow('result',result)
cv2.imshow('ROI',ROI_img)
cv2.imshow('temp',temp)
cv2.waitKey(0)
cv2.destroyAllWindows()