import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
#%matplotlib qt

def absoluteSobelFilter(img,dir,thresh,kernel_size):
    # 1) Convert to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 2) Take the derivative in x or y given orient = 'x' or 'y'
    if (dir == 'x'):
        sobel = cv2.Sobel(gray_img,cv2.CV_64F,1,0,kernel_size)
    elif (dir == 'y'):
        sobel = cv2.Sobel(gray_img,cv2.CV_64F,1,0,kernel_size)
    # 3) Take the absolute value of the derivative or gradient
    abs_sobel = np.absolute(sobel)
    # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # 5) Create a mask of 1's where the scaled gradient magnitude 
    ret,binary_output = cv2.threshold(scaled_sobel,thresh[0],thresh[1],cv2.THRESH_BINARY)
    # 6) Return this mask as your binary_output image
    plt.title("Sobel x")
    plt.imshow(binary_output,cmap='gray')
    plt.show()
    return binary_output

def colorFilter(img,thresh):
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    h_channel = hls[:,:,0]
    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]

    #s_binary = cv2.threshold(s_channel,thresh[0],thresh[1],cv2.THRESH_BINARY) # gives error when ploting
    s_binary = (s_channel>=thresh[0]) & (s_channel <= thresh[1])
    plt.title("Filtered s")
    plt.imshow(s_binary,cmap='gray')
    plt.show()
    return s_binary

def detectLines(img):
    thresh_sobel = [20,100]
    sobel_kernel_size = 3
    thresh_color = [200,255]
    sobel_binary = absoluteSobelFilter(img,'x',thresh_sobel,sobel_kernel_size)
    color_binary = colorFilter(img,thresh_color)

    output_binary = sobel_binary | color_binary
    color_binary = np.dstack(( np.zeros_like(sobel_binary), sobel_binary, color_binary)) * 255

    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
    ax1.imshow(color_binary,cmap='gray')
    ax1.set_title('Overlay Image', fontsize=30)
    ax2.imshow(output_binary,cmap='gray')
    ax2.set_title('Combined Image', fontsize=30)
    plt.show()


    return output_binary
    
    

# set grid size internal to the chess board
nCol = 9
nRow = 6
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((nRow*nCol,3), np.float32)
objp[:,:2] = np.mgrid[0:nCol, 0:nRow].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d points in real world space
imgpoints = [] # 2d points in image plane.

# Make a list of calibration images
images = glob.glob('camera_cal/calibration*.jpg')

print("Calibrating...\n")

# Step through the list and search for chessboard corners
for idx, fname in enumerate(images):
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (nCol,nRow), None)

    # If found, add object points, image points
    if ret == True:
        objpoints.append(objp)
        imgpoints.append(corners)

        # Draw and display the corners
        #cv2.drawChessboardCorners(img, (nCol,nRow), corners, ret)
        #cv2.imshow('img', img)
        #cv2.waitKey(500)

cv2.destroyAllWindows()
# Do camera calibration given object points and image points (mtx -> camera matrix , dist -> distortion coefficients)
img_size = (img.shape[1], img.shape[0])
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)

print("Calibration done!\n")

###  Test calibration ###
# Test undistortion on an image
img = cv2.imread('camera_cal/calibration2.jpg')
img_size = (img.shape[1], img.shape[0])

dst = cv2.undistort(img, mtx, dist, None, mtx)
#cv2.imwrite('camera_cal/calibration2.jpg',dst)

# Visualize undistortion
#f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
#ax1.imshow(img)
#ax1.set_title('Original Image', fontsize=30)
#ax2.imshow(dst)
#ax2.set_title('Undistorted Image', fontsize=30)
#plt.show()

### end calibration test ###

### lane detection ###
image = cv2.imread('test_images/straight_lines1.jpg')
image = cv2.imread('test_images/straight_lines2.jpg')
#image = cv2.imread('test_images/test1.jpg') # little bad with sobel only
#image = cv2.imread('test_images/test2.jpg') # ok not good not bad
#image = cv2.imread('test_images/test3.jpg')
#image = cv2.imread('test_images/test4.jpg') # bad
#image = cv2.imread('test_images/test5.jpg') # bad
#image = cv2.imread('test_images/test6.jpg') # ok not good not bad
undistorted_image = cv2.undistort(image, mtx, dist, None, mtx)
binary_image = detectLines(undistorted_image)
