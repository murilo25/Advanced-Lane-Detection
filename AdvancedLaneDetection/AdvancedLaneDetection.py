import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
#%matplotlib qt
def magnitudeSobelFilter(img,thresh,kernel_size):
    # 1) Convert to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 2) Take the derivative in x or y given orient = 'x' or 'y'
    sobel_x = cv2.Sobel(gray_img,cv2.CV_64F,1,0,kernel_size)
    sobel_y = cv2.Sobel(gray_img,cv2.CV_64F,0,1,kernel_size)
    # 3) Compute magnitude of the derivative or gradient
    grad_mag = np.sqrt(sobel_x**2 + sobel_y**2)
    # 4) Rescale to 8 bit
    scale_factor = np.max(grad_mag)/255 
    scaled_mag = (grad_mag/scale_factor).astype(np.uint8) 
    # 5) Create a mask of 1's where the scaled gradient magnitude 
    sobel_binary = np.zeros_like(scaled_mag)
    sobel_binary[(scaled_mag >= thresh[0]) & (scaled_mag <= thresh[1])] = 1
    # 6) Return this mask as your binary_output image
    #plt.title("Sobel x")
    #plt.imshow(sobel_binary,cmap='gray')
    #plt.show()
    return sobel_binary

def absoluteSobelFilter(img,dir,thresh,kernel_size):
    # 1) Convert to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 2) Take the derivative in x or y given orient = 'x' or 'y'
    if (dir == 'x'):
        sobel = cv2.Sobel(gray_img,cv2.CV_64F,1,0,kernel_size)
    elif (dir == 'y'):
        sobel = cv2.Sobel(gray_img,cv2.CV_64F,0,1,kernel_size)
    # 3) Take the absolute value of the derivative or gradient
    abs_sobel = np.absolute(sobel)
    # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # 5) Create a mask of 1's where the scaled gradient magnitude 
    sobel_binary = np.zeros_like(scaled_sobel)
    sobel_binary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    # 6) Return this mask as your binary_output image
    #plt.title("Sobel x")
    #plt.imshow(sobel_binary,cmap='gray')
    #plt.show()
    return sobel_binary

def colorFilter(img,thresh):
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    h_channel = hls[:,:,0]
    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]

    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel>=thresh[0]) & (s_channel<=thresh[1])] = 1

    #plt.title("Filtered s")
    #plt.imshow(s_binary,cmap='gray')
    #plt.show()
    return s_binary

def detectLane(img):
    thresh_sobel = [20,200]
    sobel_kernel_size = 3
    thresh_color = [150,255]
    sobel_binary = absoluteSobelFilter(img,'x',thresh_sobel,sobel_kernel_size)
    color_binary = colorFilter(img,thresh_color)

    stacked_binary = np.dstack(( np.zeros_like(sobel_binary), sobel_binary, color_binary)) * 255

    combined_binary = np.zeros_like(sobel_binary)
    combined_binary[(color_binary == 1) | (sobel_binary == 1)] = 1

    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
    ax1.imshow(stacked_binary,cmap='gray')
    ax1.set_title('Overlay detection', fontsize=30)
    ax2.imshow(combined_binary,cmap='gray')
    ax2.set_title('Combined detection', fontsize=30)
    plt.show()
    #combined_binary = cv2.blur(combined_binary,(21,21))
    return combined_binary
    
def changePerspective(img):

    img_size = (img.shape[1],img.shape[0])  #x,y
    # define vertices of the perspective polygon (hard coded based on the picture available for evaluation
    top_left = [581,458]
    top_right = [703,458]
    bottom_left = [241,676]
    bottom_right = [1053,676]

    ### begin change of perspective ###
    # original coordinates
    src = np.float32([top_left,top_right,bottom_right,bottom_left])
    # destination coordinates
    dst = np.float32([[bottom_left[0],0],[bottom_right[0],0],bottom_right,bottom_left])
    # compute transform
    M = cv2.getPerspectiveTransform(src,dst)
    # use transform to change perspective to top view
    birdsEye = cv2.warpPerspective(img,M,img_size,flags=cv2.INTER_LINEAR)

    plt.title('Birds eye')
    plt.imshow(birdsEye,cmap='gray')
    plt.show()

    return birdsEye

def findPixels(img):
    ### find lane pixels coordinates using a sliding window approach
    ### each line (left/right) has its own window that slides from the bottom to the top of the img
    ### returns pixels coordinates
    
    img_size = (img.shape[1],img.shape[0])
    # compute histogram to roughly estimate the lines x coordinates
    histogram = []
    for i in range (0,img_size[0]):
        histogram.append(np.sum(img[:,i]))
    #hist2=np.sum(img,axis = 0) # same as for loop above
    plt.title("Histogram on x-axis")
    plt.plot(histogram)
    plt.show()

    # estimates x coordinate of lines
    left_window_center = np.argmax(histogram[:img_size[0]//2])
    right_window_center = np.argmax(histogram[img_size[0]//2:]) + img_size[0]//2 

    # define window parameters
    n_windows = 9
    window_height = np.int(img_size[1]//n_windows)
    window_width = 100
    min_pix = 50    # minimum number of pixels to recompute window_center

    # create an empty array to store white pixels coordinates
    left_line_coordinates = []
    right_line_coordinates = []

    # compute initial window coordinates
    window_left_x_low_current = left_window_center - window_width//2
    window_left_x_high_current = left_window_center + window_width//2
    window_right_x_low_current = right_window_center - window_width//2
    window_right_x_high_current = right_window_center + window_width//2

    # identify all white pixels in the img
    pixel_coordinates = img.nonzero() # returns a 2xN array that contains the coordinates of white pixels (N = # of white pixels in the image)
    pixel_x = np.array(pixel_coordinates[1])
    pixel_y = np.array(pixel_coordinates[0])

    img_out = np.dstack((img,img,img))

    # for each window, extract white pixels coordinates
    for i in range (0,n_windows):
        # compute i-th window coordinates
        window_left_x_low = window_left_x_low_current
        window_left_x_high = window_left_x_high_current
        window_right_x_low = window_right_x_low_current
        window_right_x_high = window_right_x_high_current
        window_bottom = img_size[1] - window_height*(i)
        window_top = img_size[1] - window_height*(i+1)

        # Identify the nonzero pixels in x and y within the window (<val>.nonzero()[0] ??)
        window_left_coordinates = ((pixel_y >= window_top) & (pixel_y < window_bottom) & 
        (pixel_x >= window_left_x_low) &  (pixel_x < window_left_x_high)).nonzero()[0]
        window_right_coordinates = ((pixel_y >= window_top) & (pixel_y < window_bottom) & 
        (pixel_x >= window_right_x_low) &  (pixel_x < window_right_x_high)).nonzero()[0]
        
        # Append these indices to the lists
        left_line_coordinates.append(window_left_coordinates)
        right_line_coordinates.append(window_right_coordinates)

        # If found > min_pix pixels, recenter next window on their mean position
        if len(window_left_coordinates) > min_pix:
            left_window_center = np.int(np.mean(pixel_x[window_left_coordinates]))
            window_left_x_low_current = left_window_center - window_width//2
            window_left_x_high_current = left_window_center + window_width//2
        if len(window_right_coordinates) > min_pix:   
            right_window_center = np.int(np.mean(pixel_x[window_right_coordinates]))
            window_right_x_low_current = right_window_center - window_width//2
            window_right_x_high_current = right_window_center + window_width//2

        # Draw the windows on the visualization image
        cv2.rectangle(img_out,(window_left_x_high,window_bottom),(window_left_x_low,window_top),(255,255,255), 10)
        cv2.rectangle(img_out,(window_right_x_high,window_bottom),(window_right_x_low,window_top),(255,255,255), 10) 
        #plt.imshow(img_out)
        #plt.show()

        #end for
    #plt.show()
    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_line_coordinates = np.concatenate(left_line_coordinates)
        right_line_coordinates = np.concatenate(right_line_coordinates)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass
    # Extract left and right line pixel positions
    left_x = pixel_x[left_line_coordinates]
    left_y = pixel_y[left_line_coordinates] 
    right_x = pixel_x[right_line_coordinates]
    right_y = pixel_y[right_line_coordinates]

    ## Visualization ##
    # Colors pixels detected within windows
    img_out[left_y, left_x] = [255, 0, 0]
    img_out[right_y, right_x] = [0, 0, 255]
    plt.imshow(img_out)
    plt.show()

    return left_x,left_y,right_x,right_y,img_out

def drawLines(img):
    left_x,left_y,right_x,right_y,img_out = findPixels(img)

    return 0

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
image = cv2.imread('test_images/test1.jpg')
image = cv2.imread('test_images/test2.jpg')
image = cv2.imread('test_images/test3.jpg')
image = cv2.imread('test_images/test4.jpg') # good but can improve
image = cv2.imread('test_images/test5.jpg') # bad
image = cv2.imread('test_images/test6.jpg')
undistorted_image = cv2.undistort(image, mtx, dist, None, mtx)
birdsEye = changePerspective(undistorted_image) # apply region of interest mask and change perspective
binary_birdsEye = detectLane(birdsEye) # detect all edges using sobel x absolute & color
lines_parameters = drawLines(binary_birdsEye)

