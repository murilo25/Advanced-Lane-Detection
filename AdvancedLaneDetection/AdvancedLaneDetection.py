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
    thresh_color = [100,255]    ##150,255
    sobel_binary = absoluteSobelFilter(img,'x',thresh_sobel,sobel_kernel_size)
    color_binary = colorFilter(img,thresh_color)

    stacked_binary = np.dstack(( np.zeros_like(sobel_binary), sobel_binary, color_binary)) * 255

    combined_binary = np.zeros_like(sobel_binary)
    combined_binary[(color_binary == 1) | (sobel_binary == 1)] = 1

    
    #f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
    #ax1.imshow(stacked_binary,cmap='gray')
    #ax1.set_title('Overlay detection', fontsize=30)
    #ax2.imshow(combined_binary,cmap='gray')
    #ax2.set_title('Combined detection', fontsize=30)
    #plt.show()

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

    #plt.title('Birds eye')
    #plt.imshow(birdsEye,cmap='gray')
    #plt.show()

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
    ## display histogram
    #plt.title("Histogram on x-axis")
    #plt.plot(histogram)
    #plt.show()

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
    #plt.imshow(img_out)
    #plt.show()

    return left_x,left_y,right_x,right_y,img_out

def fitPolynomialLines(left_x,left_y,right_x,right_y,img_out):
    # Fit a second order polynomial to each using `np.polyfit`
    left_fit = np.polyfit(left_y, left_x, 2)
    right_fit = np.polyfit(right_y, right_x, 2)
    return left_fit,right_fit

def computeLines(img):
    left_x,left_y,right_x,right_y,img_out = findPixels(img)
    left_fit,right_fit = fitPolynomialLines(left_x,left_y,right_x,right_y,img_out)

    return left_fit,right_fit,img_out

def drawLines(left_fit,right_fit,img_out):
    # Generate x and y values for plotting
    ploty = np.linspace(0, img_out.shape[0]-1, img_out.shape[0] )
    try:
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        left_fitx = 1*ploty**2 + 1*ploty
        right_fitx = 1*ploty**2 + 1*ploty

    # Plots the left and right polynomials on the lane lines
    #plt.plot(left_fitx, ploty, color='yellow')
    #plt.plot(right_fitx, ploty, color='yellow')
    #plt.show()
    return left_fitx,right_fitx

def computeCurvateRadius(left_poly,right_poly,img_out):

    #print(left_poly)
    #print(right_poly)
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/676 # meters per pixel in y dimension (estimated length by looking at image/delta Y defined by the region of interest vertices)
    xm_per_pix = 3.7/812 # meters per pixel in x dimension (lane width in meters / delta X in pixels defined by the region of interest
   
    # rescale polynomial coefficients to convert from pixels to meters
    A_left = xm_per_pix/(ym_per_pix**2) * left_poly[0]
    B_left = xm_per_pix/ym_per_pix * left_poly[1]
    C_left = left_poly[2]
    A_right = xm_per_pix/(ym_per_pix**2) * right_poly[0]
    B_right = xm_per_pix/ym_per_pix * right_poly[1]
    C_right = right_poly[2]

    y = np.linspace(0, img_out.shape[0]-1, img_out.shape[0] )
    left = []
    right = []
    #for i in range(len(y)):
    #    left.append( ( ( 1+(2*A_left*y[i]+B_left)**2 )**(3/2) ) / ( abs(2*A_left) ) )
    #    right.append( ( ( 1+(2*A_right*y[i]+B_right)**2 )**(3/2) ) / ( abs(2*A_right) ) )
    #r_left = np.mean(left)
    #r_right = np.mean(right)
    y = np.max(y)*ym_per_pix
    r_left = ( ( 1+(2*A_left*y+B_left)**2 )**(3/2) ) / ( abs(2*A_left) ) 
    r_right = ( ( 1+(2*A_right*y+B_right)**2 )**(3/2) ) / ( abs(2*A_right) ) 
    
    #print(r_left)
    #print(r_right)
    return (r_left+r_right)/2

def highlightLane(left_line_params,right_line_params,img):
    # Find the region inside lane lines
    XX, YY = np.meshgrid(np.arange(0, img.shape[1]), np.arange(0, img.shape[0]))

    left_line = left_line_params[0]*YY**2 + left_line_params[1]*YY + left_line_params[2]
    right_line = right_line_params[0]*YY**2 + right_line_params[1]*YY + right_line_params[2]
 
    region = ( (XX > left_line) & (XX <right_line) )

    #new_img = np.zeros_like(img)
    #new_img[region] = 255
    #new_img = np.dstack(( np.zeros_like(img), img, img)) * 0
    #new_img = cv2.imread('test_images/test3.jpg')
    img[region] = [0,255,0]
    img[~region] = [0,0,0]
    #plt.imshow(img)
    #plt.show()
    return img

def changePerspectiveBack(img,original):

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
    M = cv2.getPerspectiveTransform(dst,src)
    # use transform to change perspective to top view
    normal_view = cv2.warpPerspective(img,M,img_size,flags=cv2.INTER_LINEAR)
    #normal_view = cv2.bitwise_or(normal_view,original)
    normal_view= cv2.addWeighted(normal_view, 0.3, original, 1, 0)

    #plt.title('Normal view')
    #plt.imshow(normal_view)
    #plt.show()

    return normal_view

def addText2Img(img,radius,org,txt):
    # font 
    font = cv2.FONT_HERSHEY_SIMPLEX 
    # org 
    #org = (50, 50) 
    # fontScale 
    fontScale = 1
    # Blue color in BGR 
    color = (255, 255, 255) 
    # Line thickness of 2 px 
    thickness = 2
    # Using cv2.putText() method 
    image = cv2.putText(img,txt,org, font,fontScale,color,thickness,cv2.LINE_AA) 
    return image

def computeLaneOffset(left_line,right_line,img):

    xm_per_pix = 3.7/812 # meters per pixel in x dimension (lane width in meters / delta X in pixels defined by the region of interest

    y = img.shape[0]
    x_left = left_line[0]*y**2 + left_line[1]*y + left_line[2]
    x_right = right_line[0]*y**2 + right_line[1]*y + right_line[2]


    offset_pix = abs( ((img.shape[1])/2) - ((x_right-x_left)/2) ) 
    offset = xm_per_pix * offset_pix

    return offset

def computeLinesFaster(img,left_fit,right_fit):
    margin = 100
    ploty = np.linspace(0, img.shape[0]-1, img.shape[0] )
    # Grab detected pixels coordinates
    nonzero = img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Set the area of search based on activated x-values within the +/- margin of our poly
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + 
                    left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + 
                    left_fit[1]*nonzeroy + left_fit[2] + margin)))
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + 
                    right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + 
                    right_fit[1]*nonzeroy + right_fit[2] + margin)))
    
    # Again, extract left and right line pixel positions
    left_x = nonzerox[left_lane_inds]
    left_y = nonzeroy[left_lane_inds] 
    right_x = nonzerox[right_lane_inds]
    right_y = nonzeroy[right_lane_inds]

    left_fit_new,right_fit_new = fitPolynomialLines(left_x,left_y,right_x,right_y,img)

    if ( (abs((left_fit_new-left_fit)/left_fit).any() > 0.1) | ( abs((right_fit_new-right_fit)/right_fit).any() > 0.1) ):
        bad_fit = True
        print(bad_fit)
    else:
        bad_fit = False
        left_fit = left_fit_new
        right_fit = right_fit_new
    ## Visualization ##
    img_out = np.dstack((img, img, img))*255
    # Colors pixels detected within windows
    img_out[left_y, left_x] = [255, 0, 0]
    img_out[right_y, right_x] = [0, 0, 255]
    #plt.imshow(img_out)
    #plt.show()

    left_line = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_line = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    # Plot the polynomial lines onto the image
    #plt.plot(left_line, ploty, color='yellow')
    #plt.plot(right_line, ploty, color='yellow')
    ## End visualization steps ##

    return left_fit,right_fit,img_out,bad_fit

### block comment calibration to run faster

## set grid size internal to the chess board
#nCol = 9
#nRow = 6
## prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
#objp = np.zeros((nRow*nCol,3), np.float32)
#objp[:,:2] = np.mgrid[0:nCol, 0:nRow].T.reshape(-1,2)

## Arrays to store object points and image points from all the images.
#objpoints = [] # 3d points in real world space
#imgpoints = [] # 2d points in image plane.

## Make a list of calibration images
#images = glob.glob('camera_cal/calibration*.jpg')

#print("Calibrating...\n")

## Step through the list and search for chessboard corners
#for idx, fname in enumerate(images):
#    img = cv2.imread(fname)
#    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#    # Find the chessboard corners
#    ret, corners = cv2.findChessboardCorners(gray, (nCol,nRow), None)

#    # If found, add object points, image points
#    if ret == True:
#        objpoints.append(objp)
#        imgpoints.append(corners)

#        # Draw and display the corners
#        #cv2.drawChessboardCorners(img, (nCol,nRow), corners, ret)
#        #cv2.imshow('img', img)
#        #cv2.waitKey(500)

#cv2.destroyAllWindows()
# # Do camera calibration given object points and image points (mtx -> camera matrix , dist -> distortion coefficients)
#img_size = (img.shape[1], img.shape[0])
#ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)

#print("Calibration done!\n")

### end block comment

mtx = np.array([[1143.3010,0,657.8923],[0,1139.0312,407.8527],[0,0,1]])
dist = np.array([-0.2255,-0.2080,-0.001387,0.0006729,0.3713])

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
#image = cv2.imread('test_images/straight_lines2.jpg')
#image = cv2.imread('test_images/test1.jpg')
#image = cv2.imread('test_images/test2.jpg')
#image = cv2.imread('test_images/test3.jpg')
#image = cv2.imread('test_images/test4.jpg')
#image = cv2.imread('test_images/test5.jpg') # bad
#image = cv2.imread('test_images/test6.jpg')


# load video
cap = cv2.VideoCapture('project_video.mp4')
#cap = cv2.VideoCapture('challenge_video.mp4')
#cap = cv2.VideoCapture('harder_challenge_video.mp4')
fps = cap.get(5)
frame_width = cap.get(3)
frame_height = cap.get(4)
# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
out = cv2.VideoWriter('myVideo.avi',fourcc, fps, (int(frame_width),int(frame_height)))

IIR_alpha_radius = 0.95
IIR_alpha_poly = 0.9
frameCounter = 0
bad_line = False
# process video
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret==True:
        frameCounter += 1
        print('Frame #'+str(frameCounter))  #1253
        undistorted_image = cv2.undistort(frame, mtx, dist, None, mtx)
        birdsEye = changePerspective(undistorted_image) # change perspective to top view
        binary_birdsEye = detectLane(birdsEye) # detect edges using sobel x absolute & color
        # find lane lines based on the quality of the current line
        # if 2 subsequent lines differs by 10% or more, current line is considered bad
        # sanity check can be improved by adding threshold for distance between lines, minimum curvature radius
        # if current line is bad, then line is discarded
        if ( (frameCounter == 1) | (bad_line == True) ):
            left_line_params,right_line_params,img_lines = computeLines(binary_birdsEye)
        else:
            left_line_params,right_line_params,img_lines,bad_line = computeLinesFaster(binary_birdsEye,IIR_left_line_params,IIR_right_line_params)
        # draws line based on the polyfit
        drawLines(left_line_params,right_line_params,img_lines)
        # double check calculation
        radius = computeCurvateRadius(left_line_params,right_line_params,img_lines)
        if (frameCounter == 1):
            IIR_radius = radius
            IIR_left_line_params = left_line_params
            IIR_right_line_params = right_line_params
        else:
            IIR_radius = IIR_radius*IIR_alpha_radius + radius*(1 - IIR_alpha_radius)
            IIR_left_line_params = IIR_left_line_params*IIR_alpha_poly + left_line_params*(1 - IIR_alpha_poly)
            IIR_right_line_params = IIR_right_line_params*IIR_alpha_poly + right_line_params*(1 - IIR_alpha_poly)

        # Fix calculation
        lane_offset = computeLaneOffset(IIR_left_line_params,IIR_right_line_params,birdsEye)

        if (frameCounter == 1):
            IIR_lane_offset = lane_offset
        else:
            IIR_lane_offset = IIR_lane_offset*IIR_alpha_radius + lane_offset*(1 - IIR_alpha_radius)

        highligthed_img = highlightLane(IIR_left_line_params,IIR_right_line_params,birdsEye)
        highlighted_img = changePerspectiveBack(highligthed_img,frame)
        final_image = addText2Img(highlighted_img,IIR_radius,(50,50),'Curvature radius = '+str(radius)+'m')
        final_image = addText2Img(final_image,IIR_lane_offset,(50,100),'Center lane offset = '+str(IIR_lane_offset)+'m')

        #plt.imshow(final_image)
        #plt.show()
        out.write(final_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
cap.release()
out.release()
cv2.destroyAllWindows()