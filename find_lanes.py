import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import glob

####################################### calibrate camera #####################################

images = glob.glob('./camera_cal/calibration*.jpg')


objpoints = []
imgpoints = []
nx = 9
ny = 6
objp = np.zeros((ny*nx,3), np.float32)
objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2)

fig, axes = plt.subplots(5,4, figsize=(15,15))
axes = np.ravel(axes)

for n, fname in enumerate(images):
	cal_img = cv2.imread(fname)
	gray_cal_img = cv2.cvtColor(cal_img, cv2.COLOR_BGR2GRAY)
	ret, corners = cv2.findChessboardCorners(gray_cal_img, (nx, ny), None)
	axes[n].axis('off')
	if ret == True:		
		imgpoints.append(corners)
		objpoints.append(objp)
		cv2.drawChessboardCorners(cal_img, (nx, ny), corners, ret)
	axes[n].imshow(cal_img)

fig.tight_layout()
	

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray_cal_img.shape[::-1], None, None)

fig, axes = plt.subplots(5,4, figsize=(15,15))
axes = np.ravel(axes)

for n, fname in enumerate(images):
	cal_img = cv2.imread(fname)
	dst = cv2.undistort(cal_img, mtx, dist, None, mtx)
	axes[n].axis('off')
	axes[n].imshow(dst)

fig.tight_layout()
		
################################## Test image ##################################

test_img_fname = './test_images/test5.jpg'

test_img = cv2.imread(test_img_fname)
test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
test_img_hls = cv2.cvtColor(test_img, cv2.COLOR_RGB2HLS)

fig, axes = plt.subplots(2,2, figsize=(35,18))
axes = np.ravel(axes)

axes[0].imshow(test_img)
axes[0].axis('off')
axes[0].set_title('Original image')

undist_test_img = cv2.undistort(test_img, mtx, dist, None, mtx)
axes[1].imshow(undist_test_img)
axes[1].axis('off')
axes[1].set_title('Undistroted image')


axes[2].imshow(test_img)
axes[2].axis('off')
axes[2].set_title('Perspective transform points')
axes[2].plot(579,460, '.')
axes[2].plot(705,460, '.')
axes[2].plot(270,675, '.')
axes[2].plot(1045,675, '.')

def warp(img):

	img_size = (img.shape[1], img.shape[0])	
	src = np.float32([[579,460],
					  [705,460],
					  [270,675],
					  [1045,675]])

	dst = np.float32([[300, 0],
					  [img.shape[1]-300,0],
					  [300,img.shape[0]],
					  [img.shape[1]-300,img.shape[0]]])

	M = cv2.getPerspectiveTransform(src, dst)
	Minv = cv2.getPerspectiveTransform(dst, src)
	warped = cv2.warpPerspective(img, M, img_size, flags = cv2.INTER_LINEAR)

	return warped

warped_img = warp(test_img)
axes[3].imshow(warped_img)
axes[3].axis('off')
axes[3].set_title('Warped image')



fig.tight_layout()



fig, axes = plt.subplots(2,2, figsize=(35,18))
axes = np.ravel(axes)

gray_test_img = cv2.cvtColor(undist_test_img, cv2.COLOR_RGB2GRAY)

def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):

	if orient is 'x':
		sobel = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize = sobel_kernel)
	elif orient is 'y':
		sobel = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize = sobel_kernel)
	abs_sobel = np.absolute(sobel)
	scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
	grad_binary = np.zeros_like(scaled_sobel)
	grad_binary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

	return grad_binary

def mag_thresh(img, sobel_kernel=3, thresh=(0, 255)):
	sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize = sobel_kernel)
	sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize = sobel_kernel)
	sobel_mag = np.sqrt(np.square(sobelx) + np.square(sobely))
	scaled_sobel_mag = np.uint8(255*sobel_mag/np.max(sobel_mag))
	grad_binary = np.zeros_like(scaled_sobel_mag)
	grad_binary[(scaled_sobel_mag >= thresh[0]) & (scaled_sobel_mag <= thresh[1])] = 1

	return grad_binary

def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
	sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize = sobel_kernel)
	sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize = sobel_kernel)
	abs_sobelx = np.absolute(sobelx)
	abs_sobely = np.absolute(sobely)
	sobel_angle = np.arctan2(abs_sobely, abs_sobelx)
	grad_binary = np.zeros_like(sobel_angle)
	grad_binary[(sobel_angle >= thresh[0]) & (sobel_angle <= thresh[1])] = 1

	return grad_binary

def hls_threshold(hls, channel='s', thresh=(0,255)):
	if channel is 'h':
		x = hls[:,:,0]
	elif channel is 'l':
		x = hls[:,:,1]
	elif channel is 's':
		x = hls[:,:,2]
	binary_output = np.zeros_like(x)
	binary_output[(x >= thresh[0]) & (x <= thresh[1])] = 1
	return binary_output

def rgb_threshold(rgb, channel='r', thresh=(0,255)):
	if channel is 'r':
		x = rgb[:,:,0]
	elif channel is 'g':
		x = rgb[:,:,1]
	elif channel is 'b':
		x = rgb[:,:,2]
	binary_output = np.zeros_like(x)
	binary_output[(x >= thresh[0]) & (x <= thresh[1])] = 1
	return binary_output

sxbinary = abs_sobel_thresh(gray_test_img, orient='x', sobel_kernel=3, thresh=(20, 100))
sybinary = abs_sobel_thresh(gray_test_img, orient='y', sobel_kernel=3, thresh=(20, 100))
smagbinary = mag_thresh(gray_test_img, sobel_kernel=3, thresh=(20, 100))
sanglebinary = dir_threshold(gray_test_img, sobel_kernel=3, thresh=(0.7, 1.3))


axes[0].imshow(sxbinary, cmap='gray')
axes[0].axis('off')
axes[0].set_title('Sobel x')

axes[1].imshow(sybinary, cmap='gray')
axes[1].axis('off')
axes[1].set_title('Sobel y')

axes[2].imshow(smagbinary, cmap='gray')
axes[2].axis('off')
axes[2].set_title('Sobel mag')

axes[3].imshow(sanglebinary, cmap='gray')
axes[3].axis('off')
axes[3].set_title('Sobel angle')


fig.tight_layout()


fig, axes = plt.subplots(2,3, figsize=(35,18))
axes = np.ravel(axes)

axes[0].imshow(test_img[:,:,0], cmap='gray')
axes[0].axis('off')
axes[0].set_title('R channel')
axes[1].imshow(test_img[:,:,1], cmap='gray')
axes[1].axis('off')
axes[1].set_title('G channel')
axes[2].imshow(test_img[:,:,2], cmap='gray')
axes[2].axis('off')
axes[2].set_title('B channel')
axes[3].imshow(test_img_hls[:,:,0], cmap='gray')
axes[3].axis('off')
axes[3].set_title('H channel')
axes[4].imshow(test_img_hls[:,:,1], cmap='gray')
axes[4].axis('off')
axes[4].set_title('L channel')
axes[5].imshow(test_img_hls[:,:,2], cmap='gray')
axes[5].axis('off')
axes[5].set_title('S channel')

fig.tight_layout()


rthresh_binary = rgb_threshold(test_img, channel='r', thresh=(200,255))
gthresh_binary = rgb_threshold(test_img, channel='g', thresh=(200,255))
bthresh_binary = rgb_threshold(test_img, channel='b', thresh=(200,255))

hthresh_binary = hls_threshold(test_img_hls, channel='h', thresh=(0,50))
lthresh_binary = hls_threshold(test_img_hls, channel='l', thresh=(200,255))
sthresh_binary = hls_threshold(test_img_hls, channel='s', thresh=(200,255))


fig, axes = plt.subplots(2,3, figsize=(35,18))
axes = np.ravel(axes)

axes[0].imshow(rthresh_binary, cmap='gray')
axes[0].axis('off')
axes[0].set_title('R channel')
axes[1].imshow(gthresh_binary, cmap='gray')
axes[1].axis('off')
axes[1].set_title('G channel')
axes[2].imshow(bthresh_binary, cmap='gray')
axes[2].axis('off')
axes[2].set_title('B channel')
axes[3].imshow(hthresh_binary, cmap='gray')
axes[3].axis('off')
axes[3].set_title('H channel')
axes[4].imshow(lthresh_binary, cmap='gray')
axes[4].axis('off')
axes[4].set_title('L channel')
axes[5].imshow(sthresh_binary, cmap='gray')
axes[5].axis('off')
axes[5].set_title('S channel')

fig.tight_layout()



combined_binary = np.zeros_like(sxbinary)
combined_binary[(sxbinary == 1) | (rthresh_binary == 1) | (gthresh_binary == 1) | (sthresh_binary == 1) ] = 1

fig, axes = plt.subplots()
axes.imshow(combined_binary, cmap='gray')
axes.axis('off')
axes.set_title('Combined threshold')

binary_warped = warp(combined_binary)


fig, axes = plt.subplots()
axes.imshow(binary_warped, cmap='gray')
axes.axis('off')
axes.set_title('Combined threshold + Perspective transform')

img = binary_warped
plt.figure()
histogram = np.sum(img[img.shape[0]//2:,:], axis=0)
plt.plot(histogram)


# Assuming you have created a warped binary image called "binary_warped"
# Take a histogram of the bottom half of the image
histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
# Create an output image to draw on and  visualize the result
out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
# Find the peak of the left and right halves of the histogram
# These will be the starting point for the left and right lines
midpoint = np.int(histogram.shape[0]//2)
leftx_base = np.argmax(histogram[:midpoint])
rightx_base = np.argmax(histogram[midpoint:]) + midpoint

# Choose the number of sliding windows
nwindows = 9
# Set height of windows
window_height = np.int(binary_warped.shape[0]//nwindows)
# Identify the x and y positions of all nonzero pixels in the image
nonzero = binary_warped.nonzero()
nonzeroy = np.array(nonzero[0])
nonzerox = np.array(nonzero[1])
# Current positions to be updated for each window
leftx_current = leftx_base
rightx_current = rightx_base
# Set the width of the windows +/- margin
margin = 100
# Set minimum number of pixels found to recenter window
minpix = 50
# Create empty lists to receive left and right lane pixel indices
left_lane_inds = []
right_lane_inds = []

# Step through the windows one by one
for window in range(nwindows):
    # Identify window boundaries in x and y (and right and left)
    win_y_low = binary_warped.shape[0] - (window+1)*window_height
    win_y_high = binary_warped.shape[0] - window*window_height
    win_xleft_low = leftx_current - margin
    win_xleft_high = leftx_current + margin
    win_xright_low = rightx_current - margin
    win_xright_high = rightx_current + margin
    # Draw the windows on the visualization image
    cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),
    (0,255,0), 2) 
    cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),
    (0,255,0), 2) 
    # Identify the nonzero pixels in x and y within the window
    good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
    (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
    good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
    (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
    # Append these indices to the lists
    left_lane_inds.append(good_left_inds)
    right_lane_inds.append(good_right_inds)
    # If you found > minpix pixels, recenter next window on their mean position
    if len(good_left_inds) > minpix:
        leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
    if len(good_right_inds) > minpix:        
        rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

# Concatenate the arrays of indices
left_lane_inds = np.concatenate(left_lane_inds)
right_lane_inds = np.concatenate(right_lane_inds)

# Extract left and right line pixel positions
leftx = nonzerox[left_lane_inds]
lefty = nonzeroy[left_lane_inds] 
rightx = nonzerox[right_lane_inds]
righty = nonzeroy[right_lane_inds] 

# Fit a second order polynomial to each
left_fit = np.polyfit(lefty, leftx, 2)
right_fit = np.polyfit(righty, rightx, 2)

# Generate x and y values for plotting
ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
plt.figure()
plt.imshow(out_img)
plt.plot(left_fitx, ploty, color='yellow')
plt.plot(right_fitx, ploty, color='yellow')
plt.xlim(0, 1280)
plt.ylim(720, 0)













plt.show()