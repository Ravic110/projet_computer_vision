import cv2
import numpy as np
import webcolors  # Importez la bibliothèque webcolors

# Read the image
img = cv2.imread("test4.jpg")

# Convert the image to HSV format
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Define the range of colors to detect in HSV format
lower_range = np.array([0, 0, 0])
upper_range = np.array([255, 255, 255])

# Create a mask for the range of colors
mask = cv2.inRange(hsv, lower_range, upper_range)

# Apply the mask to the original image
result = cv2.bitwise_and(img, img, mask=mask)

# Convert the result to grayscale
gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

# Threshold the grayscale image to binary
ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)

# Find contours in the binary image
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Create an empty list to store the colors found in the image
colors = []

# Loop through each contour and find the average color within it
for contour in contours:
    # Create a mask for each contour
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [contour], 0, 255, -1)

    # Find the average color within the contour using the mask
    mean_color = cv2.mean(img, mask=mask)[:3]

    # Add the mean color to the list of colors
    colors.append(mean_color)

