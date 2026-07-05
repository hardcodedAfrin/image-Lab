import numpy as np
from PIL import Image, ImageOps
from matplotlib import pyplot as plt

# 1. Load the image and convert to grayscale using Pillow
img = Image.open('images/Baboon.png').convert('L')
img_array = np.array(img)

# 2. Compute and plot the global histogram
plt.figure(figsize=(8, 5))
plt.hist(img_array.flatten(), bins=256, range=(0, 256), color='blue', histtype='step')
#img_array.flatten(): Converts a 2D or 3D image matrix (Rows*Columns) into a single, massive 1D linear array of numbers. 
#bins =256, Creates exactly 256 structural "baskets" or columns.
plt.title('Global Histogram')
plt.xlabel('Pixel Intensity (0-255)')
plt.ylabel('Number of Pixels')
plt.xlim([0, 256])
plt.show()

# 3. Process the image using Global Histogram Equalization
processed_img = ImageOps.equalize(img)
processed_array = np.array(processed_img)

# 4. Show the processed image separately (Second Figure)
plt.figure(figsize=(6, 6))
plt.imshow(img_array, cmap='gray')
plt.title('Original Image')
plt.axis('off')  
plt.show()

# 5. Show the processed image separately (Second Figure)
plt.figure(figsize=(6, 6))
plt.imshow(processed_array, cmap='gray')
plt.title('Processed Image After Global Histogram Equalization')
plt.axis('off')  
plt.show()


#if want to show together, use the below code instead of the above two sections
# Show the original image
from matplotlib import axes
#Create a layout to show the histogram and the processed image side-by-side
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

axes[0].imshow(img_array, cmap='gray')
axes[0].set_title('Original Image')
axes[0].axis('off')  
#  Show the processed image besides the original image
axes[1].imshow(processed_array, cmap='gray')
axes[1].set_title('Processed Image')
axes[1].axis('off') 