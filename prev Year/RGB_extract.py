import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# 1. Load the original color image using Pillow
pil_img = Image.open('images/lady.jpg')
rgb_img = np.array(pil_img)

# 2. Convert to grayscale using Pillow's 'L' (Luminance) mode
# This uses the standard ITU-R 601-2 luma transform under the hood
gray_img = np.array(pil_img.convert('L'))

# 3. Create isolated color channel images by zeroing out other channels
# Create empty black canvasses with the exact same shape as the original image
red_img = np.zeros_like(rgb_img)
green_img = np.zeros_like(rgb_img)
blue_img = np.zeros_like(rgb_img)

# Copy over ONLY the specific channel data, [Rows, Columns, Channels]
red_img[:, :, 0] = rgb_img[:, :, 0]    # Copy Red, leave Green/Blue as 0
green_img[:, :, 1] = rgb_img[:, :, 1]  # Copy Green, leave Red/Blue as 0
blue_img[:, :, 2] = rgb_img[:, :, 2]   # Copy Blue, leave Red/Green as 0

# 4. Display all 5 images side-by-side
plt.figure(figsize=(20, 4.5))

# Original RGB Image
plt.subplot(151)
plt.imshow(rgb_img)
plt.title('Original RGB')
plt.axis('off')

# Grayscale Image (Mode L)
plt.subplot(152)
plt.imshow(gray_img, cmap='gray')
plt.title('Grayscale (Mode L)')
plt.axis('off')

# Red Channel Image (Now visibly Red)
plt.subplot(153)
plt.imshow(red_img)  # No cmap='gray' needed here because it is a true color array
plt.title('Red Channel Only')
plt.axis('off')

# Green Channel Image (Now visibly Green)
plt.subplot(154)
plt.imshow(green_img)
plt.title('Green Channel Only')
plt.axis('off')

# Blue Channel Image (Now visibly Blue)
plt.subplot(155)
plt.imshow(blue_img)
plt.title('Blue Channel Only')
plt.axis('off')

plt.tight_layout()
plt.show()