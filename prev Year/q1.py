import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Load image (GRAYSCALE)

img = Image.open('images/panda.jpeg').convert('L').resize((256, 256))
imgMat = np.array(img, dtype=np.float32)

# (b) Add Gaussian Noise

mean = 0
var = 0.01
sigma = var ** 0.5

gaussian_noise = np.random.normal(mean, sigma, imgMat.shape)
noisy_gaussian = imgMat + gaussian_noise * 255

noisy_gaussian = np.clip(noisy_gaussian, 0, 255)

# (c) Low-pass filter (Box filter)

def box_filter(mat, n=1):
    h, w = mat.shape
    out = np.zeros_like(mat)

    for i in range(h):
        for j in range(w):
            total = 0
            count = 0

            for dx in range(-n, n+1):
                for dy in range(-n, n+1):
                    x, y = i+dx, j+dy
                    if 0 <= x < h and 0 <= y < w:
                        total += mat[x][y]
                        count += 1

            out[i][j] = total / count

    return out

low_pass = box_filter(noisy_gaussian, n=1)


# (d) Display effect (comparison)
print("Original vs Noisy vs Low-pass ready")

# (e) Add Salt & Pepper Noise

def salt_pepper(img, density=0.05):
    out = img.copy()
    h, w = out.shape

    num_pixels = int(density * h * w)

    # salt (white)
    coords = [np.random.randint(0, i, num_pixels) for i in (h, w)]
    out[coords[0], coords[1]] = 255

    # pepper (black)
    coords = [np.random.randint(0, i, num_pixels) for i in (h, w)]
    out[coords[0], coords[1]] = 0

    return out

sp_noisy = salt_pepper(imgMat, density=0.05)


# (f) Remove Salt & Pepper noise using Median filter

def median_filter(mat, n=1):
    h, w = mat.shape
    out = np.zeros_like(mat)

    for i in range(h):
        for j in range(w):
            window = []

            for dx in range(-n, n+1):
                for dy in range(-n, n+1):
                    x, y = i+dx, j+dy
                    if 0 <= x < h and 0 <= y < w:
                        window.append(mat[x][y])

            out[i][j] = np.median(window)

    return out

median_filtered = median_filter(sp_noisy, n=1)


# (g) High-pass filter (Edge enhancement)

def high_pass(mat):
    blurred = box_filter(mat, n=1)
    hp = mat - blurred
    return np.clip(hp + 128, 0, 255)  # shift for visibility

high_pass_img = high_pass(imgMat)


# Show results

plt.figure(figsize=(12, 8))

plt.subplot(2,3,1)
plt.title("Original")
plt.imshow(imgMat, cmap='gray')

plt.subplot(2,3,2)
plt.title("Gaussian Noise")
plt.imshow(noisy_gaussian, cmap='gray')

plt.subplot(2,3,3)
plt.title("Low Pass Filter")
plt.imshow(low_pass, cmap='gray')

plt.subplot(2,3,4)
plt.title("Salt & Pepper")
plt.imshow(sp_noisy, cmap='gray')

plt.subplot(2,3,5)
plt.title("Median Filter")
plt.imshow(median_filtered, cmap='gray')

plt.subplot(2,3,6)
plt.title("High Pass")
plt.imshow(high_pass_img, cmap='gray')

plt.tight_layout()
plt.show()