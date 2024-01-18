import cv2
from skimage import filters
import matplotlib.pyplot as plt

# Load the image
image_path = 'one.jpg'  # Replace with your image path
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Apply a threshold
threshold_value = filters.threshold_otsu(image)
binary_image = image > threshold_value

# Save the segmented image
output_path = 'segmented_image.jpg'  # Replace with your desired output path
cv2.imwrite(output_path, binary_image.astype('uint8') * 255)

# Display the original and segmented image for visualization
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(binary_image, cmap='gray')
plt.title('Segmented Image')
plt.axis('off')

plt.show()
