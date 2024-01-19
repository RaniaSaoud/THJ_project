import cv2
import numpy as np

def preprocess_image(image_path, size=(256, 256), convert_color=True):
 

    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    
    if image is None:
        raise ValueError("Image not found or invalid image path.")


    image = cv2.resize(image, size)

    if convert_color:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    image = image.astype(np.float32) / 255.0

    return image

def display_image(image, window_name="Image"):
    
    if image.dtype == np.float32:
        image = (image * 255).astype(np.uint8)

    cv2.imshow(window_name, image)
    cv2.waitKey(0)  
    cv2.destroyAllWindows()

def threshold_segmentation(image):
  
    if len(image.shape) > 2 and image.shape[2] == 3:
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
  
    if image.dtype == np.float32:
        image = (image * 255).astype(np.uint8)
    
   
    _, thresh_img = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh_img


def edge_detection_segmentation(image, low_threshold=50, high_threshold=150):
    
    if image.dtype == np.float32:
        image = (image * 255).astype(np.uint8)

    edges = cv2.Canny(image, low_threshold, high_threshold)
    return edges



def calculate_accuracy(segmented, ground_truth):
  
    
    assert segmented.shape == ground_truth.shape, "Images must be the same size for accuracy calculation"
    
    
    correct = np.sum(segmented == ground_truth)
    
    
    total_pixels = ground_truth.size
    accuracy = correct / total_pixels
    
    return accuracy



def preprocess_ground_truth(ground_truth_path, target_size):
   
    
    ground_truth = cv2.imread(ground_truth_path, cv2.IMREAD_GRAYSCALE)
    
    if ground_truth is None:
        raise ValueError("Ground truth image not found or invalid image path.")

   
    ground_truth = cv2.resize(ground_truth, target_size)

    
    _, ground_truth = cv2.threshold(ground_truth, 127, 255, cv2.THRESH_BINARY)

    return ground_truth


def watershed_segmentation(image):
    
    if image.dtype != np.uint8:
        image = np.uint8(image*255)

    
    if len(image.shape) == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    
    
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    
    
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    
   
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
    
    
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    
    
    _, markers = cv2.connectedComponents(sure_fg)
    
    
    markers = markers + 1
    
    
    markers[unknown == 255] = 0
    
   
    markers = cv2.watershed(image, markers)
    image[markers == -1] = [255, 0, 0]  

    markers[markers == -1] = 0  
    segmented_image = markers > 1  
    
    return segmented_image.astype(np.uint8) * 255
    


# def kmeans_segmentation(image, K=2):
    
#     if image.dtype != np.uint8:
#         image = (image * 255).astype(np.uint8)
    
   
#     pixel_values = image.reshape((-1, 3)) 
#     pixel_values = np.float32(pixel_values)
    
  
#     criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
#     _, labels, centers = cv2.kmeans(pixel_values, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
  
#     centers = np.uint8(centers)
#     segmented_image = centers[labels.flatten()]
#     segmented_image = segmented_image.reshape(image.shape)
    
#     return segmented_image



image_path = 'one.jpg' 
ground_truth_path = 'segmented_image.jpg'  


preprocessed_image = preprocess_image(image_path, convert_color=False)
print(f"Preprocessed image size: {preprocessed_image.shape}")

ground_truth = preprocess_ground_truth(ground_truth_path, target_size=preprocessed_image.shape[:2])
print(f"Ground truth size: {ground_truth.shape}")

thresh_image = threshold_segmentation(preprocessed_image)
print(f"Threshold image size: {thresh_image.shape}")

edge_image = edge_detection_segmentation(preprocessed_image)
print(f"Edge image size: {edge_image.shape}")

# kmeans_image = kmeans_segmentation(preprocessed_image, K=2)
# kmeans_image_gray = cv2.cvtColor(kmeans_image, cv2.COLOR_BGR2GRAY)
# _, kmeans_image_binary = cv2.threshold(kmeans_image_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# print(f"K-means image size after conversion: {kmeans_image_binary.shape}")



watershed_image = watershed_segmentation(preprocessed_image)
print(f"watershed image size: {watershed_image.shape}")


thresh_accuracy = calculate_accuracy(thresh_image, ground_truth)
edge_accuracy = calculate_accuracy(edge_image, ground_truth)

watershed_accuracy = calculate_accuracy(watershed_image, ground_truth)
# kmeans_accuracy = calculate_accuracy(kmeans_image_binary, ground_truth)


print(f"Watershed Segmentation Accuracy: {watershed_accuracy}")
# print(f"K-means Segmentation Accuracy: {kmeans_accuracy}")
print(f"Threshold Segmentation Accuracy: {thresh_accuracy}")
print(f"Edge Detection Segmentation Accuracy: {edge_accuracy}")




method_accuracies = {
    'watershed': watershed_accuracy,
    'threshold': thresh_accuracy,
    'edge': edge_accuracy
}


binary_watershed = watershed_segmentation(preprocessed_image) > 0


# binary_kmeans = kmeans_segmentation(preprocessed_image) > 0
# kmeans_image_gray = cv2.cvtColor(kmeans_image, cv2.COLOR_BGR2GRAY)
# _, kmeans_image_binary = cv2.threshold(kmeans_image_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)


binary_threshold = threshold_segmentation(preprocessed_image) > 0

binary_edge = edge_detection_segmentation(preprocessed_image) > 0

segmentations_stack = np.stack((binary_watershed,  binary_threshold, binary_edge), axis=-1)


dominant_methods_idx = np.argmax(segmentations_stack * list(method_accuracies.values()), axis=-1)


dominant_segmentation = np.take_along_axis(segmentations_stack, np.expand_dims(dominant_methods_idx, axis=-1), axis=-1).squeeze()


recolored_image = np.where(dominant_segmentation[..., None], 0, 255)  

# Displaying all the results of the methods

binary_watershed_8bit = (binary_watershed * 255).astype(np.uint8)
#binary_kmeans_8bit = (binary_kmeans * 255).astype(np.uint8)
binary_threshold_8bit = (binary_threshold * 255).astype(np.uint8)
binary_edge_8bit = (binary_edge * 255).astype(np.uint8)


display_image(binary_watershed_8bit, "Watershed Segmentation")
#display_image(binary_kmeans_8bit, "K-means Segmentation")
display_image(binary_threshold_8bit, "Threshold Segmentation")
display_image(binary_edge_8bit, "Edge Detection Segmentation")




display_image(recolored_image.astype(np.uint8), "Final Recolored Image")