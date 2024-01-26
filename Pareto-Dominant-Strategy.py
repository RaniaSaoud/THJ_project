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
    
    if image.dtype != np.uint8:
        if image.max() <= 1.0:  
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)

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
    
def is_pareto_dominant(current_strategies, new_strategy, method_accuracies):
    for strategy, accuracy in current_strategies.items():
        if method_accuracies[new_strategy] <= accuracy:
            return False
    return True


def find_pareto_dominant_strategies(method_accuracies):
    pareto_dominant_strategies = []
    for strategy, accuracy in method_accuracies.items():
        is_dominated = False
        for other_strategy, other_accuracy in method_accuracies.items():
            if strategy != other_strategy and other_accuracy >= accuracy:
                is_dominated = True
                break
        if not is_dominated:
            pareto_dominant_strategies.append(strategy)
    return pareto_dominant_strategies


image_path = '0005.jpg'
ground_truth_path = '0005.png'

preprocessed_image = preprocess_image(image_path, convert_color=False)
ground_truth = preprocess_ground_truth(ground_truth_path, target_size=preprocessed_image.shape[:2])

thresh_image = threshold_segmentation(preprocessed_image)
display_image(thresh_image, "Threshold Segmentation")

edge_image = edge_detection_segmentation(preprocessed_image)
display_image(edge_image, "Edge Detection Segmentation")

watershed_image = watershed_segmentation(preprocessed_image)
watershed_display = (watershed_image > 1).astype(np.uint8) * 255  
display_image(watershed_display, "Watershed Segmentation")

thresh_accuracy = calculate_accuracy(thresh_image, ground_truth)
edge_accuracy = calculate_accuracy(edge_image, ground_truth)
watershed_accuracy = calculate_accuracy(watershed_image, ground_truth)

method_accuracies = {
    'watershed': watershed_accuracy,
    'threshold': thresh_accuracy,
    'edge': edge_accuracy
}

binary_watershed = watershed_image > 1
binary_threshold = thresh_image > 0
binary_edge = edge_image > 0

height, width = preprocessed_image.shape[:2]
final_segmentation = np.zeros((height, width), dtype=np.uint8)


pareto_dominant_strategies = find_pareto_dominant_strategies(method_accuracies)


for y in range(height):
    for x in range(width):
        pixel_strategies = {
            'watershed': binary_watershed[y, x],
            'threshold': binary_threshold[y, x],
            'edge': binary_edge[y, x]
        }
        
        pixel_payoffs = {method: accuracy for method, accuracy in method_accuracies.items() if method in pareto_dominant_strategies}
        dominant_strategy = max(pixel_payoffs, key=pixel_payoffs.get)
        player1_gain = pixel_payoffs[dominant_strategy] if pixel_strategies[dominant_strategy] else 1 - pixel_payoffs[dominant_strategy]
        player2_gain = 1 - player1_gain
        
        if player1_gain > player2_gain:
            final_segmentation[y, x] = 255 
        else:
            final_segmentation[y, x] = 0  


display_image(final_segmentation, "Final Segmentation")