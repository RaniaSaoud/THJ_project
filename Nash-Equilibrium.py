# import cv2
# import numpy as np
# from scipy.optimize import linprog

# def preprocess_image(image_path, size=(256, 256), convert_color=True):
#     image = cv2.imread(image_path, cv2.IMREAD_COLOR)
#     if image is None:
#         raise ValueError("Image not found or invalid image path.")
#     image = cv2.resize(image, size)
#     if convert_color:
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     image = image.astype(np.float32) / 255.0
#     return image

# def display_image(image, window_name="Image"):
#     if image.dtype != np.uint8:
#         if image.max() <= 1.0:
#             image = (image * 255).astype(np.uint8)
#         else:
#             image = image.astype(np.uint8)
#     cv2.imshow(window_name, image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

# # Segmentation functions
# def threshold_segmentation(image):
#     if len(image.shape) > 2 and image.shape[2] == 3:
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     if image.dtype == np.float32:
#         image = (image * 255).astype(np.uint8)
#     _, thresh_img = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#     return thresh_img

# def edge_detection_segmentation(image, low_threshold=50, high_threshold=150):
#     if image.dtype == np.float32:
#         image = (image * 255).astype(np.uint8)
#     edges = cv2.Canny(image, low_threshold, high_threshold)
#     return edges

# def watershed_segmentation(image):
#     if image.dtype != np.uint8:
#         image = np.uint8(image * 255)
#     if len(image.shape) == 3 and image.shape[2] == 3:
#         gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     else:
#         gray = image
#     _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
#     kernel = np.ones((3, 3), np.uint8)
#     opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
#     sure_bg = cv2.dilate(opening, kernel, iterations=3)
#     dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
#     _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
#     sure_fg = np.uint8(sure_fg)
#     unknown = cv2.subtract(sure_bg, sure_fg)
#     _, markers = cv2.connectedComponents(sure_fg)
#     markers = markers + 1
#     markers[unknown == 255] = 0
#     markers = cv2.watershed(image, markers)
#     image[markers == -1] = [255, 0, 0]
#     markers[markers == -1] = 0
#     segmented_image = markers > 1
#     return segmented_image.astype(np.uint8) * 255


# def calculate_accuracy(segmented, ground_truth):
#     assert segmented.shape == ground_truth.shape, "Images must be the same size for accuracy calculation"
#     correct = np.sum(segmented == ground_truth)
#     total_pixels = ground_truth.size
#     accuracy = correct / total_pixels
#     return accuracy


# def preprocess_ground_truth(ground_truth_path, target_size):
#     ground_truth = cv2.imread(ground_truth_path, cv2.IMREAD_GRAYSCALE)
#     if ground_truth is None:
#         raise ValueError("Ground truth image not found or invalid image path.")
#     ground_truth = cv2.resize(ground_truth, target_size)
#     _, ground_truth = cv2.threshold(ground_truth, 127, 255, cv2.THRESH_BINARY)
#     return ground_truth


# def calculate_mixed_strategy_accuracies(method_accuracies):
#     """
#     Calculate the mixed strategy Nash Equilibrium accuracies for each method.
    
#     :param method_accuracies: A dictionary with the accuracy of each method.
#     :return: A dictionary with the mixed strategy probabilities for each method.
#     """
    
#     # Number of methods
#     num_methods = len(method_accuracies)
    
#     # Objective function (c vector in linprog) - we want to minimize the negative of accuracies since linprog does minimization
#     c = [-accuracy for accuracy in method_accuracies.values()]

#     # Constraints - sum of probabilities must be 1
#     A_eq = [np.ones(num_methods)]
#     b_eq = [1]

#     # Bounds for each probability variable - between 0 and 1
#     bounds = [(0, 1) for _ in range(num_methods)]

#     # Solve the linear program
#     res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')

#     # Check if the optimization was successful
#     if not res.success:
#         raise ValueError("Optimization failed")

#     # The mixed strategy probabilities
#     probabilities = res.x

#     # Map the probabilities back to the methods
#     mixed_strategy_accuracies = dict(zip(method_accuracies.keys(), probabilities))
    
#     return mixed_strategy_accuracies

import cv2
import numpy as np
from scipy.optimize import linprog

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

# Segmentation functions
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

def watershed_segmentation(image):
    if image.dtype != np.uint8:
        image = np.uint8(image * 255)
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


def calculate_mixed_strategy_accuracies(method_accuracies):
    """
    Calculate the mixed strategy Nash Equilibrium accuracies for each method.
    
    :param method_accuracies: A dictionary with the accuracy of each method.
    :return: A dictionary with the mixed strategy probabilities for each method.
    """
    
    # Number of methods
    num_methods = len(method_accuracies)
    
    # Objective function (c vector in linprog) - we want to minimize the negative of accuracies since linprog does minimization
    c = [-accuracy for accuracy in method_accuracies.values()]

    # Constraints - sum of probabilities must be 1
    A_eq = [np.ones(num_methods)]
    b_eq = [1]

    # Bounds for each probability variable - between 0 and 1
    bounds = [(0, 1) for _ in range(num_methods)]

    # Solve the linear program
    res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')

    # Check if the optimization was successful
    if not res.success:
        raise ValueError("Optimization failed")

    # The mixed strategy probabilities
    probabilities = res.x

    # Map the probabilities back to the methods
    mixed_strategy_accuracies = dict(zip(method_accuracies.keys(), probabilities))
    
    return mixed_strategy_accuracies

def find_nash_equilibrium(height, width, pixel_strategies, method_accuracies):

    # Calculate mixed strategy Nash equilibrium accuracies
    mixed_strategy_accuracies = calculate_mixed_strategy_accuracies(method_accuracies)
    
    # Initialize the final segmentation array
    final_segmentation = np.zeros((height, width), dtype=np.uint8)
    
    # Iterate over each pixel to find Nash equilibrium
    for y in range(height):
        for x in range(width):
            # For each pixel, compute the gains for object and background using the mixed strategies
            gain_object = sum(mixed_strategy_accuracies[method] * pixel_strategies[method][y, x] for method in method_accuracies)
            gain_background = sum(mixed_strategy_accuracies[method] * (1 - pixel_strategies[method][y, x]) for method in method_accuracies)

            # Determine the pixel color based on the higher gain
            # If gain for object is higher, the pixel belongs to the object; otherwise, it's background
            final_segmentation[y, x] = 0 if gain_object > gain_background else 255  # 0 for object (black), 255 for background (white)
    
    return final_segmentation




# Main processing
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

pixel_strategies = {
    'watershed': binary_watershed,
    'threshold': binary_threshold,
    'edge': binary_edge
}

height, width = preprocessed_image.shape[:2]
final_segmentation = find_nash_equilibrium(height, width, pixel_strategies, method_accuracies)

# Display the final recolored image
display_image(final_segmentation, "Nash Equilibrium Segmentation")




# image_path = '0005.jpg'
# ground_truth_path = '0005.png'

# preprocessed_image = preprocess_image(image_path, convert_color=False)
# ground_truth = preprocess_ground_truth(ground_truth_path, target_size=preprocessed_image.shape[:2])

# thresh_image = threshold_segmentation(preprocessed_image)
# display_image(thresh_image, "Threshold Segmentation")
# edge_image = edge_detection_segmentation(preprocessed_image)
# display_image(edge_image, "Edge Detection Segmentation")
# watershed_image = watershed_segmentation(preprocessed_image)
# watershed_display = (watershed_image > 1).astype(np.uint8) * 255
# display_image(watershed_display, "Watershed Segmentation")

# thresh_accuracy = calculate_accuracy(thresh_image, ground_truth)
# edge_accuracy = calculate_accuracy(edge_image, ground_truth)
# watershed_accuracy = calculate_accuracy(watershed_image, ground_truth)

# method_accuracies = {
#     'watershed': watershed_accuracy,
#     'threshold': thresh_accuracy,
#     'edge': edge_accuracy
# }

# binary_watershed = watershed_image > 1
# binary_threshold = thresh_image > 0
# binary_edge = edge_image > 0

# pixel_strategies = {
#     'watershed': binary_watershed,
#     'threshold': binary_threshold,
#     'edge': binary_edge
# }

# height, width = preprocessed_image.shape[:2]
# final_segmentation = find_nash_equilibrium(height, width, pixel_strategies, method_accuracies)


# display_image(final_segmentation, "Nash Equilibrium Segmentation")
