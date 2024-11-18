import cv2
import numpy as np

def tight_coordinate_generator(word_img):
    """
    Generates the tight bounding box coordinates for the given image.

    Args:
        word_img (numpy.ndarray): Input image (grayscale).

    Returns:
        tuple: Bounding box coordinates (xmin, xmax, ymin, ymax).
    """
    
    threshold = 100
    coords = np.column_stack(np.where(word_img < threshold))
    ymin = np.inf
    ymax = -np.inf
    for cord in coords:
        if cord[0] < ymin:
            ymin = cord[0]
        elif cord[0] > ymax:
            ymax = cord[0]

    xmin = np.inf
    xmax = -np.inf
    for cord in coords:
        if cord[1] < xmin:
            xmin = cord[1]
        elif cord[1] > xmax:
            xmax = cord[1]

    return xmin, xmax, ymin, ymax

def select_text_area(image):
    """
    Removes the non-black pixels around the image based on black pixel detection.

    Args:
        image (numpy.ndarray): Input image (grayscale).
    
    Returns:
        numpy.ndarray: Cropped image containing only the black pixel region.
    """
    # Define a threshold for black pixels
    threshold = 40  # Adjust based on the specific requirements

    # Identify coordinates of black pixels
    coords = np.column_stack(np.where(image < threshold))

    if coords.size == 0:
        # If no black pixels are found, return an empty image of the same size
        return np.zeros_like(image)

    # Calculate tight bounding box
    xmin, xmax, ymin, ymax = tight_coordinate_generator(image)

    # Crop the image using the bounding box
    cropped_image = image[ymin:ymax+1, xmin:xmax+1]

    return cropped_image