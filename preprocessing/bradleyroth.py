import numpy as np
import cv2
from PIL import Image

def replace_yellow_with_black(image):
    """
    Replace all yellow pixels in a PIL image with black and return the modified image.

    Args:
    - pil_image (PIL.Image.Image): The input image in RGB format.

    Returns:
    - PIL.Image.Image: The modified image with yellow pixels replaced by black.
    """
    
    # Convert PIL image to NumPy array
    image_np = np.array(image)

    # Convert RGB to BGR for OpenCV compatibility
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    # Convert the image to HSV color space
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    # Define a range for the yellow color in HSV
    lower_yellow = np.array([15, 150, 150])  # Lower bound for yellow
    upper_yellow = np.array([45, 255, 255])  # Upper bound for yellow

    # Create a mask for yellow pixels
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # Set all yellow pixels to black (0, 0, 0)
    image_bgr[mask != 0] = [0, 0, 0]

    # Convert BGR back to RGB
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # Convert NumPy array back to PIL image
    return Image.fromarray(image_rgb)

def binarize_image(image, s=None, t=None):
    """
    Binarize a PIL image using the Bradley-Roth algorithm.

    Args:
    - image (PIL.Image.Image): The input image in RGB format.
    - s (int): The window size for computing the local mean and standard deviation.
    - t (float): The threshold for binarization.

    Returns:
    - PIL.Image.Image: The binarized image.
    """
    
    image = replace_yellow_with_black(image)
    image = image.convert('L')

    # Convert image to numpy array
    img = np.array(image).astype(np.float64)

    # Default window size is round(cols/8)
    if s is None:
        s = np.round(img.shape[1]/8)

    # Default threshold is 15% of the total
    # area in the window
    if t is None:
        t = 15.0

    # Compute integral image
    intImage = np.cumsum(np.cumsum(img, axis=1), axis=0)

    # Define grid of points
    (rows,cols) = img.shape[:2]
    (X,Y) = np.meshgrid(np.arange(cols), np.arange(rows))

    # Make into 1D grid of coordinates for easier access
    X = X.ravel()
    Y = Y.ravel()

    # Ensure s is even so that we are able to index into the image
    # properly
    s = s + np.mod(s,2)

    # Access the four corners of each neighbourhood
    x1 = X - s/2
    x2 = X + s/2
    y1 = Y - s/2
    y2 = Y + s/2

    # Ensure no coordinates are out of bounds
    x1[x1 < 0] = 0
    x2[x2 >= cols] = cols-1
    y1[y1 < 0] = 0
    y2[y2 >= rows] = rows-1

    # Ensures coordinates are integer
    x1 = x1.astype(np.int64)
    x2 = x2.astype(np.int64)
    y1 = y1.astype(np.int64)
    y2 = y2.astype(np.int64)

    # Count how many pixels are in each neighbourhood
    count = (x2 - x1) * (y2 - y1)

    # Compute the row and column coordinates to access
    # each corner of the neighbourhood for the integral image
    f1_x = x2
    f1_y = y2
    f2_x = x2
    f2_y = y1 - 1
    f2_y[f2_y < 0] = 0
    f3_x = x1-1
    f3_x[f3_x < 0] = 0
    f3_y = y2
    f4_x = f3_x
    f4_y = f2_y

    # Compute areas of each window
    sums = intImage[f1_y, f1_x] - intImage[f2_y, f2_x] - intImage[f3_y, f3_x] + intImage[f4_y, f4_x]

    # Compute thresholded image and reshape into a 2D grid
    out = np.ones(rows*cols, dtype=np.bool_)
    out[img.ravel()*count <= sums*(100.0 - t)/100.0] = False

    # Also convert back to uint8
    out = 255*np.reshape(out, (rows, cols)).astype(np.uint8)

    # convert to rgb
    out = cv2.cvtColor(out, cv2.COLOR_GRAY2RGB)

    # Return PIL image
    return out
