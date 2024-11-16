import cv2
import numpy as np


def _process_image(img):
    """Some image processing is performed
       (Gaussian Blur, OTSU Binarization and Dilation)

    Args:
        img (<numpy.array>): Image as numpy array

    Returns:
        <numpy.array>: Image as numpy array
    """
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img = cv2.GaussianBlur(img, (7, 7), 0)
    img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (6, 1))
    img = cv2.dilate(img, kernel, iterations=5)

    return img


def _rotate_image(img, angle:float):
    """Rotates an Image for a given angle

    Args:
        img (<numpy.array>): Image as numpy array
        angle (float): specify the target angle

    Returns:
        <numpy.array>: Image as numpy array
    """ 
    h, w = img.shape[:2]

    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    img = cv2.warpAffine(
        img, M, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255)
    )

    return img


def _find_angle(img):
    """Finds the skewness present in an image in terms of angle

    Args:
        img (<numpy.array>): Image as numpy array

    Returns:
        float: resulting angle
    """
    img = _process_image(img)

    contours = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[0]

    considered_contours = []
    contour_properties = []
    widths = []
    heights = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        angle = cv2.minAreaRect(contour)[-1]
        contour_properties.append([x, y, w, h, angle])
        considered_contours.append(contour)
        widths.append(w)
        heights.append(h)
        contour_area = w * h

    mean_width = np.mean(widths)
    std_width = np.std(widths)
    mean_height = np.mean(heights)
    std_height = np.std(heights)

    considered_angles = []
    tmp_ctr = []

    for i, width in enumerate(widths):
        if (width >= mean_width) and (heights[i] <= (mean_height + 2.5*std_height)):

            tmp_angle = contour_properties[i][-1]

            if tmp_angle < -45:
                tmp_angle = 90 + tmp_angle

            elif tmp_angle > 45:
                tmp_angle = tmp_angle - 90

            if abs(tmp_angle) < 45:
                considered_angles.append(tmp_angle)
                tmp_ctr.append(considered_contours[i])

    if(len(considered_angles)==0):
        return 0

    data_mean, data_std = np.mean(considered_angles), np.std(considered_angles)
    cut_off = data_std * 1
    lower, upper = data_mean - cut_off, data_mean + cut_off

    skew_angles = []

    for i, angle in enumerate(considered_angles):
        if abs(angle) < 44:
            if (lower) <= angle <= (upper):
                skew_angles.append(angle)
   
    result_angle = np.mean(skew_angles)

    if result_angle < -45:
        result_angle = 90 + result_angle

    return result_angle


def correct_skew(img):
    """Corrects the skewness present in an image

    Args:
        img (<numpy.array>): Image as numpy array

    Returns:
        <numpy.array>: Image as numpy array
    """
    angle = _find_angle(img)
    final_img = _rotate_image(img, angle)

    return final_img, angle