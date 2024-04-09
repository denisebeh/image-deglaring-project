import cv2

"""
Helper utility functions to preprocess raw training data
"""
def resize_image(img):
    """
    takes in a cv2 loaded image and splits it into 3 equal parts (512x512)
    """
    h, w, channels = img.shape
    
    one_third = w//3
    
    # this will be the first column
    left_part = img[:, :one_third] 
    middle_part = img[:, one_third:one_third*2] 
    right_part = img[:, one_third*2:] 

    return (left_part, middle_part, right_part)


def grayscale_image(img):
    """
    - converts image to grayscale
    - as the image has an alpha channel (i.e. 4 channels), it is first converted to RGB then to grayscale 
    - only for glare image and ground truth image
    """
    if img.shape[2] == 4:
        rgb_img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    else:
        rgb_img = img

    gray_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)
    return gray_img


def normalize_image(img):
    """
    performs image normalization
    """
    img_normalized = cv2.normalize(img, None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    return img_normalized
