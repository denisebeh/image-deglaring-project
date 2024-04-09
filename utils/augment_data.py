import cv2
import numpy as np

"""
Utility functions to perform data augmentation techniques on the training image
(rotation/color shifting/contrast adjustment etc.)
"""
def rotate_image_left(img):
    height, width = img.shape[:2]
    # first param: rotate from center, second param: angle, third param: scale
    rotation_matrix = cv2.getRotationMatrix2D((width/2, height/2), 20, 1)
    rotated_img = cv2.warpAffine(img, rotation_matrix, (width, height))
    return rotated_img


def rotate_image_right(img):
    height, width = img.shape[:2]
    # first param: rotate from center, second param: angle, third param: scale
    rotation_matrix = cv2.getRotationMatrix2D((width/2, height/2), 340, 1)
    rotated_img = cv2.warpAffine(img, rotation_matrix, (width, height))
    return rotated_img


def zoom_image(img, zoom_factor):
    height, width = img.shape[:2]
    # define new boundaries
    x1 = int(0.5 * width * (1 - 1/zoom_factor))
    x2 = int(width - 0.5 * width * (1 - 1/zoom_factor))
    y1 = int(0.5 * height *(1 - 1/zoom_factor))
    y2 = int(height - 0.5 * height * (1 - 1/zoom_factor))
    
    img_cropped = img[y1:y2,x1:x2]
    return cv2.resize(img_cropped, None, fx=zoom_factor, fy=zoom_factor)


def increase_brightness(img):
    bright = np.ones(img.shape, dtype="uint8") * 150
    bright_increase_img = cv2.add(img, bright)
    return bright_increase_img


def decrease_brightness(img):
    bright = np.ones(img.shape, dtype="uint8") * 70
    bright_decrease_img = cv2.subtract(img, bright)
    return bright_decrease_img


def flip_image_horizontal(img):
    flipped_img = cv2.flip(img, 3)
    return flipped_img


def sharpen_image(img):
    kernel = np.array([ [0, -1, 0],
                        [-1, 5, -1],
                        [0, -1, 0]  ])
    sharpened_img = cv2.filter2D(img, -1, kernel)
    return sharpened_img


def box_blur_image(img):
    blurred_img = cv2.blur(img, (3, 3))
    blurred_img = cv2.blur(blurred_img, (3, 3))
    return blurred_img


def positive_contrast_image(img):
    brightness = 2
    contrast = 10
    dummy = np.int16(img)
    dummy = dummy * (contrast/127+1) - contrast + brightness
    dummy = np.clip(dummy, 0, 255)
    img = np.uint8(dummy)
    return img
