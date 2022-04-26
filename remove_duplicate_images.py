import os
import argparse
import cv2
import math
from imaging_interview import preprocess_image_change_detection, compare_frames_change_detection

import logging
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)


CHANGE_PERCENTAGE = 0.1
CONTOUR_AREA_PERCENTAGE = 0.01


def is_valid_image(directory, image):
    """
    Check if the file is a valid image.

    Parameters
    ----------
    directory : str
        Directory with the images to check for duplicates
    image: str
        input image to check validity

    Returns
    -------
    boolean
        Whether the image is valid or not
    """

    if not image.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
        return False

    path = os.path.join(directory, image)
    image = cv2.imread(path)

    if image is None:
        return False

    return True


def create_image_dict(directory, image):
    """
    Create an image dictionary

    Parameters
    ----------
    directory : str
        Directory with the images to check for duplicates
    image: str
        input image to create dictionary

    Returns
    -------
    image_dict : dict
        Dictionary of an image
    """

    image_dict = dict()
    image_dict['path'] = os.path.join(directory, image)
    image_dict['data'] = cv2.imread(image_dict['path'])

    return image_dict


def compare_images(image1, image2):
    """
    Finds changes between two images.

    Parameters
    ----------
    image1 : dict
        First image to compare
    image2: dict
        Second image to compare

    Returns
    -------
    score float
        Value quantify how two images differ from each other
    thresh : numpy.ndarray
        Binary image of the absolute difference between the two images
    """

    image1['data'] = preprocess_image_change_detection(image1['data'])
    image2['data'] = preprocess_image_change_detection(image2['data'])

    image1Size = image1['imageHeight'] * image1['imageWidth']
    image2Size = image2['imageHeight'] * image2['imageWidth']

    # check if both images have the same size
    if image1Size != image2Size:
        # resize images when sizes are not matched
        if image1Size > image2Size:
            image2['data'] = cv2.resize(image2['data'], image1['data'].shape[::-1])
        elif image1Size < image2Size:
            image1['data'] = cv2.resize(image1['data'], image2['data'].shape[::-1])

    # set the minimum contour area to be 1% of the image size
    min_contour_area = image1['imageHeight'] * image1['imageWidth'] * CONTOUR_AREA_PERCENTAGE ** 2
    score, _, thresh = compare_frames_change_detection(image1['data'], image2['data'],
                                                       min_contour_area=min_contour_area)

    return score, thresh


def find_duplicates(directory):
    """
    Finds duplicated images in a directory based on a similarity measure. Images are considered duplicated if the
    difference in pixel area between them is less than 10%

    Parameters
    ----------
    directory : str
        Directory with the images to check for duplicates

    Returns
    -------
    duplicates : list(str)
        List of the lowest resolution duplicate images
    """

    images = os.listdir(directory)

    duplicates = []

    for i in range(len(images)):

        if not is_valid_image(directory, images[i]):
            continue

        image1 = create_image_dict(directory, images[i])

        if image1['path'] in duplicates:
            continue

        logging.info('Searching duplicates for {}'.format(images[i]))

        for j in range(i+1, len(images)):

            if not is_valid_image(directory, images[j]):
                continue

            image2 = create_image_dict(directory, images[j])

            if image2['path'] in duplicates:
                continue

            image1['imageHeight'], image1['imageWidth'] = image1['data'].shape[:2]
            image2['imageHeight'], image2['imageWidth'] = image2['data'].shape[:2]

            score, thresh = compare_images(image1.copy(), image2.copy())

            if score > (CHANGE_PERCENTAGE * math.prod(thresh.shape)):
                continue

            logging.info('Found duplicate {}'.format(images[j]))
            # adding the lower resolution image to the duplicate list
            if (image2['imageHeight']*image2['imageWidth']) > (image1['imageHeight']*image1['imageWidth']):
                duplicates.append(image1['path'])
            else:
                duplicates.append(image2['path'])

    logging.info('The search has been completed')

    return duplicates


def delete_duplicates(duplicateImages):
    """
    Removes duplicated images from the directory.

    Parameters
    ----------
    duplicateImages : list(str)
        List of the lowest resolution duplicate images
    """

    for image in duplicateImages:
        os.remove(image)
        logging.info('Image deleted'.format(image))

    logging.info('The deletion has been completed')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=str, help="directory with the images")
    args = parser.parse_args()

    duplicateImages = find_duplicates(args.directory)
    delete_duplicates(duplicateImages)
