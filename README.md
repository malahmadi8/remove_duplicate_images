# remove_duplicate_images


Find and remove all similar-looking images in a directory.

## Run
```bash
> python remove_duplicate_images.py './dataset'
```

## How the program works

### FIND DUPLICATES

The program first runs find_duplicates function. This function perform the following:

1- Loop over all the files in the directory.

2- Check if the file is a valid image.

- A valid image file should ends with one of the extension ('.png', '.jpg', '.jpeg', '.tiff', '.bmp', ‘.gif’).
- Check if the image file is corrupted or contain an image.

3- Create a dictionary for the first image we want to compare

4- Run a nest loop over the other files in the directory. To speedup the search, this loop starts at the iterator of the first loop.

5- Again, the program check if the file is a valid image, then create a dictionary for the second image we want to compare.

6- Find change between the two images by running compare_images function. This function perform the following:

- Perform image preprocessing using preprocess_image_change_detection function.

- Resize the two images when their resolutions are not matched using resize function from Opencv.

- Compare the two images using compare_frames_change_detection.

7- If the similarity score is below a predefined threshold, then the low resolution image is added to the duplicate list


### DELETE DUPLICATES

Finally, the program runs delete_duplicates to delete all the duplicated images from the directory.


## Requirements
```bash
> pip install -r requirements.txt
```
