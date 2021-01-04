# image_creation

Used in EGH455 to create image data for training. Uses pillow, overlays a range of images in the images folder 
over a range of backgrounds and saves the images aswell as the bounding box coordinates of the places image.

Randomly adds rotation, contrast, eculidian noise, resizes the imgaes to the overlayed images. Other files is to decrease training time
by converting the data to a single file which can be used to train tensorflow 1 models.
