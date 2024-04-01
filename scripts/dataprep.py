#!/usr/bin/env python

"""utils.py: converts numpy spectrograms to matplotlib images"""

__author__      = "Sahib Julka <sahib.julka@uni-passau.de>"
__copyright__   = "GPL"


import os
import numpy as np
import matplotlib.pyplot as plt


# replace the paths below to match local data dir
DIR_IN = os.path.join(os.getcwd(), '../chorus_wave/npy_2/nochorus')
DIR_OUT = os.path.join(os.getcwd(), 'processed', 'nochorus')
if not os.path.exists(DIR_OUT):
    os.makedirs(DIR_OUT)
if not os.path.exists(DIR_IN):
    DIR_IN = input("Please enter input dat a path:")


def convert_to_img(filename, dir_in, dir_out):
    """
    Converts a numpy array to an image and saves it as a PNG file.

    Args:
        filename (str): Name of the input numpy array file.
        dir_in (str): Directory path of the input numpy array file.
        dir_out (str): Directory path to save the output PNG file.
    """
    name, ext = os.path.splitext(filename)
    # Load the input numpy array
    img_arr = np.load(os.path.join(dir_in, filename))
    # Apply logarithm transformation
    img_log = np.log10(img_arr)
    # Get the dimensions of the image
    height, width = img_log.shape
    dpi = 125  # Dots per inch for the output image

    # Set the output file path
    out = os.path.join(dir_out, f'{name}.png')
    # Create a figure with the appropriate dimensions
    fig = plt.figure(figsize=(width / 100, height / 100))

    # Display the image
    plt.imshow(img_log, origin = 'lower')
    plt.axis('off')  # Turn off the axes
    plt.savefig(out, dpi=dpi, bbox_inches='tight', pad_inches=0.0)

    #close the image
    plt.close(fig)
    # Uncomment the line below if you want to save the cropped image array as a numpy file
    # np.save(out, np.array(img_))
    
 
# def tile(filename, dir_in, dir_out):
#     name, ext= os.path.splitext(filename)

#     img_arr = np.load(os.path.join(dir_in, filename))
#     img_log = np.log10(img_arr)

#     img = Image.fromarray(img_log)
#     w, h = img.size
#     h = int(h)
#     d = int(w / 4)  # Divide height by 4 to get the equal-sized parts~500

#     grid = [(0, j, j + d) for j in range(0, w - d + 1, d)]  # Generate grid coordinates
#     for i, j, k in grid:
#         box = (j, i, k, i + h)  # Use height as the top coordinate for cropping
#         out = os.path.join(dir_out, f'{name}_{i}_{j}.png')

#         img_ = img.crop(box)
#         plt.imshow(img_)
#         plt.savefig(out, bbox_inches = 'tight')
#         #np.save(out, np.array(img_))  # Save the cropped image array as numpy file


if __name__ == "__main__":
    for image in os.listdir(DIR_IN):
        convert_to_img(image, DIR_IN, DIR_OUT)

