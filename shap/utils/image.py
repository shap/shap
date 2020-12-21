import cv2
import json
import matplotlib.pyplot as plt 
import numpy as np
import os
import requests
from textwrap import wrap

def is_empty(path): 
    """
    Function to check if folder at given path exists and is not empty. Returns True if folder is empty or does not exist.
    """
    empty = False
    if os.path.exists(path) and not os.path.isfile(path): 
        # Checking if the directory is empty or not 
        if not os.listdir(path): 
            empty = True
            print("'test_images' folder is empty. Please place images to be tested in this folder.") 
    else: 
        empty = True
        print("There is no 'test_images' folder under current directory. Please create one and place images to be tested there.")
    return empty

def make_dir(path):
    """
    Function to create a new directory with given path or empty if it already exists.
    """
    if not os.path.exists(path):
        if not os.path.isfile(path): 
            # make directory if it does not exist
            os.makedirs(path)
        else:
            print("Please give a valid folder path.")
    else:
        # Check if empty or not 
        if os.listdir(path): 
            # if exists, empty directory
            for file in os.listdir(path):
                os.remove(path+file)

def load_image(path_to_image):
    """
    Function to load image at given path and return numpy array of RGB float values. 
    """
    image = cv2.imread(path_to_image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
    return np.array(image).astype('float')

def check_valid_image(path_to_image):
    """
    Function to check if a file has valid image extensions and return True if it does. 
    Note: Azure Cognitive Services only accepts below file formats.  
    """
    valid_extensions = ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.jfif']
    if path_to_image.endswith((tuple(valid_extensions))):
        return True 

def save_image(array, path_to_image):
    """
    Function to save image(RGB values array) at given path (filename and location).
    """
    # saving array of RGB values as an image 
    image = np.array(array)/255.0
    plt.imsave(path_to_image, image)


def resize_image(path_to_image, reshaped_dir):
    """
    Function to resize given image and save in given directory 'reshaped_dir'. 
    Returns numpy array of resized image and path where resized file is saved. 
    Note: Resizing is done for following purposes:
    1. Azure Cognitive Service does not run for large image sizes. File size should not be greater than 4MB and should have atleast 50x50 size. 
    2. Reducing image size speedens the SHAP explanation process.
    """
    image = load_image(path_to_image)
    
    # checking if either of (pixel_size, pixel_size) dimension is greater than 500. 
    reshaped_path = None
    x = 500 if image.shape[0] > 500 else image.shape[0]
    y = 500 if image.shape[1] > 500 else image.shape[1]

    # reshape image
    if x != image.shape[0] or y != image.shape[1]:
        image = cv2.resize(image, dsize = (x,y))
        reshaped_path = reshaped_dir + path_to_image.split("/")[-1] + ".png"
        save_image(image, reshaped_path)
        print("Reshaped image size:", image.shape)
        image = np.array(image).astype('float')
    
    return image, reshaped_path


def display_grid_plot(list_of_captions, list_of_images, max_columns=4, figsize=(20,20)):
    """
    Function to display grid of images and their titles/captions.
    """

    # load list of images
    masked_images = []
    for filename in  list_of_images:
        image = load_image(filename) 
        masked_images.append(image.astype(int))

    # display grid plot with wrapping
    fig = plt.figure(figsize=figsize)
    column = 0
    for i in range(len(masked_images)):
        column += 1
        #  check for end of column and create a new figure
        if column == max_columns+1:
            fig = plt.figure(figsize=figsize)
            column = 1
        fig.add_subplot(1, max_columns, column)
        plt.imshow(masked_images[i])
        plt.axis('off')
        if len(list_of_captions) >= len(masked_images):
            plt.title("\n".join(wrap(str(list_of_captions[i]), width = 40)))