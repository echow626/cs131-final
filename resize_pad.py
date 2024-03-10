import os
import numpy as np
from PIL import Image
import concurrent.futures

def resize_and_pad(image_path, target_width, target_height, output_folder, filename, background_color=(0, 0, 0)):
    # Open the image
    image = Image.open(image_path)

    # Convert image to numpy array
    image_array = np.array(image)

    # Get original image dimensions
    original_height, original_width, _ = image_array.shape

    # Calculate aspect ratios
    aspect_ratio_x = target_width / original_width
    aspect_ratio_y = target_height / original_height

    # Determine scaling factor and resize the image
    scaling_factor = min(aspect_ratio_x, aspect_ratio_y)
    scaled_width = int(original_width * scaling_factor)
    scaled_height = int(original_height * scaling_factor)
    scaled_image = np.array(image.resize((scaled_width, scaled_height)))

    # Create a new array with the background color
    padded_image = np.full((target_height, target_width, 3), background_color, dtype=np.uint8)

    # Paste the resized image onto the padded image at the top left corner
    padded_image[:scaled_height, :scaled_width] = scaled_image

    output_path = os.path.join(output_folder, filename)
    Image.fromarray(padded_image).save(output_path)

def process_image(img):
    resize_and_pad('images/'+img, 1920, 1080, 'resized_images', img[:-4]+'_resized.png')

folder_path = os.path.join(os.getcwd(), 'images')
images = [filename for filename in os.listdir(folder_path) if filename != '.DS_Store']

with concurrent.futures.ThreadPoolExecutor() as executor:
    executor.map(lambda image_path: process_image(image_path), images)