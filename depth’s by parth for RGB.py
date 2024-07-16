import numpy as np
from PIL import Image

def calculate_depth(focal_length, real_height, screen_height):
    """
    Calculate the depth based on the screen height and the real height of the object.
    
    Parameters:
    focal_length (float): The focal length of the virtual camera.
    real_height (float): The real height of the object in mm.
    screen_height (float): The height of the object on the screen in pixels.
    
    Returns:
    float: The calculated depth in mm.
    """
    depth = (real_height * focal_length) / screen_height
    return depth

def rgb_offset_blur(image_array, depth, min_depth, max_depth):
    """
    Apply blur effect by shifting RGB channels based on depth.
    
    Parameters:
    image_array (ndarray): Input image as a numpy array.
    depth (float): The calculated depth.
    min_depth (float): The minimum depth for DoF effect.
    max_depth (float): The maximum depth for DoF effect.
    
    Returns:
    ndarray: Blurred image array.
    """
    # Normalize depth to a range between 0 and 1
    normalized_depth = (depth - min_depth) / (max_depth - min_depth)
    # Invert the normalized depth for DoF effect (closer objects have less blur)
    blur_factor = int(normalized_depth * 10)  # Scale the blur factor appropriately
    
    # Create an empty array for the output image
    output_image = np.zeros_like(image_array)

    # Shift the RGB channels
    output_image[blur_factor:, blur_factor:, 0] = image_array[:-blur_factor, :-blur_factor, 0]  # Red channel
    output_image[:-blur_factor, :-blur_factor, 1] = image_array[blur_factor:, blur_factor:, 1]  # Green channel
    output_image[:-blur_factor, blur_factor:, 2] = image_array[blur_factor:, :-blur_factor, 2]  # Blue channel

    return output_image

# Example parameters
focal_length = 50.0  # Focal length in mm
real_height = 170.0  # Real height of the object in mm (e.g., 170 mm for a known object)
screen_height = 200.0  # Height of the object on the screen in pixels
min_depth = 500.0  # Minimum depth in mm for DoF effect
max_depth = 3000.0  # Maximum depth in mm for DoF effect

# Load the image and convert to numpy array
image_path = 'path_to_image.jpg'
image = Image.open(image_path)
image_array = np.array(image)

# Calculate the depth
depth = calculate_depth(focal_length, real_height, screen_height)
print(f"Calculated Depth: {depth} mm")

# Apply the RGB offset blur based on depth
blurred_image_array = rgb_offset_blur(image_array, depth, min_depth, max_depth)

# Convert the numpy array back to an image
blurred_image = Image.fromarray(blurred_image_array)

# Save or display the blurred image
blurred_image.show()
blurred_image.save('blurred_image.jpg')