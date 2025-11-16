import numpy as np
from PIL import Image
from image_processing import load_image_rgb, rgb_to_grayscale, resize_gray

path_gambar = "covers/1.jpg"   
rgb = load_image_rgb(path_gambar)
gray = rgb_to_grayscale(rgb)
img_gray = Image.fromarray(np.clip(gray, 0, 255).astype(np.uint8))
img_gray.save("test_output_gray.jpg")