import sys
import os.path
import math
import numpy as np

from PIL import Image, ImageDraw

MAX_WIDTH = 600
MAX_HEIGHT = 400

if len(sys.argv) < 2:
    print "Usage: python equalizer.py IMAGE_PATH"
    sys.exit(1)

image_path = sys.argv[1]
image_ext = os.path.splitext(image_path)[1]

if image_ext.lower() not in ['.jpg', '.jpeg', '.png', '.bmp']:
    print "Invalid image format. Use one of these format: .jpg, .jpeg, .png, .bmp"
    sys.exit(1)

try:
    base_image = Image.open(image_path)
    width, height = base_image.size
    if width > MAX_WIDTH:
        height, width = height * MAX_WIDTH / width, MAX_WIDTH
    if height > MAX_HEIGHT:
        width, height = width * MAX_HEIGHT / height, MAX_HEIGHT
    base_image = base_image.resize((width, height))
except Exception as e:
    print "Cannot open image"
    sys.exit(1)

def calculate_image_matrix_histogram(matrix):
    _, _, channel = matrix.shape
    flat_matrix = matrix.flatten()
    histogram = np.zeros((256, channel))
    for i, val in enumerate(flat_matrix):
        histogram[val][i % channel] += 1
    return histogram

def equalize_image_matrix(matrix):
    map_value_to_cumulative = {}
    
    height, width, channel = matrix.shape
    cumulative_value = np.zeros(channel)
    histogram = calculate_image_matrix_histogram(matrix)
    for value, freq in enumerate(histogram):
        cumulative_value = [cumulative_value[i] + freq[i] for i in range(channel)]
        map_value_to_cumulative[value] = cumulative_value

    total_frequency = width * height
    result = np.zeros(matrix.shape)
    for y in range(height):
        for x in range(width):
            initial_value = matrix[y][x]
            result[y][x] = [math.floor(float(map_value_to_cumulative[v][c]) / float(total_frequency) * 255) for c, v in enumerate(initial_value)]
    
    return result

result_matrix = equalize_image_matrix(np.array(base_image))
result_image = Image.fromarray(np.uint8(result_matrix))

base_image.show("Base Image")
result_image.show("Result Image")
